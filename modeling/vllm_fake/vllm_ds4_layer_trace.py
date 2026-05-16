import json
import os
import types

import torch

import fake_imple_patch
from fake_imple_patch import FakeTensorMode, to_fake_model, make_vllm_config, patch_deepseek_kernels

from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config
from vllm.config.vllm import set_current_vllm_config
from vllm.model_executor.models.deepseek_v4 import DeepseekV4DecoderLayer

from model_tracer import ModelTracer, _name_children

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../deepseek-ai/DeepSeek-V4-Pro")

with open(os.path.join(MODEL_DIR, "config.json")) as f:
    raw_config = json.load(f)

model_type = raw_config.get("model_type", "")

rope = raw_config.get("rope_scaling", {}) or {}
if "rope_type" not in rope:
    rope["rope_type"] = "default"
if "factor" not in rope:
    rope["factor"] = 1.0
raw_config["rope_parameters"] = rope
raw_config.setdefault("v_head_dim", 128)

config = DeepseekV4Config.from_dict(raw_config)

vllm_config = make_vllm_config(config, cache_dtype="fp8_ds_mla")
vllm_config.model_config.use_mla = True
vllm_config.model_config.max_model_len = 4096

patch_deepseek_kernels()

LAYER_PREFIX = "layers.4"

with set_current_vllm_config(vllm_config):
    layer = DeepseekV4DecoderLayer(
        vllm_config=vllm_config,
        prefix=LAYER_PREFIX,
    )

# ── layer-level patches (skip CUDA kernel ops, call nn.Module sub-layers) ──

# HC pre/post — CUDA kernel mhc_pre/mhc_post
def _fake_hc_pre(self_mod, x, hc_fn, hc_scale, hc_base):
    B, H, D = x.shape
    layer_input = x[:, 0, :].reshape(B, D)
    post_mix = x.new_empty(B, H, dtype=torch.float32)
    comb = x.new_empty(B, H * H, dtype=torch.float32)
    return layer_input, post_mix, comb


def _fake_hc_post(self_mod, x, residual, post, comb):
    return x.unsqueeze(1).expand(-1, self_mod.hc_mult, -1).contiguous()


layer.hc_pre = lambda x, fn, s, b, _l=layer: _fake_hc_pre(_l, x, fn, s, b)
layer.hc_post = lambda x, r, p, c, _l=layer: _fake_hc_post(_l, x, r, p, c)

# mla_attn — call sub-modules (fused_wqa_wkv, wo_b) so they appear in trace
_orig_mla_forward = layer.attn.mla_attn.forward


def _fake_mla_forward(self, positions, hidden_states, llama_4_scaling=None):
    qr_kv, _ = self.fused_wqa_wkv(hidden_states)
    num_tokens = hidden_states.shape[0]
    dummy = qr_kv.new_empty(num_tokens, self.n_local_groups * self.o_lora_rank)
    return self.wo_b(dummy)


layer.attn.mla_attn.forward = types.MethodType(_fake_mla_forward, layer.attn.mla_attn)

# MoE — gate is NOT called when is_internal_router=True;
# shared_experts is skipped by global _patched_moe_forward_shared.
# Call both explicitly to trace their sub-modules.
_orig_moe_forward = layer.ffn.forward


def _fake_moe_forward(self, hidden_states, input_ids=None):
    self.gate(hidden_states)
    if self.shared_experts is not None:
        self.shared_experts(hidden_states)
    return _orig_moe_forward(hidden_states, input_ids)


layer.ffn.forward = types.MethodType(_fake_moe_forward, layer.ffn)

# ── set up trace name hierarchy ──

layer._trace_name = LAYER_PREFIX
_name_children(layer, LAYER_PREFIX)

hidden_size = config.hidden_size
hc_mult = config.hc_mult
batch_size = 128

print(f"model Structure: {layer}")
with FakeTensorMode() as fake_mode, ModelTracer() as tracer:
    to_fake_model(layer, fake_mode)
    positions = fake_mode.from_tensor(torch.arange(batch_size, device="meta"))
    input_ids = fake_mode.from_tensor(torch.randint(0, 100, (batch_size,), device="meta"))
    hidden_states = fake_mode.from_tensor(
        torch.randn(batch_size, hc_mult, hidden_size, device="meta")
    )
    out = layer(hidden_states, positions, input_ids)

tracer.dump("ds4_layer_trace.json")

print(f"Config loaded from:       {MODEL_DIR}")
print(f"Model type:               {model_type}")
print(f"Hidden size:              {hidden_size}")
print(f"hc_mult:                  {hc_mult}")
print(f"Num layers in config:     {config.num_hidden_layers}")
print(f"Positions shape:          {positions.shape}")
print(f"Hidden states shape:      {hidden_states.shape}")
print(f"Output shape:             {out.shape}")
print(f"Decoder layer:            {type(layer).__name__}")
