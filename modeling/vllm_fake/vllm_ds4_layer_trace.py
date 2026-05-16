import json
import os

import torch

import fake_imple_patch
from fake_imple_patch import FakeTensorMode, to_fake_model, make_vllm_config, patch_deepseek_kernels

from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config
from vllm.config.vllm import set_current_vllm_config
from vllm.model_executor.models.deepseek_v4 import DeepseekV4DecoderLayer

from model_tracer import ModelTracer

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../deepseek-ai/DeepSeek-V4-Pro")

with open(os.path.join(MODEL_DIR, "config.json")) as f:
    raw_config = json.load(f)

model_type = raw_config.get("model_type", "")
# Keep index_topk — the indexer reads it during __init__
# but the fake forward will skip actual indexing

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

with set_current_vllm_config(vllm_config):
    layer = DeepseekV4DecoderLayer(
        vllm_config=vllm_config,
        prefix="layers.4",
    )


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

# Patch attention forward to skip CUDA kernel ops
layer.attn.mla_attn.forward = lambda pos, hs, sc=None: hs.new_empty(
    hs.shape[0], hs.shape[1], dtype=hs.dtype
)

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
