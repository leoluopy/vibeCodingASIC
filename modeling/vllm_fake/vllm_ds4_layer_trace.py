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

# ── helpers ──────────────────────────────────────────────────────────

def _patch_complex_modules(root):
    """Pre-patch known CUDA-kernel modules so their forward succeeds in
    fake mode.  The call still goes through ``nn.Module.__call__`` and
    gets traced by ``ModelTracer``."""
    for mod in root.modules():
        cls = type(mod).__name__

        if cls == "DeepseekV4SWACache":
            mod.forward = lambda: None
        elif cls == "DeepseekV4IndexerCache":
            mod.forward = lambda: None
        elif cls == "CompressorStateCache":
            mod.forward = lambda: None
        elif cls == "SparseAttnIndexer":
            mod.forward = lambda *a, **kw: None
        elif cls == "QuantFP8":
            def _qfp8(self, x, *a, **kw):
                return (torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device),
                        torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device))
            mod.forward = types.MethodType(_qfp8, mod)
        elif cls == "ApplyRotaryEmb":
            mod.forward = types.MethodType(
                lambda self, x, cos, sin: torch.empty_like(x), mod)
        elif cls == "DeepseekV4Indexer":
            def _idx(self, hs, qr, pos, rot):
                return hs.new_empty(hs.shape[0], self.n_head, self.head_dim)
            mod.forward = types.MethodType(_idx, mod)
        elif cls == "DeepseekCompressor":
            mod.forward = types.MethodType(lambda self, x, pos, rot: None, mod)
        elif cls == "DeepseekV4MLAAttention":
            mod.forward = types.MethodType(lambda self, q, kv, pos, out: None, mod)


# ── layer-level patches (skip CUDA ops, call nn.Module sub-layers) ──

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


_patch_complex_modules(layer)


# mla_attn forward — call ALL sub-modules so they appear in trace
def _fake_mla_forward(self, positions, hidden_states, llama_4_scaling=None):
    B = hidden_states.shape[0]
    ref = hidden_states

    qr_kv, _ = self.fused_wqa_wkv(hidden_states)
    qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)

    # ── mla_attn direct sub-modules ──
    self.q_norm(qr)
    self.wq_b(qr)
    self.kv_norm(kv)
    self.q_head_norm(kv)

    wo_sz = self.n_local_heads * self.head_dim // max(self.n_local_groups, 1)
    self.wo_a(ref.new_empty(B, wo_sz))

    if hasattr(self, "_wo_a_act_quant"):
        self._wo_a_act_quant(ref.new_empty(B, self.n_local_groups, self.o_lora_rank))

    # rotary_emb + its apply_rotary_emb sub-module
    rq = ref.new_empty(B, self.n_local_heads, self.head_dim)
    rc = ref.new_empty(B, self.head_dim)
    rs = ref.new_empty(B, self.head_dim)
    self.rotary_emb(positions, rq)
    self.rotary_emb.apply_rotary_emb(rq, rc, rs)

    # indexer_rotary_emb (same object as rotary_emb for V4-Pro)
    if hasattr(self, "indexer_rotary_emb"):
        self.indexer_rotary_emb(positions, rq)

    # swa_cache_layer
    if hasattr(self, "swa_cache_layer"):
        self.swa_cache_layer()

    # indexer + all sub-modules
    if hasattr(self, "indexer") and self.indexer is not None:
        idx = self.indexer
        idx.wq_b(qr)
        idx.weights_proj(hidden_states)
        nd = idx.k_norm.weight.shape[-1] if idx.k_norm.weight is not None else 128
        idx.k_norm(ref.new_empty(B, nd))

        if hasattr(idx, "k_cache"):
            idx.k_cache()

        comp = idx.compressor
        comp.fused_wkv_wgate(hidden_states)
        if hasattr(comp, "norm") and comp.norm is not None and comp.norm.weight is not None:
            comp.norm(ref.new_empty(B, comp.norm.weight.shape[-1]))
        if hasattr(comp, "state_cache"):
            comp.state_cache()
        comp(hidden_states, positions, self.rotary_emb)

        if hasattr(idx, "indexer_op"):
            if hasattr(idx.indexer_op, "k_cache"):
                idx.indexer_op.k_cache()
            idx.indexer_op(hidden_states, None, None, None)

        idx(hidden_states, qr, positions, self.rotary_emb)

    # mla_attn's own compressor
    if hasattr(self, "compressor") and self.compressor is not None:
        comp = self.compressor
        comp.fused_wkv_wgate(hidden_states)
        if hasattr(comp, "norm") and comp.norm is not None and comp.norm.weight is not None:
            comp.norm(ref.new_empty(B, comp.norm.weight.shape[-1]))
        if hasattr(comp, "state_cache"):
            comp.state_cache()
        comp(hidden_states, positions, self.rotary_emb)

    # mla_attn (DeepseekV4MLAAttention) — indexer & swa same objects as above
    mla = self.mla_attn
    dummy_o = ref.new_empty(B, self.n_local_heads, self.head_dim)
    mla(qr, kv, positions, dummy_o)

    # wo_b
    dummy = ref.new_empty(B, self.n_local_groups * self.o_lora_rank)
    return self.wo_b(dummy)


layer.attn.mla_attn.forward = types.MethodType(_fake_mla_forward, layer.attn.mla_attn)


# MoE — gate NOT called when is_internal_router=True;
# shared_experts skipped by global _patched_moe_forward_shared.
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
