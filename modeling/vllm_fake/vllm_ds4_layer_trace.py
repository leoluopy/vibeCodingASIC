import json
import os
import types

import torch
import torch.nn as nn

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

def _convert_params_to_bf16(module):
    """Convert all parameters/buffers to bfloat16 (matching config
    torch_dtype), except those explicitly set to float32 (hc fn/scale/base)."""
    for m in module.modules():
        for name, p in list(m._parameters.items()):
            if p is not None and p.dtype == torch.float32 and "hc_" not in name:
                m._parameters[name] = torch.nn.Parameter(
                    p.to(torch.bfloat16), requires_grad=p.requires_grad)
        for name, b in list(m._buffers.items()):
            if b is not None and b.dtype == torch.float32 and not name.endswith("_numel"):
                m._buffers[name] = b.to(torch.bfloat16)


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
            def _sparse_idx(self, hs, q_quant, k, weights):
                topk = getattr(self, "topk", 1024)
                return hs.new_empty(hs.shape[0], topk, dtype=torch.int32)
            mod.forward = types.MethodType(_sparse_idx, mod)
        elif cls == "QuantFP8":
            def _qfp8(self, x, *a, **kw):
                return (torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device),
                        torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device))
            mod.forward = types.MethodType(_qfp8, mod)
        elif cls == "RotaryEmbedding":
            # Swap arg order: forward(query, positions, key=None)
            # so tracer captures query tensor as input (not positions).
            def _rotary(self, query, positions, key=None):
                return query, key
            mod.forward = types.MethodType(_rotary, mod)
        elif cls == "ApplyRotaryEmb":
            mod.forward = types.MethodType(
                lambda self, x, cos, sin: torch.empty_like(x), mod)
        elif cls == "DeepseekV4Indexer":
            def _idx(self, hs, qr, pos, rot):
                itopk = getattr(self, "index_topk", 1024)
                return hs.new_empty(hs.shape[0], itopk, dtype=torch.int32)
            mod.forward = types.MethodType(_idx, mod)
        elif cls == "DeepseekCompressor":
            mod.forward = types.MethodType(lambda self, x, pos, rot: None, mod)
        elif cls == "DeepseekV4MLAAttention":
            mod.forward = types.MethodType(lambda self, q, kv, pos, out: out, mod)


# ── layer-level patches (skip CUDA ops, call nn.Module sub-layers) ──

# Wrap HC pre/post as nn.Module so the tracer captures their multi-tensor I/O.
# Real signature: hc_pre(x: [B,H,D], fn, scale, base) → (layer_input:[B,D], post_mix:[B,H], comb:[B,H*H])

class _TracedHCPre(nn.Module):
    def forward(self, x, hc_fn, hc_scale, hc_base):
        B, H, D = x.shape
        layer_input = x[:, 0, :].reshape(B, D)
        post_mix = x.new_empty(B, H, dtype=x.dtype)
        comb = x.new_empty(B, H * H, dtype=x.dtype)
        return layer_input, post_mix, comb


class _TracedHCPost(nn.Module):
    def __init__(self, hc_mult: int):
        super().__init__()
        self.hc_mult = hc_mult

    def forward(self, x, residual, post, comb):
        return x.unsqueeze(1).expand(-1, self.hc_mult, -1).contiguous()


# nn.Module.__setattr__ registers in _modules but NOT __dict__,
# so class method would shadow it.  Force __dict__ entry.
layer.hc_pre = _TracedHCPre()
layer.__dict__['hc_pre'] = layer._modules['hc_pre']
layer.hc_post = _TracedHCPost(layer.hc_mult)
layer.__dict__['hc_post'] = layer._modules['hc_post']


_patch_complex_modules(layer)

# Convert parameters to bf16 to match config torch_dtype: bfloat16.
# The hc_* parameters are explicitly float32 in __init__ — skip them.
_convert_params_to_bf16(layer)

# ── attention forward ─────────────────────────────────────────────────
# We swap arg order: forward(hidden_states, positions) instead of
# forward(positions, hidden_states) so the tracer captures the correct
# hidden_states shape for DeepseekV4Attention / Wrapper traces.

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
    # NOTE: call with (query, positions) so tracer captures query shape
    rq = ref.new_empty(B, self.n_local_heads, self.head_dim)
    rc = ref.new_empty(B, self.head_dim)
    rs = ref.new_empty(B, self.head_dim)
    self.rotary_emb(rq, positions)
    self.rotary_emb.apply_rotary_emb(rq, rc, rs)

    # indexer_rotary_emb (same object as rotary_emb for V4-Pro)
    if hasattr(self, "indexer_rotary_emb"):
        self.indexer_rotary_emb(rq, positions)

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
            q_quant = ref.new_empty(B, idx.n_head, idx.head_dim)
            weights = ref.new_empty(B, idx.n_head)
            idx.indexer_op(hidden_states, q_quant, None, weights)

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


# ── arg-swapping wrappers so tracer captures hidden_states shape ────
# DeepseekV4Attention forward signature: (positions, hidden_states)
# We want attn(hidden_states, positions) → tracer sees (B, D) not (B,)

# First set _fake_mla_forward as mla_attn.forward (replaces original)
layer.attn.mla_attn.forward = types.MethodType(
    _fake_mla_forward, layer.attn.mla_attn)

# 1) Wrap mla_attn.forward: _fake_mla_forward takes (positions, hidden_states)
#    We make mla_attn(hidden_states, positions) → _fake_mla_forward(positions, hidden_states)
_fake_mla_bound = layer.attn.mla_attn.forward  # MethodType bound func

def _mla_swapped(self, hidden_states, positions):
    return _fake_mla_bound(positions, hidden_states)

layer.attn.mla_attn.forward = types.MethodType(_mla_swapped, layer.attn.mla_attn)

# 2) Wrap DeepseekV4Attention.forward: let it accept (hidden_states, positions)
#    and delegate to mla_attn with same order (propagating correct arg order)
def _attn_swapped(self, hidden_states, positions):
    return self.mla_attn(hidden_states, positions)

layer.attn.forward = types.MethodType(_attn_swapped, layer.attn)

# 3) Custom decoder-layer forward: calls attn(hidden_states, positions)
_orig_layer_forward = layer.forward


def _custom_layer_forward(self, hidden_states, positions, input_ids=None):
    B, H, D = hidden_states.shape

    # hc_pre — the actual fn/scale/base values are ignored by _fake_hc_pre
    hs, post_mix, comb = self.hc_pre(
        hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
    )

    # attn_norm
    hs = self.attn_norm(hs)

    # attention with hidden_states FIRST so tracer captures (B, D)
    hs = self.attn(hs, positions)

    # hc_post
    hs = self.hc_post(hs, None, post_mix, comb)

    # 1st residual — hc_post already expanded to (B, H, D)
    hidden_states = hidden_states + hs

    # ffn_norm
    hs = self.ffn_norm(hidden_states)
    hs_flat = hs[:, 0, :].contiguous() if H > 1 else hs

    # ffn (MoE)
    hs_flat = self.ffn(hs_flat, input_ids)
    hs = hs_flat.unsqueeze(1).expand(-1, H, -1).contiguous() if H > 1 else hs_flat

    # 2nd residual
    hidden_states = hidden_states + hs

    return hidden_states


layer.forward = types.MethodType(_custom_layer_forward, layer)

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
batch_size = 6
dtype = torch.bfloat16

print(f"model Structure: {layer}")
with FakeTensorMode() as fake_mode, ModelTracer() as tracer:
    to_fake_model(layer, fake_mode)
    positions = fake_mode.from_tensor(torch.arange(batch_size, device="meta"))
    input_ids = fake_mode.from_tensor(torch.randint(0, 100, (batch_size,), device="meta"))
    hidden_states = fake_mode.from_tensor(
        torch.empty(batch_size, hc_mult, hidden_size, device="meta", dtype=dtype)
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

# model Structure: DeepseekV4DecoderLayer(
#   (attn): DeepseekV4Attention(
#     (fused_wqa_wkv): MergedColumnParallelLinear(in_features=7168, output_features=2048, bias=False, tp_size=1, gather_output=False)
#     (q_norm): RMSNorm(hidden_size=1536, eps=1e-06)
#     (wq_b): ColumnParallelLinear(in_features=1536, output_features=65536, bias=False, tp_size=1, gather_output=False)
#     (kv_norm): RMSNorm(hidden_size=512, eps=1e-06)
#     (wo_a): ColumnParallelLinear(in_features=4096, output_features=16384, bias=False, tp_size=1, gather_output=False)
#     (wo_b): RowParallelLinear(in_features=16384, output_features=7168, bias=False, tp_size=1, reduce_results=True)
#     (rotary_emb): RotaryEmbedding(
#       head_size=512, rotary_dim=64, max_position_embeddings=1048576, base=160000, is_neox_style=False
#       (apply_rotary_emb): ApplyRotaryEmb(is_neox_style=False, enable_fp32_compute=False)
#     )
#     (indexer): DeepseekV4Indexer(
#       (wq_b): ReplicatedLinear(in_features=1536, output_features=8192, bias=False)
#       (weights_proj): ReplicatedLinear(in_features=7168, output_features=64, bias=False)
#       (k_norm): LayerNorm()
#       (k_cache): DeepseekV4IndexerCache()
#       (compressor): DeepseekCompressor(
#         (fused_wkv_wgate): MergedColumnParallelLinear(in_features=7168, output_features=512, bias=False, tp_size=1, gather_output=False)
#         (norm): RMSNorm(hidden_size=128, eps=1e-06)
#         (state_cache): CompressorStateCache()
#       )
#       (indexer_op): SparseAttnIndexer(
#         (k_cache): DeepseekV4IndexerCache()
#       )
#     )
#     (mla_attn): DeepseekV4MultiHeadLatentAttentionWrapper(
#       (fused_wqa_wkv): MergedColumnParallelLinear(in_features=7168, output_features=2048, bias=False, tp_size=1, gather_output=False)
#       (q_norm): RMSNorm(hidden_size=1536, eps=1e-06)
#       (wq_b): ColumnParallelLinear(in_features=1536, output_features=65536, bias=False, tp_size=1, gather_output=False)
#       (kv_norm): RMSNorm(hidden_size=512, eps=1e-06)
#       (wo_a): ColumnParallelLinear(in_features=4096, output_features=16384, bias=False, tp_size=1, gather_output=False)
#       (_wo_a_act_quant): QuantFP8()
#       (wo_b): RowParallelLinear(in_features=16384, output_features=7168, bias=False, tp_size=1, reduce_results=True)
#       (rotary_emb): RotaryEmbedding(
#         head_size=512, rotary_dim=64, max_position_embeddings=1048576, base=160000, is_neox_style=False
#         (apply_rotary_emb): ApplyRotaryEmb(is_neox_style=False, enable_fp32_compute=False)
#       )
#       (indexer_rotary_emb): RotaryEmbedding(
#         head_size=512, rotary_dim=64, max_position_embeddings=1048576, base=160000, is_neox_style=False
#         (apply_rotary_emb): ApplyRotaryEmb(is_neox_style=False, enable_fp32_compute=False)
#       )
#       (indexer): DeepseekV4Indexer(
#         (wq_b): ReplicatedLinear(in_features=1536, output_features=8192, bias=False)
#         (weights_proj): ReplicatedLinear(in_features=7168, output_features=64, bias=False)
#         (k_norm): LayerNorm()
#         (k_cache): DeepseekV4IndexerCache()
#         (compressor): DeepseekCompressor(
#           (fused_wkv_wgate): MergedColumnParallelLinear(in_features=7168, output_features=512, bias=False, tp_size=1, gather_output=False)
#           (norm): RMSNorm(hidden_size=128, eps=1e-06)
#           (state_cache): CompressorStateCache()
#         )
#         (indexer_op): SparseAttnIndexer(
#           (k_cache): DeepseekV4IndexerCache()
#         )
#       )
#       (q_head_norm): RMSNorm(hidden_size=512, eps=1e-06)
#       (swa_cache_layer): DeepseekV4SWACache()
#       (mla_attn): DeepseekV4MLAAttention(
#         (indexer): DeepseekV4Indexer(
#           (wq_b): ReplicatedLinear(in_features=1536, output_features=8192, bias=False)
#           (weights_proj): ReplicatedLinear(in_features=7168, output_features=64, bias=False)
#           (k_norm): LayerNorm()
#           (k_cache): DeepseekV4IndexerCache()
#           (compressor): DeepseekCompressor(
#             (fused_wkv_wgate): MergedColumnParallelLinear(in_features=7168, output_features=512, bias=False, tp_size=1, gather_output=False)
#             (norm): RMSNorm(hidden_size=128, eps=1e-06)
#             (state_cache): CompressorStateCache()
#           )
#           (indexer_op): SparseAttnIndexer(
#             (k_cache): DeepseekV4IndexerCache()
#           )
#         )
#         (swa_cache_layer): DeepseekV4SWACache()
#       )
#       (compressor): DeepseekCompressor(
#         (fused_wkv_wgate): MergedColumnParallelLinear(in_features=7168, output_features=2048, bias=False, tp_size=1, gather_output=False)
#         (norm): RMSNorm(hidden_size=512, eps=1e-06)
#         (state_cache): CompressorStateCache()
#       )
#     )
#   )
#   (ffn): DeepseekV4MoE(
#     (gate): GateLinear(in_features=7168, output_features=384, bias=False)
#     (shared_experts): DeepseekV4MLP(
#       (gate_up_proj): MergedColumnParallelLinear(in_features=7168, output_features=6144, bias=False, tp_size=1, gather_output=False)
#       (down_proj): RowParallelLinear(in_features=3072, output_features=7168, bias=False, tp_size=1, reduce_results=False)
#       (act_fn): SiluAndMulWithClamp()
#     )
#     (experts): FusedMoE(
#       global_num_experts=384, local_num_experts=384, top_k=6, intermediate_size_per_partition=3072, tp_size=1,
#       ep_size=1, 
#       (quant_method): UnquantizedFusedMoEMethod()
#       (base_quant_method): UnquantizedFusedMoEMethod()
#     )
#   )
#   (attn_norm): RMSNorm(hidden_size=7168, eps=1e-06)
#   (ffn_norm): RMSNorm(hidden_size=7168, eps=1e-06)
# )
# Config loaded from:       /media/leo/work/mvp/vibeCodingASIC/modeling/vllm_fake/../deepseek-ai/DeepSeek-V4-Pro