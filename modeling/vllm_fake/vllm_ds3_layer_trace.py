import json
import os

import torch

import fake_imple_patch
from fake_imple_patch import FakeTensorMode, to_fake_model, make_vllm_config

from transformers import PretrainedConfig
from vllm.config.vllm import set_current_vllm_config
from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer

from model_tracer import ModelTracer

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../deepseek-ai/DeepSeek-V3.2-Exp")

with open(os.path.join(MODEL_DIR, "config.json")) as f:
    raw_config = json.load(f)

model_type = raw_config.get("model_type", "")
raw_config.pop("index_topk", None)

if model_type == "deepseek_v32":
    raw_config.setdefault("rope_parameters", {"rope_type": "default", "factor": 1.0})

config = PretrainedConfig.from_dict(raw_config)

vllm_config = make_vllm_config(config, cache_dtype="fp8_ds_mla")
vllm_config.model_config.use_mla = True
vllm_config.model_config.max_model_len = 4096

with set_current_vllm_config(vllm_config):
    layer = DeepseekV2DecoderLayer(
        vllm_config=vllm_config,
        prefix="layers.4",
    )

hidden_size = config.hidden_size
print(f"model Structure: {layer}")
with FakeTensorMode() as fake_mode, ModelTracer() as tracer:
    to_fake_model(layer, fake_mode)
    positions = fake_mode.from_tensor(torch.arange(128, device="meta"))
    hidden_states = fake_mode.from_tensor(
        torch.randn(128, hidden_size, device="meta")
    )
    out = layer(positions, hidden_states, residual=None)

tracer.dump("ds3_layer_trace.json")

print(f"Config loaded from:       {MODEL_DIR}")
print(f"Model type:               {model_type}")
print(f"Hidden size:              {hidden_size}")
print(f"Num layers in config:     {config.num_hidden_layers}")
print(f"Positions shape:          {positions.shape}")
print(f"Hidden states shape:      {hidden_states.shape}")
print(f"Output hidden_states:     {out[0].shape}")
print(f"Output residual:          {out[1].shape}")
print(f"Decoder layer:            {type(layer).__name__}")

# model structure
# model Structure: DeepseekV2DecoderLayer(
#   (self_attn): DeepseekV2MLAAttention(
#     (fused_qkv_a_proj): DeepSeekV2FusedQkvAProjLinear(in_features=7168, output_features=2112, bias=False, tp_size=1, gather_output=False)
#     (q_a_layernorm): RMSNorm(hidden_size=1536, eps=1e-06)
#     (q_b_proj): ColumnParallelLinear(in_features=1536, output_features=24576, bias=False, tp_size=1, gather_output=False)
#     (kv_a_layernorm): RMSNorm(hidden_size=512, eps=1e-06)
#     (kv_b_proj): ColumnParallelLinear(in_features=512, output_features=32768, bias=False, tp_size=1, gather_output=False)
#     (o_proj): RowParallelLinear(in_features=16384, output_features=7168, bias=False, tp_size=1, reduce_results=True)
#     (rotary_emb): RotaryEmbedding(
#       head_size=64, rotary_dim=64, max_position_embeddings=163840, base=10000, is_neox_style=False
#       (apply_rotary_emb): ApplyRotaryEmb(is_neox_style=False, enable_fp32_compute=False)
#     )
#     (mla_attn): MultiHeadLatentAttentionWrapper(
#       (fused_qkv_a_proj): DeepSeekV2FusedQkvAProjLinear(in_features=7168, output_features=2112, bias=False, tp_size=1, gather_output=False)
#       (q_a_layernorm): RMSNorm(hidden_size=1536, eps=1e-06)
#       (q_b_proj): ColumnParallelLinear(in_features=1536, output_features=24576, bias=False, tp_size=1, gather_output=False)
#       (kv_a_layernorm): RMSNorm(hidden_size=512, eps=1e-06)
#       (kv_b_proj): ColumnParallelLinear(in_features=512, output_features=32768, bias=False, tp_size=1, gather_output=False)
#       (rotary_emb): RotaryEmbedding(
#         head_size=64, rotary_dim=64, max_position_embeddings=163840, base=10000, is_neox_style=False
#         (apply_rotary_emb): ApplyRotaryEmb(is_neox_style=False, enable_fp32_compute=False)
#       )
#       (o_proj): RowParallelLinear(in_features=16384, output_features=7168, bias=False, tp_size=1, reduce_results=True)
#       (mla_attn): MLAAttention(
#         (kv_b_proj): ColumnParallelLinear(in_features=512, output_features=32768, bias=False, tp_size=1, gather_output=False)
#         (_decode_concat_quant_fp8_op): _DecodeConcatQuantFP8()
#         (_quant_fp8_op): QuantFP8()
#       )
#     )
#   )
#   (mlp): DeepseekV2MoE(
#     (gate): GateLinear(in_features=7168, output_features=256, bias=False)
#     (shared_experts): DeepseekV2MLP(
#       (gate_up_proj): MergedColumnParallelLinear(in_features=7168, output_features=4096, bias=False, tp_size=1, gather_output=False)
#       (down_proj): RowParallelLinear(in_features=2048, output_features=7168, bias=False, tp_size=1, reduce_results=False)
#       (act_fn): SiluAndMul()
#     )
#     (experts): FusedMoE(
#       global_num_experts=256, local_num_experts=256, top_k=8, intermediate_size_per_partition=2048, tp_size=1,
#       ep_size=1, 
#       (quant_method): UnquantizedFusedMoEMethod()
#       (base_quant_method): UnquantizedFusedMoEMethod()
#     )
#   )
#   (input_layernorm): RMSNorm(hidden_size=7168, eps=1e-06)
#   (post_attention_layernorm): RMSNorm(hidden_size=7168, eps=1e-06)
# )
# Config loaded from:       /media/leo/work/mvp/vibeCodingASIC/modeling/vllm_fake/../deepseek-ai/DeepSeek-V3.2-Exp