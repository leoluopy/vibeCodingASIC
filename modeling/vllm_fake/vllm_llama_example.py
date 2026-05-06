"""使用 vLLM 的 LlamaModel 直接构建完整 Llama 模型推理示例（CPU + FakeTensor）

所有组件均直接 import，无需自定义实现。
"""

import torch

import fake_imple_patch
from fake_imple_patch import FakeTensorMode, to_fake_model, make_vllm_config

from transformers import LlamaConfig
from vllm.config.vllm import set_current_vllm_config
from vllm.model_executor.models.llama import LlamaModel


# ── 模型超参 ──
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 11008
NUM_LAYERS = 2
NUM_HEADS = 32
VOCAB_SIZE = 32000

llama_config = LlamaConfig(
    hidden_size=HIDDEN_SIZE,
    intermediate_size=INTERMEDIATE_SIZE,
    num_hidden_layers=NUM_LAYERS,
    num_attention_heads=NUM_HEADS,
    num_key_value_heads=NUM_HEADS,
    vocab_size=VOCAB_SIZE,
    hidden_act="silu",
    rms_norm_eps=1e-6,
)

# ── 初始化 vLLM 原生 LlamaModel ──
vllm_config = make_vllm_config(llama_config)

with set_current_vllm_config(vllm_config):
    model = LlamaModel(vllm_config=vllm_config)

# ── FakeTensor 模式推理（无真实计算，使用 flat tokens 如 vLLM） ──
with FakeTensorMode() as fake_mode:
    to_fake_model(model, fake_mode)
    input_ids = fake_mode.from_tensor(torch.randint(0, 100, (128,), device="meta"))
    positions = fake_mode.from_tensor(torch.arange(128, device="meta"))
    out = model(input_ids, positions, intermediate_tensors=None)

print(f"Input shape:          {input_ids.shape}")
print(f"Positions shape:      {positions.shape}")
print(f"Output shape:         {out.shape}")
print(f"Output dtype:         {out.dtype}")
print(f"Output:               {out}")
print(f"Num decoder layers:   {len(model.layers)}")
print(f"Hidden size:          {HIDDEN_SIZE}")
