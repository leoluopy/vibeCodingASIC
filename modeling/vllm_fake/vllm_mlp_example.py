"""使用 vLLM 的 LlamaMLP 进行推理的示例（CPU + FakeTensor）"""

import torch
import fake_imple_patch

from vllm.model_executor.models.llama import LlamaMLP

# ── 初始化 MLP ──
with fake_imple_patch.vllm_config_ctx():
    mlp = LlamaMLP(
        hidden_size=4096,
        intermediate_size=11008,
        hidden_act="silu",
        bias=False,
        disable_tp=True,
        reduce_results=False,
    )

# ── FakeTensor 模式推理（无真实计算） ──
with fake_imple_patch.FakeTensorMode() as fake_mode:
    fake_imple_patch.to_fake_model(mlp, fake_mode)
    x = fake_mode.from_tensor(torch.randn(2, 4, 4096, device="meta"))
    out = mlp(x)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {out.shape}")
print(f"Output dtype: {out.dtype}")
print(f"Output:       {out}")
