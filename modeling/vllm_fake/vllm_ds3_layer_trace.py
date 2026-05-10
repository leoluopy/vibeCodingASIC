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
