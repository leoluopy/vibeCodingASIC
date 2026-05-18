import json
import os
import sys

import fake_imple_patch
import torch
import importlib
from transformers import AutoConfig, PretrainedConfig
from vllm.config.vllm import set_current_vllm_config
from fake_imple_patch import FakeTensorMode, to_fake_model, make_vllm_config
from vllm.outputs import RequestOutput, CompletionOutput

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.utils import random_uuid
from model_tracer import ModelTracer
import asyncio


_MODEL_MODULE_OVERRIDE = {
    "deepseek_v32": "deepseek_v2",
}

_MODEL_CLASS_OVERRIDE = {
    "deepseek_v4": "DeepseekV4ForCausalLM",
    "deepseek_v32": "DeepseekV2Model",
    "minimax_m2": "MiniMaxM2ForCausalLM",
    "kimi_k25": "KimiK25ForConditionalGeneration",
}

_CONFIG_FIELD_PATCHES = {
    "deepseek_v32": {
        "v_head_dim": 128,
        "rope_parameters": {"rope_type": "default", "factor": 1.0},
    },
}

_CONFIG_FIELD_REMOVE = {
    "deepseek_v32": {"index_topk"},
}


_CONFIG_CLASS_MAP: dict[str, str] = {
    "deepseek_v4": "vllm.transformers_utils.configs.deepseek_v4.DeepseekV4Config",
    "kimi_k25": "vllm.transformers_utils.configs.kimi_k25.KimiK25Config",
}


def _import_config_class(model_type: str):
    path = _CONFIG_CLASS_MAP.get(model_type)
    if path is None:
        return PretrainedConfig
    module_path, cls_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), cls_name)


def _load_hf_config(model_path: str):
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        from vllm.transformers_utils.config import patch_rope_parameters
        patch_rope_parameters(config)
        return config
    except (KeyError, ValueError, OSError):
        pass
    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        d = json.load(f)
    model_type = d.get("model_type", "")
    for k in _CONFIG_FIELD_REMOVE.get(model_type, set()):
        d.pop(k, None)
    patches = _CONFIG_FIELD_PATCHES.get(model_type, {})
    for k, v in patches.items():
        d.setdefault(k, v)
    config_cls = _import_config_class(model_type)
    config = config_cls.from_dict(d)
    from vllm.transformers_utils.config import patch_rope_parameters
    patch_rope_parameters(config)
    return config


def _get_root(model):
    """获取模型根模块（deepseek 有 .model 包装，其余直接是模型本身）"""
    return model.model if hasattr(model, "model") and hasattr(model.model, "layers") else model


def _patch_deepseek_layers(model, root, model_type):
    """替换 DeepSeek MLA/hc_pre/hc_post 为 shape-only 实现，跳过 CUDA kernel"""
    if "deepseek" not in model_type:
        return
    layers = root.layers if hasattr(root, "layers") else []
    for layer in layers:
        if hasattr(layer, "attn") and hasattr(layer.attn, "mla_attn"):
            layer.attn.mla_attn.forward = lambda pos, hs, sc=None: hs.new_empty(
                hs.shape[0], hs.shape[1], dtype=torch.bfloat16)

        if hasattr(layer, "hc_pre") and hasattr(layer, "hc_mult"):

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


@classmethod
def _fake_from_engine_args(cls, engine_args):
    engine = cls.__new__(cls)
    engine._engine_args = engine_args
    engine.engine_args = engine_args
    return engine


AsyncLLMEngine.from_engine_args = _fake_from_engine_args


async def _fake_generate(self, prompt, sampling_params, request_id):
    max_tokens = sampling_params.max_tokens

    hf_config = _load_hf_config(self._engine_args.model)
    model_type = hf_config.model_type

    vllm_module = _MODEL_MODULE_OVERRIDE.get(model_type, model_type)
    model_cls_name = _MODEL_CLASS_OVERRIDE.get(model_type, model_type.capitalize() + "Model")
    module = importlib.import_module(f"vllm.model_executor.models.{vllm_module}")
    model_cls = getattr(module, model_cls_name)

    # Patch DeepSeekV4 CUDA kernel functions before model init
    if "deepseek" in model_type:
        from fake_imple_patch import patch_deepseek_kernels
        patch_deepseek_kernels()

    cache_dtype = "fp8_ds_mla" if "deepseek" in model_type else "auto"
    vllm_config = make_vllm_config(hf_config, cache_dtype=cache_dtype)
    if "deepseek" in model_type:
        vllm_config.model_config.use_mla = True
    vllm_config.model_config.max_model_len = 4096
    vllm_config.model_config.hf_text_config = hf_config
    with set_current_vllm_config(vllm_config):
        model = model_cls(vllm_config=vllm_config)

    root = _get_root(model)
    _patch_deepseek_layers(model, root, model_type)

    print(f" ######## Model Structure: {model}")
    with FakeTensorMode() as fake_mode, ModelTracer() as tracer:
        to_fake_model(model, fake_mode)
        # Convert plain tensor attributes that to_fake_model misses
        for attr in ("_mtp_hidden_buffer", "topk_indices_buffer"):
            if hasattr(root, attr):
                b = getattr(root, attr)
                setattr(root, attr, fake_mode.from_tensor(
                    torch.empty(b.shape, dtype=b.dtype, device=b.device)
                ))

        input_ids = fake_mode.from_tensor(torch.randint(0, 100, (128,), device="meta"))
        positions = fake_mode.from_tensor(torch.arange(128, device="meta"))
        _ = model(input_ids, positions, intermediate_tensors=None)

    trace_name = model_type.replace("_", "-") + "_trace.json"
    tracer.dump(os.path.dirname(__file__) + "/" + trace_name)

    text_buffer = ""
    chars = "机器学习模型通过大量数据训练可以自动发现模式与规律"
    for i in range(max_tokens):
        await asyncio.sleep(0)
        text_buffer += chars[i % len(chars)]
        finished = i == max_tokens - 1
        yield RequestOutput(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=[],
            prompt_logprobs=None,
            outputs=[CompletionOutput(
                index=0, text=text_buffer, token_ids=[],
                cumulative_logprob=None, logprobs=None,
                finish_reason="length" if finished else None,
            )],
            finished=finished,
        )


AsyncLLMEngine.generate = _fake_generate


_MODEL_PATHS = {
    "qwen": os.path.dirname(__file__) + "/../Qwen/Qwen2.5-0.5B-Instruct",
    "deepseek_v3_2": os.path.dirname(__file__) + "/../deepseek-ai/DeepSeek-V3.2-Exp",
    "deepseek_v4": os.path.dirname(__file__) + "/../deepseek-ai/DeepSeek-V4-Pro",
    "minimax_m2": os.path.dirname(__file__) + "/../MiniMax/MiniMax-M2.7",
    "kimi_k25": os.path.dirname(__file__) + "/../moonshotai/Kimi-K2.6",
}


async def async_stream_generate(model_name: str = "deepseek_v3_2"):
        engine_args = AsyncEngineArgs(
            model=_MODEL_PATHS[model_name],
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            enable_chunked_prefill=True,
        )
        
        # 创建异步引擎
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        # 采样参数
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=15,
        )
        
        # 生成请求ID
        request_id = random_uuid()
        
        # 正确的流式输出方式
        final_text = ""
        
        # 开始生成
        results_generator = engine.generate(
            prompt="写一个关于机器学习的短故事",
            sampling_params=sampling_params,
            request_id=request_id
        )
        
        # 逐token获取结果
        async for request_output in results_generator:
            # 检查是否完成
            if not request_output.finished:
                # 获取新生成的文本
                new_text = request_output.outputs[0].text
                # 计算增量
                delta = new_text[len(final_text):]
                final_text = new_text
                
                # 输出增量内容
                if delta:
                    print(delta, end="", flush=True)
            else:
                break
        
        print()  # 最终换行

# 运行
if __name__ == "__main__":
    for model_name in _MODEL_PATHS:
        asyncio.run(async_stream_generate(model_name))
    