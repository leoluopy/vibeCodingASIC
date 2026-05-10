"""vLLM CPU/无GPU环境统一补丁包。import 即自动应用所有补丁。"""

import os
import typing
import types as pytypes
from collections.abc import Callable
from unittest.mock import patch
from contextlib import contextmanager

os.environ.setdefault("VLLM_NO_IMPORT_WARNING", "1")
os.environ.setdefault("TRITON_INTERPRET", "1")

import torch._dynamo
torch._dynamo.config.disable = True

import torch
import torch.nn as nn
import torch.library
import torch._library.infer_schema as infer_schema_mod
from torch._subclasses.fake_tensor import FakeTensorMode

# ═══════════════════════════════════════════════
# PyTorch 2.6 + vLLM 0.18 类型标注兼容
# ═══════════════════════════════════════════════

for ann, schema in list(infer_schema_mod.SUPPORTED_PARAM_TYPES.items()):
    origin = typing.get_origin(ann)
    args = typing.get_args(ann)
    if origin is list:
        infer_schema_mod.SUPPORTED_PARAM_TYPES[list[args[0]]] = schema

_orig_infer = infer_schema_mod.infer_schema


def _normalize(ann):
    if isinstance(ann, str):
        return ann
    if isinstance(ann, pytypes.UnionType):
        args = typing.get_args(ann)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return typing.Optional[_normalize(non_none[0])]
    origin = typing.get_origin(ann)
    args = typing.get_args(ann)
    if origin is list:
        return typing.List[args[0]] if args else typing.List
    if origin is tuple:
        return typing.Tuple[args] if args else typing.Tuple
    if origin is dict:
        return typing.Dict[args] if args else typing.Dict
    return ann


def _patched_infer(prototype_function, /, *, mutates_args, op_name=None):
    import inspect
    import functools

    sig = inspect.signature(prototype_function)
    new_params = []
    for name, param in sig.parameters.items():
        ann = param.annotation
        if ann is not inspect.Parameter.empty:
            ann = _normalize(ann)
            if isinstance(ann, str):
                try:
                    ann = eval(ann)
                except Exception:
                    pass
        if ann is not param.annotation:
            new_params.append(param.replace(annotation=ann))
        else:
            new_params.append(param)
    ret = sig.return_annotation
    if ret is not inspect.Parameter.empty:
        ret = _normalize(ret)
        if isinstance(ret, str):
            try:
                ret = eval(ret)
            except Exception:
                pass

    new_sig = sig.replace(parameters=new_params, return_annotation=ret)

    def wrapper(*a, **kw):
        return prototype_function(*a, **kw)

    wrapper.__signature__ = new_sig
    functools.update_wrapper(wrapper, prototype_function)
    return _orig_infer(wrapper, mutates_args=mutates_args, op_name=op_name)


infer_schema_mod.infer_schema = _patched_infer
torch.library.infer_schema = _patched_infer

# ═══════════════════════════════════════════════
# Triton AttrsDescriptor 补丁
# ═══════════════════════════════════════════════

import triton.compiler.compiler as _tcc
import triton.backends.compiler as _tbc

if not hasattr(_tcc, "AttrsDescriptor"):
    class _AttrsDescriptor:
        pass
    _tcc.AttrsDescriptor = _AttrsDescriptor
    _tbc.AttrsDescriptor = _AttrsDescriptor

# ═══════════════════════════════════════════════
# Mock 分布式通信
# ═══════════════════════════════════════════════

import vllm.distributed.parallel_state as ps
import vllm.model_executor.parameter as pm
import vllm.distributed.device_communicators.base_device_communicator as bc


class _FakeGroupCoordinator:
    def __init__(self):
        self.rank = 0
        self.local_rank = 0
        self.rank_in_group = 0
        self.world_size = 1
        self.ranks = [0]
        self.cpu_group = self
        self.device_group = self
        self.is_first_rank = True
        self.is_last_rank = True

    def all_reduce(self, *a, **kw):
        return a[0] if a else None

    def all_gather(self, *a, **kw):
        return a[0] if a else None

    def reduce_scatter(self, *a, **kw):
        return a[0] if a else None

    def broadcast(self, *a, **kw):
        return a[0] if a else None

    def gather(self, *a, **kw):
        return [a[0]] if a else None

    def size(self):
        return 1

    def destroy(self):
        pass


for _p in [
    patch.object(ps, "_TP", _FakeGroupCoordinator()),
    patch.object(ps, "_PP", _FakeGroupCoordinator()),
    patch.object(ps, "_DP", _FakeGroupCoordinator()),
    patch.object(ps, "_EP", _FakeGroupCoordinator()),
    patch.object(ps, "_EPLB", _FakeGroupCoordinator()),
    patch.object(ps, "_PCP", _FakeGroupCoordinator()),
    patch.object(pm, "get_tensor_model_parallel_rank", return_value=0),
    patch.object(pm, "get_tensor_model_parallel_world_size", return_value=1),
    patch.object(bc, "DeviceCommunicatorBase"),
]:
    _p.start()

# ═══════════════════════════════════════════════
# Attention 后端选择补丁（无 GPU 环境使用 CPU attention）
# ═══════════════════════════════════════════════

import vllm.platforms as _vllm_platforms
from vllm.platforms.interface import PlatformEnum

_cur = _vllm_platforms.current_platform
if getattr(_cur, "_enum", None) == PlatformEnum.UNSPECIFIED:
    patch.object(
        _cur,
        "get_attn_backend_cls",
        return_value="vllm.v1.attention.backends.cpu_attn.CPUAttentionBackend",
    ).start()
    patch.object(
        _cur,
        "opaque_attention_op",
        return_value=True,
    ).start()
    patch.object(_cur, "device_type", "cpu").start()
    patch.object(_cur, "is_cpu", return_value=True).start()
    patch.object(_cur, "_enum", PlatformEnum.CPU).start()
    # DeepSeekV4 attention checks get_device_capability();
    # return SM100 capability to bypass the CUDA assertion
    from vllm.platforms.interface import DeviceCapability
    patch.object(
        _cur,
        "get_device_capability",
        return_value=DeviceCapability(major=10, minor=0),
    ).start()

# ═══════════════════════════════════════════════
# MLA Attention 补丁（CPU 环境忽略 MLA 参数）
# ═══════════════════════════════════════════════

import vllm.v1.attention.backends.cpu_attn as _cpu_attn

_orig_cpu_attn_impl_init = _cpu_attn.CPUAttentionBackendImpl.__init__


def _patched_cpu_attn_impl_init(self, *args, **kwargs):
    mla_kwargs = {
        "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim",
        "qk_head_dim", "v_head_dim", "kv_b_proj", "indexer",
    }
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in mla_kwargs}
    _orig_cpu_attn_impl_init(self, *args, **filtered_kwargs)


_cpu_attn.CPUAttentionBackendImpl.__init__ = _patched_cpu_attn_impl_init
# Bypass FP8 KV cache check in CPU attention (fake mode, no actual cache)
_cpu_attn.is_quantized_kv_cache = lambda x: False


# ═══════════════════════════════════════════════
# CPU GEMM 补丁（FakeTensor 模式下跳过 cpu_linear 检查）
# ═══════════════════════════════════════════════

import vllm.model_executor.layers.utils as _linear_utils

_orig_cpu_unquantized_gemm = _linear_utils.cpu_unquantized_gemm


def _patched_cpu_unquantized_gemm(layer, x, weight, bias=None):
    if not hasattr(layer, "cpu_linear"):
        result = x.new_empty(x.shape[:-1] + (weight.shape[0],))
        if bias is not None:
            result = result + bias
        return result
    return _orig_cpu_unquantized_gemm(layer, x, weight, bias)


_linear_utils.cpu_unquantized_gemm = _patched_cpu_unquantized_gemm


# ═══════════════════════════════════════════════
# MoE Runner 补丁（FakeTensor 模式下跳过上下文检查）
# ═══════════════════════════════════════════════

import vllm.model_executor.layers.fused_moe.runner.moe_runner as _moe_runner



def _patched_moe_forward(hidden_states, router_logits, shared_experts_input, input_ids, layer_name):
    return torch.empty_like(hidden_states)


def _patched_moe_forward_shared(hidden_states, router_logits, shared_experts_input, input_ids, layer_name):
    fused_out = torch.empty_like(hidden_states)
    if shared_experts_input is not None:
        shared_out = torch.empty_like(shared_experts_input)
    else:
        shared_out = torch.empty_like(hidden_states)
    return shared_out, fused_out


_moe_runner._moe_forward = _patched_moe_forward
_moe_runner._moe_forward_shared = _patched_moe_forward_shared


def patch_deepseek_kernels():
    """Patch DeepSeekV4 CUDA kernel functions for fake mode. Call after
    deepseek_v4_attention module is imported (right before model init)."""
    import vllm.v1.attention.ops.deepseek_v4_ops.fused_inv_rope_fp8_quant as _firfq
    import vllm.v1.attention.ops.deepseek_v4_ops.cache_utils as _cache_utils

    def _patched_fused_inv_rope_fp8_quant(
        o, positions, cos_sin_cache,
        n_groups, heads_per_group,
        nope_dim=448, rope_dim=64,
        quant_group_size=128, tma_aligned_scales=False,
    ):
        from vllm.utils.deep_gemm import get_tma_aligned_size
        num_tokens, num_heads, head_dim = o.shape
        d = heads_per_group * head_dim
        num_scale_blocks = d // quant_group_size
        fp8_dtype = torch.float8_e4m3fn
        fp8_buf = torch.empty((n_groups, num_tokens, d), dtype=fp8_dtype, device=o.device)
        tma_aligned_T = get_tma_aligned_size(num_tokens, 4)
        if tma_aligned_scales:
            packed_sf_k = (num_scale_blocks + 3) // 4
            scale_buf = torch.empty(
                n_groups * packed_sf_k * tma_aligned_T, dtype=torch.int32, device=o.device,
            ).as_strided(
                (n_groups, num_tokens, packed_sf_k),
                (packed_sf_k * tma_aligned_T, 1, tma_aligned_T),
            )
        else:
            scale_buf = torch.empty(
                n_groups * num_scale_blocks * tma_aligned_T, dtype=torch.float32, device=o.device,
            ).as_strided(
                (n_groups, num_tokens, num_scale_blocks),
                (num_scale_blocks * tma_aligned_T, 1, tma_aligned_T),
            )
        return fp8_buf.transpose(0, 1), scale_buf.transpose(0, 1)

    _firfq.fused_inv_rope_fp8_quant = _patched_fused_inv_rope_fp8_quant

    # Also patch the import binding in deepseek_v4_attention if it exists
    try:
        import vllm.model_executor.layers.deepseek_v4_attention as _ds_attn
        _ds_attn.fused_inv_rope_fp8_quant = _patched_fused_inv_rope_fp8_quant
    except ImportError:
        pass

    # cache_utils – quantize_and_insert_k
    def _patched_quantize_and_insert_k(*a, **kw):
        return
    _cache_utils.quantize_and_insert_k_cache = _patched_quantize_and_insert_k


# ═══════════════════════════════════════════════
# 预设所有 torch.empty 为 meta device 避免显式/内存分配。
# ═══════════════════════════════════════════════

_orig_empty = torch.empty
_orig_ones = torch.ones
_orig_zeros = torch.zeros


def _fake_tensor_factory(fn):
    def wrapper(*args, **kwargs):
        kwargs.setdefault("device", "meta")
        return fn(*args, **kwargs)
    return wrapper


torch.empty = _fake_tensor_factory(_orig_empty)
torch.ones = _fake_tensor_factory(_orig_ones)
torch.zeros = _fake_tensor_factory(_orig_zeros)

# ═══════════════════════════════════════════════
# Fake tilelang mock (mhc.py imports it lazily; on CPU it's None
# but the @tilelang.jit decorator still gets evaluated)
# ═══════════════════════════════════════════════

import sys as _sys


class _FakeTilelang:
    class PassConfigKey:
        TL_DISABLE_WARP_SPECIALIZED = None
        TL_DISABLE_TMA_LOWER = None
        TL_PTXAS_REGISTER_USAGE_LEVEL = None

    @staticmethod
    def jit(*args, **kwargs):
        return lambda f: f


_sys.modules.setdefault("tilelang", _FakeTilelang())

# ═══════════════════════════════════════════════
# CUDA Stream Mock (DeepSeekV4 creates CUDA streams in __init__)
# ═══════════════════════════════════════════════

class _FakeCUDASteam:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def wait_stream(self, *a, **kw): pass
    def record_event(self, *a, **kw): return None
    def synchronize(self): pass

class _FakeCUDASteamEvent:
    def __init__(self, *a, **kw): pass
    def record(self, *a, **kw): pass
    def wait(self, *a, **kw): pass
    def synchronize(self): pass
    def query(self): return True

torch.cuda.Stream = _FakeCUDASteam
torch.cuda.Event = _FakeCUDASteamEvent
torch.cuda.is_available = lambda: True

# ═══════════════════════════════════════════════
# VllmConfig 上下文
# ═══════════════════════════════════════════════

from vllm.config.vllm import set_current_vllm_config, VllmConfig
from vllm.config import DeviceConfig, ParallelConfig


@contextmanager
def vllm_config_ctx():
    cfg = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        parallel_config=ParallelConfig(
            pipeline_parallel_size=1, tensor_parallel_size=1
        ),
    )
    with set_current_vllm_config(cfg):
        yield


# ═══════════════════════════════════════════════
# FakeTensor 工具函数
# ═══════════════════════════════════════════════


def to_fake_model(model, fake_mode=None):
    if fake_mode is None:
        fake_mode = FakeTensorMode()
    for m in model.modules():
        for name, p in list(m._parameters.items()):
            if p is not None:
                m._parameters[name] = torch.nn.Parameter(
                    fake_mode.from_tensor(
                        torch.empty(p.shape, dtype=p.dtype, device=p.device)
                    ),
                    requires_grad=p.requires_grad,
                )
        for name, b in list(m._buffers.items()):
            if b is not None and not name.endswith("_numel"):
                m._buffers[name] = fake_mode.from_tensor(
                    torch.empty(b.shape, dtype=b.dtype, device=b.device)
                )


# ═══════════════════════════════════════════════
# vLLM Config 工厂（跳过 HF 下载）
# ═══════════════════════════════════════════════


def make_vllm_config(llama_config: "LlamaConfig", cache_dtype: str = "auto") -> "VllmConfig":
    """Create a VllmConfig with a custom LlamaConfig, bypassing HF download."""
    from types import SimpleNamespace
    import torch
    from vllm.config import CompilationConfig, CacheConfig, ParallelConfig, DeviceConfig
    from vllm.config.vllm import VllmConfig

    model_config = SimpleNamespace(
        hf_config=llama_config,
        dtype=torch.float32,
        is_mm_prefix_lm=False,
        use_mla=False,
    )
    model_config.compute_hash = lambda: "fake_hash"
    cache_config = CacheConfig(block_size=16, cache_dtype=cache_dtype)
    compilation_config = CompilationConfig()
    compilation_config.mode = 0  # disable compilation
    compilation_config.custom_ops = ["none"]
    vllm_config = VllmConfig(
        cache_config=cache_config,
        parallel_config=ParallelConfig(pipeline_parallel_size=1, tensor_parallel_size=1),
        device_config=DeviceConfig(device="cpu"),
        compilation_config=compilation_config,
        quant_config=None,
        load_config={},
    )
    object.__setattr__(vllm_config, "model_config", model_config)
    return vllm_config


__all__ = [
    "FakeTensorMode",
    "vllm_config_ctx",
    "to_fake_model",
    "make_vllm_config",
]
