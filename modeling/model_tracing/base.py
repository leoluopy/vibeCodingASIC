import re
from .config import chip_specs as default_chip_specs, dtype_size

"""
=============================================================================
典型形状示例 (Typical Shape Examples)
--------------------------------------
以下为常见 LLM 配置下的张量形状参考（batch_size=1, seq_len=1, 即 prefill 场景）:

  Llama-7B:    H=4096, NH=32,   HEAD_DIM=128, intermediate=11008
  Llama-13B:   H=5120, NH=40,   HEAD_DIM=128, intermediate=13824
  Llama-70B:   H=8192, NH=64,   HEAD_DIM=128, intermediate=28672
  Qwen2-7B:    H=3584, NH=28,   HEAD_DIM=128, intermediate=18944
  DeepSeekV2:  H=5120, NH=64,   HEAD_DIM=128, kv_latent_dim=512, MoE

=============================================================================
估计过程推导 (Estimate Process Derivation)
------------------------------------------
延迟估计采用 Roofline 模型:  latency = max(compute_time, mem_time)

  1) compute_time = total_flops / peak_flops
     - 矩阵乘法 (matmul) → 2d_peak_flops
     - 逐元素运算 (element-wise: add/mul) → 1d_peak_flops
     - 特殊函数 (softmax/silu/rope) → sfu_peak_flops

  2) mem_time = total_bytes / memory_bandwidth
     - 考虑所有中间张量的读写字节数

  3) 最终延迟 = max(compute, mem) * 1e6  (单位: μs)
=============================================================================
"""


def parse_shape(shape_str):
    shapes = shape_str.split('|')
    result = []
    for s in shapes:
        s = s.strip()
        nums = re.findall(r'\d+', s)
        result.append(tuple(int(n) for n in nums))
    return result if len(result) > 1 else result[0]


def prod(shape):
    p = 1
    for d in shape:
        p *= d
    return p


def first_shape(shapes):
    if isinstance(shapes, list):
        return shapes[0] if shapes else ()
    return shapes


def get_elem_size(dtype):
    return dtype_size.get(dtype, 4)


def latency_us(compute_flops, mem_bytes, chip_specs=None, peak_key='2d_peak_flops', sfu_flops=0):
    cs = chip_specs or default_chip_specs
    compute_time = compute_flops / cs[peak_key]
    if sfu_flops > 0:
        compute_time += sfu_flops / cs['sfu_peak_flops']
    mem_time = mem_bytes / cs['memory_bandwidth']
    return max(compute_time, mem_time) * 1e6


class BaseModeler:

    def __init__(self, chip_specs=None):
        self.chip_specs = chip_specs or default_chip_specs

    def mem_bytes(self, *shapes, dtype='torch.float32'):
        es = get_elem_size(dtype)
        total = 0
        for sh in shapes:
            total += prod(sh) * es
        return total

    def estimate(self, name, args):
        raise NotImplementedError
