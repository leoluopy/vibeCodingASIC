from .base import BaseModeler, parse_shape, get_elem_size, prod, latency_us

"""
典型形状示例:
  Llama-7B RotaryEmbedding: output=(1, 32, 1, 128) → NH=32, S=1, HEAD_DIM=128
  或多个输出: [(1,32,1,64), (1,32,1,64)] → Q和K各一半

估计推导:
  RoPE 对每个元素施加旋转:
    x_out = x * cos(theta) + rotate_half(x) * sin(theta)  (复数乘法)
  每元素: 1 cos, 1 sin (SFU), 2 mul, 1 add (1d)
  简化: 2 SFU ops / element (cos+sin 占主导)
  FLOPs_sfu = 2 * total_elems
  延迟 = max(FLOPs_sfu/sfu_peak, total_bytes/bandwidth)
"""


class RotaryModeler(BaseModeler):

    def estimate(self, name, args):
        raw = parse_shape(args['output_shape'])
        dtype = args.get('output_dtype', 'torch.float32')

        if isinstance(raw, list):
            all_shapes = raw
        else:
            all_shapes = [raw]

        total_sfu = 0
        total_bytes = 0
        es = get_elem_size(dtype)

        for sh in all_shapes:
            elems = prod(sh)
            total_sfu += elems * 2
            total_bytes += elems * es

        cs = self.chip_specs
        compute_time = total_sfu / cs['sfu_peak_flops']
        mem_time = total_bytes / cs['memory_bandwidth']
        return max(compute_time, mem_time) * 1e6
