from .base import BaseModeler, parse_shape, first_shape, get_elem_size, prod, latency_us

"""
典型形状示例:
  Llama-7B MLP gate:  input=(1, 1, 11008), output=(1, 1, 5461)
  Llama-7B MLP up:    input=(1, 1, 11008), output=(1, 1, 5461)
  → B=1, H=11008, H//2=5504

估计推导:
  SiLU(x1) * x2,  x1,x2: (B, H//2)
  逐元素:  x2_sig = sigmoid(x2) → 1 SFU
           x1 * x2_sig          → 1 mul (1d)
           total: 1 SFU + 1 mul per elem
  但由于 fused silu_and_mul 也包含 gate 的 reshape, 额外开销:
    flops_1d = 4 * B * (H//2)   (2 read + 2 write per elem)
    flops_sfu = 1 * B * (H//2)  (sigmoid)
  延迟 = max(flops_1d/1d_peak + flops_sfu/sfu_peak, mem/bandwidth)
"""


class SiluAndMulModeler(BaseModeler):

    def estimate(self, name, args):
        input_shape = first_shape(parse_shape(args['input_shape']))
        output_shape = first_shape(parse_shape(args['output_shape']))
        dtype = args.get('input_dtype', 'torch.float32')

        B = input_shape[0]
        H = input_shape[-1]

        flops_1d = 4 * B * (H // 2)
        flops_sfu = 1 * B * (H // 2)

        es = get_elem_size(dtype)
        in_bytes = prod(input_shape) * es
        out_bytes = prod(output_shape) * es

        cs = self.chip_specs
        compute_time = flops_1d / cs['1d_peak_flops'] + flops_sfu / cs['sfu_peak_flops']
        mem_time = (in_bytes + out_bytes) / cs['memory_bandwidth']
        return max(compute_time, mem_time) * 1e6
