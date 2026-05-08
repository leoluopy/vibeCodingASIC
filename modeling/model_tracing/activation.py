from .base import BaseModeler, parse_shape, first_shape, get_elem_size, prod, latency_us

"""
典型形状示例:
  Llama-7B MLP SiluAndMul:  input=(1, 1, 11008), output=(1, 1, 5461)
  → B=1, H=11008, H//2=5504, intermediate=H//2

估计推导:
  SiluAndMul 接收合并的 merged 张量 (B, H), H=2*intermediate
  内部拆分: gate=x[:d], up=x[d:], d=H//2
  计算: out = silu(gate) * up = gate * sigmoid(gate) * up
  逐元素:
    sigmoid(gate) → 1 SFU
    gate * sigmoid(gate) → 1 mul (1d)
    result * up → 1 mul (1d)
  flops_1d = 2 * B * (H//2)   (2 element-wise mul per elem)
  flops_sfu = 1 * B * (H//2)  (sigmoid)
  读: merged 张量 (B, H) = 2 * B * (H//2) elements
  写: output 张量 (B, H//2) = B * (H//2) elements
  延迟 = max(flops_1d/1d_peak + flops_sfu/sfu_peak, mem/bandwidth)
"""


class SiluAndMulModeler(BaseModeler):

    def estimate(self, name, args):
        input_shape = first_shape(parse_shape(args['input_shape']))
        output_shape = first_shape(parse_shape(args['output_shape']))
        dtype = args.get('input_dtype', 'torch.float32')

        B = input_shape[0]
        H = input_shape[-1]
        N = B * (H // 2)

        flops_1d = 2 * N
        flops_sfu = 1 * N

        es = get_elem_size(dtype)
        in_bytes = prod(input_shape) * es
        out_bytes = prod(output_shape) * es

        cs = self.chip_specs
        compute_time = flops_1d / cs['1d_peak_flops'] + flops_sfu / cs['sfu_peak_flops']
        mem_time = (in_bytes + out_bytes) / cs['memory_bandwidth']
        return max(compute_time, mem_time) * 1e6
