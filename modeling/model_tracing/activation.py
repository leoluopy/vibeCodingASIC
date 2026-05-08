from .base import BaseModeler, parse_shape, first_shape, get_elem_size, prod, latency_us

"""
典型形状示例:
  Llama-7B MLP SiluAndMul:  input=(1, 1, 11008), output=(1, 1, 5461)
  → B=1, H=11008, H//2=5504, intermediate=H//2

估计推导:
  SiluAndMul 接收合并的 merged 张量 (B, H), H=2*intermediate
  内部拆分: gate=x[:d], up=x[d:], d=H//2
  计算: out = silu(gate) * up =  sigmoid(gate) * up

"""


class SiluAndMulModeler(BaseModeler):

    def estimate(self, name, args):
        input_shape = first_shape(parse_shape(args['input_shape']))
        output_shape = first_shape(parse_shape(args['output_shape']))
        dtype = args.get('input_dtype', 'torch.float32')

        B = input_shape[0]
        H = input_shape[-1]
        N = B * (H // 2)

        flops_1d = 1 * N
        flops_sfu = 1 * N

        es = get_elem_size(dtype)
        in_bytes = prod(input_shape) * es
        out_bytes = prod(output_shape) * es

        cs = self.chip_specs
        compute_time = flops_1d / cs['1d_peak_flops'] + flops_sfu / cs['sfu_peak_flops']
        mem_time = (in_bytes + out_bytes) / cs['memory_bandwidth']
        return max(compute_time, mem_time) * 1e6
