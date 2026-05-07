from .base import BaseModeler, parse_shape, first_shape, get_elem_size, prod, latency_us

"""
典型形状示例:
  prefill:  input=(1, 4096), output=(1, 4096)    → B=1,  M=4096, N=4096
  prefill:  input=(1, 4096), output=(1, 11008)   → B=1,  M=4096, N=11008  (gate/up proj)
  decode:   input=(1, 1, 4096), output=(1, 1, 4096) → B=1, M=4096, N=4096

估计推导:
  y = x @ W^T,  x: (B, M), W: (N, M)
  FLOPs = 2 * B * M * N   (M 次乘加, 每次 2 ops)
  Mem   = input(B*M) + output(B*N)  (单位: bytes, 权重带宽通常不计入)
  延迟 = max(FLOPs/2d_peak, Mem/bandwidth)
"""


class LinearModeler(BaseModeler):

    def estimate(self, name, args):
        input_shape = first_shape(parse_shape(args['input_shape']))
        output_shape = first_shape(parse_shape(args['output_shape']))
        dtype = args.get('input_dtype', 'torch.float32')

        B = input_shape[0]
        M = input_shape[-1]
        N = output_shape[-1]

        flops = 2 * B * M * N

        es = get_elem_size(dtype)
        in_bytes = prod(input_shape) * es
        out_bytes = prod(output_shape) * es

        return latency_us(flops, in_bytes + out_bytes, self.chip_specs, '2d_peak_flops')
