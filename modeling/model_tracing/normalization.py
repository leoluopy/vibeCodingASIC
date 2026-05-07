from .base import BaseModeler, parse_shape, first_shape, get_elem_size, prod, latency_us

"""
典型形状示例:
  Llama-7B RMSNorm:  input=(1, 1, 4096), output=(1, 1, 4096) → B=1, H=4096
  Llama-70B RMSNorm: input=(1, 1, 8192), output=(1, 1, 8192) → B=1, H=8192

估计推导:
  RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
  每元素: x^2 (mul), mean (reduce/scatter), rsqrt (SFU), x * rsqrt (mul), x * weight (mul)
  约 4 次 1d 运算 / 元素
  FLOPs = 4 * B * H
  延迟 = max(FLOPs/1d_peak, (in+out)*elem_size/bandwidth)
"""


class NormModeler(BaseModeler):

    def estimate(self, name, args):
        input_shape = first_shape(parse_shape(args['input_shape']))
        output_shape = first_shape(parse_shape(args['output_shape']))
        dtype = args.get('input_dtype', 'torch.float32')

        B = input_shape[0]
        H = input_shape[-1]

        flops = 4 * B * H

        es = get_elem_size(dtype)
        in_bytes = prod(input_shape) * es
        out_bytes = prod(output_shape) * es

        return latency_us(flops, in_bytes + out_bytes, self.chip_specs, '1d_peak_flops')
