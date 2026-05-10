from .base import BaseModeler, parse_shape, first_shape, get_elem_size, prod, latency_us

"""
线性层: y = x @ W^T + bias

GEMM 等价形式:
  A = x 重塑为 (m, k)
  B = W^T, 形状 (k, n)
  C = y 重塑为 (m, n)

  其中:
    m = prod(input_shape[:-1])   = batch * seq_len (token 总数)
    n = output_shape[-1]         = 输出特征数
    k = input_shape[-1]          = 输入特征数 (reduction dim)

  权重形状: W: (n, k), 即 (N_out, M_in)
  权重字节: n * k * elem_size (一次性从 HBM 加载)

Prefill vs Decode (不同 token 数):
  场景        输入形状             m         说明
  prefill    (S, M)            = S       全序列并行, S = seq_len
  prefill    (B, S, M)         = B * S   带 batch 的全序列
  decode     (1, M)            = 1       单 token 自回归
  decode     (B, 1, M)         = B       带 batch 的单 token

  典型示例:
    prefill:  x=(128, 4096),  y=(128, 4096)    → m=128,  n=4096, k=4096
    prefill:  x=(128, 4096),  y=(128, 11008)   → m=128,  n=11008, k=4096
    decode:   x=(1, 4096),    y=(1, 4096)      → m=1,    n=4096,  k=4096

FLOPs = 2 * m * n * k
  (m 次 n×k 矩阵乘法, 每次乘加 2 ops)

Mem = input(m*k) + output(m*n) + weight(n*k)  (单位: bytes)
  权重一次性加载, 对所有 token 复用

延迟 = max(FLOPs / 2d_peak_flops, Mem / bandwidth)
"""


class LinearModeler(BaseModeler):

    def estimate(self, name, args):
        input_shape = first_shape(parse_shape(args['input_shape']))
        output_shape = first_shape(parse_shape(args['output_shape']))
        dtype = args.get('input_dtype', 'torch.float32')

        m = prod(input_shape[:-1])
        n = output_shape[-1]
        k = input_shape[-1]

        flops = 2 * m * n * k

        es = get_elem_size(dtype)
        in_bytes = prod(input_shape) * es
        out_bytes = prod(output_shape) * es
        weight_bytes = n * k * es

        return latency_us(flops, in_bytes + out_bytes + weight_bytes, self.chip_specs, '2d_peak_flops')
