from .base import BaseModeler, parse_shape, get_elem_size, prod, latency_us

HEAD_DIM = 128

"""
典型形状示例 (CoreAttention - 单核 attention 计算):
  prefill:  input=(S, 4096)  → NH=32, S=seq_len, HEAD_DIM=128
  decode:   input=(1, 4096)  → NH=32, S=1, HEAD_DIM=128
  其中 4096 = NH * HEAD_DIM = 32 * 128

计算推导 (per head, per sequence, 不含 batch 维):

  Q, K, V: (S, HEAD_DIM)

  [1] Q @ K^T → attn_logits: (S, HEAD_DIM) @ (HEAD_DIM, S) → (S, S)
      M=S, K=HEAD_DIM, N=S
      flops_per_head = 2 * M * N * K = 2 * S * S * HEAD_DIM

  [2] attn @ V → output: (S, S) @ (S, HEAD_DIM) → (S, HEAD_DIM)
      M=S, K=S, N=HEAD_DIM
      flops_per_head = 2 * M * N * K = 2 * S * HEAD_DIM * S
                      = 2 * S * S * HEAD_DIM (与 [1] 相同)

  单头总 matmul: 4 * S * S * HEAD_DIM
  NH 头总计:    matmul_flops = 4 * NH * S * S * HEAD_DIM

  Softmax输入Prefill和Decode维度分别是 [ NH, S, S]和 [ NH, 1, cached_S]，且都是对最后一维归一化）
  Softmax 分解 (per row of S elements, NH heads):

    exp(x_i):  S 次 exp                            → SFU
    sum(exp):  S 次 add  (reduction)                → 1D
    div:       S 次 div  (x_i / sum)                → 1D

    per_row  = S exp (SFU) + S add (1D) + S div (1D)
             = S (SFU) + 2*S (1D)

    per_head = S * per_row = S*S (SFU) + 2*S*S (1D)
    NH heads:
      softmax_sfu = NH * S * S       (exp)
      softmax_1d  = 2 * NH * S * S   (sum + div)

  score_bytes = NH * S * S * 4  (fp32 中间分数)

  延迟 = max(matmul/2d_peak + softmax_sfu/sfu_peak + softmax_1d/1d_peak, (in+out+score)/bandwidth)
  注: prefill 时 S 为 prompt 长度, decode 时 S=1

--------------------------------------------------------------
MLAAttentionModeler - DeepSeekV2 MLA (Multi-head Latent Attention):
  prefill:  input=(S, 512), output=(S, 4096)  → S=seq_len, latent_dim=512, H_out=4096
  decode:   input=(1, 512),  output=(1, 4096)  → S=1, latent_dim=512, H_out=4096

  MLA 将 Q/KV 压缩到低维 latent space 后做 attention
  本质是一个 matmul: x @ W_out

  x: (S, latent_dim), W_out: (latent_dim, H_out)
  x @ W_out → (S, H_out)

  M=S, K=latent_dim, N=H_out
  flops = 2 * M * N * K = 2 * S * latent_dim * H_out
"""


class CoreAttentionModeler(BaseModeler):

    def estimate(self, name, args):
        input_shape = parse_shape(args['input_shape'])
        output_shape = parse_shape(args['output_shape'])
        dtype = args.get('input_dtype', 'torch.float32')

        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        if isinstance(output_shape, list):
            output_shape = output_shape[0]

        B = input_shape[0]
        H = input_shape[-1]

        S = input_shape[1] if len(input_shape) >= 3 else B
        NH = max(1, H // HEAD_DIM)

        matmul_flops = 4 * NH * S * S * HEAD_DIM
        softmax_sfu = NH * S * S          # exp
        softmax_1d = 2 * NH * S * S       # sum + div

        es = get_elem_size(dtype)
        in_bytes = prod(input_shape) * es
        out_bytes = prod(output_shape) * es
        score_bytes = NH * S * S * 4

        cs = self.chip_specs
        compute_time = (matmul_flops / cs['2d_peak_flops']
                        + softmax_sfu / cs['sfu_peak_flops']
                        + softmax_1d / cs['1d_peak_flops'])
        mem_time = (in_bytes + out_bytes + score_bytes) / cs['memory_bandwidth']
        return max(compute_time, mem_time) * 1e6



class MLAAttentionModeler(BaseModeler):

    def estimate(self, name, args):
        input_shape = parse_shape(args['input_shape'])
        output_shape = parse_shape(args['output_shape'])
        dtype = args.get('input_dtype', 'torch.float32')

        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        if isinstance(output_shape, list):
            output_shape = output_shape[0]

        if len(input_shape) == 3:
            B, S, latent_dim = input_shape
        else:
            B = input_shape[0]
            latent_dim = input_shape[-1]
            S = 1

        H_out = output_shape[-1]

        flops = 2 * B * S * latent_dim * H_out

        es = get_elem_size(dtype)
        in_bytes = prod(input_shape) * es
        out_bytes = prod(output_shape) * es

        return latency_us(flops, in_bytes + out_bytes, self.chip_specs, '2d_peak_flops')
