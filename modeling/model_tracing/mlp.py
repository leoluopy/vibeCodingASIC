from .base import BaseModeler, parse_shape, first_shape, get_elem_size, prod

GATED_MLP_EXPAND = 16 / 3

"""
典型形状示例:
  Llama-7B:  H=4096, merged_dim=10922, intermediate=5461
  Llama-13B: H=5120, merged_dim=13654, intermediate=6827
  Llama-70B: H=8192, merged_dim=21846, intermediate=10923

  prefill: output=(1, 1, 4096) → B=1
  decode:  output=(1, 1, 4096) → B=1

估计推导:
  SwiGLU MLP:  output = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down

  权重形状:
    W_gate, W_up:   (H, intermediate)  → 但 fused matmul 使用 merged_dim=2*intermediate
    W_down:          (intermediate, H)

  step 1: merged = x @ [W_gate, W_up]   → 2*B*H*merged_dim flops
    merged 形状: (B, merged_dim), 分为 gate=(B, intermediate), up=(B, intermediate)
  step 2: act = SiLU(gate) * up
    flops_1d = 4 * B * intermediate  (元素级 mul/add)
    flops_sfu = 1 * B * intermediate (sigmoid)
  step 3: out = act @ W_down  → 2 * B * intermediate * H flops

  total_matmul = merged_flops + down_flops
  延迟 = max(total_matmul/2d_peak + act_1d/1d_peak + act_sfu/sfu_peak, mem/bandwidth)
  mem = in(B*H) + merged(B*merged_dim) + out(B*H)
"""


class MLPModeler(BaseModeler):

    def estimate(self, name, args):
        output_shape = first_shape(parse_shape(args['output_shape']))
        dtype = args.get('output_dtype', 'torch.float32')

        B = output_shape[0]
        H = output_shape[-1]

        merged_dim = int(round(H * GATED_MLP_EXPAND / 2) * 2)
        intermediate = merged_dim // 2

        merged_flops = 2 * B * H * merged_dim
        act_flops_1d = 4 * B * intermediate
        act_flops_sfu = 1 * B * intermediate
        down_flops = 2 * B * intermediate * H

        total_matmul = merged_flops + down_flops

        es = get_elem_size(dtype)
        in_bytes = B * H * es
        merged_bytes = B * merged_dim * es
        out_bytes = prod(output_shape) * es

        total_bytes = in_bytes + merged_bytes + out_bytes

        cs = self.chip_specs
        compute_time = total_matmul / cs['2d_peak_flops'] + act_flops_sfu / cs['sfu_peak_flops'] + act_flops_1d / cs['1d_peak_flops']
        mem_time = total_bytes / cs['memory_bandwidth']
        return max(compute_time, mem_time) * 1e6
