from .base import BaseModeler, parse_shape, first_shape, get_elem_size, prod, latency_us

HEAD_DIM = 128
GATED_MLP_EXPAND = 16 / 3


class DecoderLayerModeler(BaseModeler):

    def estimate(self, name, args):
        output_shape = first_shape(parse_shape(args['output_shape']))
        dtype = args.get('output_dtype', 'torch.float32')

        B = output_shape[0]
        H = output_shape[-1]
        S = B
        NH = H // HEAD_DIM
        merged_dim = int(round(H * GATED_MLP_EXPAND / 2) * 2)
        intermediate = merged_dim // 2

        norm_flops = 2 * 4 * B * H

        qkv_flops = 6 * B * H * H
        rope_sfu = B * H
        attn_matmul = 4 * NH * S * S * HEAD_DIM
        attn_softmax = 3 * NH * S * S
        out_proj_flops = 2 * B * H * H
        attn_total = qkv_flops + attn_matmul + out_proj_flops

        merged_flops = 2 * B * H * merged_dim
        act_flops_1d = 4 * B * intermediate
        act_flops_sfu = 1 * B * intermediate
        down_flops = 2 * B * intermediate * H
        mlp_total = merged_flops + down_flops

        total_matmul = norm_flops + attn_total + mlp_total
        total_sfu = rope_sfu + attn_softmax + act_flops_sfu

        es = get_elem_size(dtype)
        in_bytes = B * H * es
        out_bytes = B * H * es
        qkv_bytes = B * 3 * H * es
        score_bytes = NH * S * S * 4
        merged_bytes = B * merged_dim * es
        total_bytes = in_bytes + out_bytes + qkv_bytes + score_bytes + merged_bytes

        cs = self.chip_specs
        compute_time = total_matmul / cs['2d_peak_flops'] + total_sfu / cs['sfu_peak_flops'] + (4 * B * intermediate) / cs['1d_peak_flops']
        mem_time = total_bytes / cs['memory_bandwidth']
        return max(compute_time, mem_time) * 1e6
