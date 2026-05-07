from .base import BaseModeler, parse_shape, first_shape, get_elem_size, prod

GATED_MLP_EXPAND = 16 / 3


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
