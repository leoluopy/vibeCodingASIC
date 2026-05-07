from .base import BaseModeler, parse_shape, first_shape, get_elem_size, prod, latency_us


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
