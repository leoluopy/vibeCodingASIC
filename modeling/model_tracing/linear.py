from .base import BaseModeler, parse_shape, first_shape, get_elem_size, prod, latency_us


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
