from .base import BaseModeler, parse_shape, get_elem_size, prod, latency_us


class RotaryModeler(BaseModeler):

    def estimate(self, name, args):
        raw = parse_shape(args['output_shape'])
        dtype = args.get('output_dtype', 'torch.float32')

        if isinstance(raw, list):
            all_shapes = raw
        else:
            all_shapes = [raw]

        total_sfu = 0
        total_bytes = 0
        es = get_elem_size(dtype)

        for sh in all_shapes:
            elems = prod(sh)
            total_sfu += elems * 2
            total_bytes += elems * es

        cs = self.chip_specs
        compute_time = total_sfu / cs['sfu_peak_flops']
        mem_time = total_bytes / cs['memory_bandwidth']
        return max(compute_time, mem_time) * 1e6
