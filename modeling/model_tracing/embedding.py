from .base import BaseModeler, parse_shape, first_shape, get_elem_size, prod


class EmbeddingModeler(BaseModeler):

    def estimate(self, name, args):
        output_shape = first_shape(parse_shape(args['output_shape']))
        dtype = args.get('output_dtype', 'torch.float32')

        es = get_elem_size(dtype)
        out_bytes = prod(output_shape) * es

        cs = self.chip_specs
        mem_time = out_bytes / cs['memory_bandwidth']
        return mem_time * 1e6
