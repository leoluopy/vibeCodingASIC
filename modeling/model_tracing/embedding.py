from .base import BaseModeler, parse_shape, first_shape, get_elem_size, prod

"""
典型形状示例:
  VocabParallelEmbedding: input=(1, 1), output=(1, 1, 4096) → B=1, H=4096
  即: token_id → hidden_state lookup

估计推导:
  Embedding 本质是查表 (gather), 几乎无计算量
  延迟仅受内存带宽限制:
  Mem = output(B, H) * elem_size
  延迟 = Mem / memory_bandwidth
"""


class EmbeddingModeler(BaseModeler):

    def estimate(self, name, args):
        output_shape = first_shape(parse_shape(args['output_shape']))
        dtype = args.get('output_dtype', 'torch.float32')

        es = get_elem_size(dtype)
        out_bytes = prod(output_shape) * es

        cs = self.chip_specs
        mem_time = out_bytes / cs['memory_bandwidth']
        return mem_time * 1e6
