from .base import BaseModeler, parse_shape, first_shape, get_elem_size, prod, latency_us


class FusedMoEModeler(BaseModeler):

    def estimate(self, name, args):
        input_shape = first_shape(parse_shape(args['input_shape']))
        output_shape = first_shape(parse_shape(args['output_shape']))
        dtype = args.get('input_dtype', 'torch.float32')

        B = input_shape[0]
        H_in = input_shape[-1]
        H_out = output_shape[-1]

        flops = 4 * B * H_in * H_out

        es = get_elem_size(dtype)
        in_bytes = prod(input_shape) * es
        out_bytes = prod(output_shape) * es

        return latency_us(flops, in_bytes + out_bytes, self.chip_specs, '2d_peak_flops')


class MoEModeler(BaseModeler):

    def estimate(self, name, args):
        output_shape = first_shape(parse_shape(args['output_shape']))
        dtype = args.get('output_dtype', 'torch.float32')

        B = output_shape[0]
        H = output_shape[-1]

        n_experts = 8
        top_k = 2
        intermediate = int(round(H * 16 / 3 / 2) * 2)

        gate_flops = 2 * B * H * n_experts
        expert_up_flops = n_experts * top_k * (B // n_experts) * H * intermediate * 2
        expert_down_flops = n_experts * top_k * (B // n_experts) * intermediate * H * 2
        total_matmul = gate_flops + expert_up_flops + expert_down_flops

        es = get_elem_size(dtype)
        in_bytes = B * H * es
        out_bytes = B * H * es

        cs = self.chip_specs
        compute_time = total_matmul / cs['2d_peak_flops']
        mem_time = (in_bytes + out_bytes) / cs['memory_bandwidth']
        return max(compute_time, mem_time) * 1e6
