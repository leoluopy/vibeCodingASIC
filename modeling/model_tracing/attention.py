from .base import BaseModeler, parse_shape, get_elem_size, prod, latency_us

HEAD_DIM = 128


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

        S = B
        NH = max(1, H // HEAD_DIM)

        matmul_flops = 4 * NH * S * S * HEAD_DIM
        softmax_flops = 3 * NH * S * S

        es = get_elem_size(dtype)
        in_bytes = prod(input_shape) * es
        out_bytes = prod(output_shape) * es
        score_bytes = NH * S * S * 4

        cs = self.chip_specs
        compute_time = matmul_flops / cs['2d_peak_flops'] + softmax_flops / cs['sfu_peak_flops']
        mem_time = (in_bytes + out_bytes + score_bytes) / cs['memory_bandwidth']
        return max(compute_time, mem_time) * 1e6


class CompositeAttentionModeler(BaseModeler):

    def estimate(self, name, args):
        raw = parse_shape(args['output_shape'])
        if isinstance(raw, list):
            output_shape = raw[0]
        else:
            output_shape = raw
        dtype = args.get('output_dtype', 'torch.float32')

        B = output_shape[0]
        H = output_shape[-1]
        S = B
        NH = H // HEAD_DIM

        qkv_flops = 6 * B * H * H
        rope_sfu = B * H * 2
        attn_matmul = 4 * NH * S * S * HEAD_DIM
        attn_softmax = 3 * NH * S * S
        out_proj_flops = 2 * B * H * H

        total_matmul = qkv_flops + attn_matmul + out_proj_flops
        total_sfu = rope_sfu + attn_softmax

        es = get_elem_size(dtype)
        in_bytes = B * H * es
        out_bytes = prod(output_shape) * es
        qkv_bytes = B * 3 * H * es
        score_bytes = NH * S * S * 4

        total_bytes = in_bytes + out_bytes + qkv_bytes + score_bytes

        cs = self.chip_specs
        compute_time = total_matmul / cs['2d_peak_flops'] + total_sfu / cs['sfu_peak_flops']
        mem_time = total_bytes / cs['memory_bandwidth']
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
