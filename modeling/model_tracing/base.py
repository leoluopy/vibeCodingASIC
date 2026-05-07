import re
from .config import chip_specs as default_chip_specs, dtype_size


def parse_shape(shape_str):
    shapes = shape_str.split('|')
    result = []
    for s in shapes:
        s = s.strip()
        nums = re.findall(r'\d+', s)
        result.append(tuple(int(n) for n in nums))
    return result if len(result) > 1 else result[0]


def prod(shape):
    p = 1
    for d in shape:
        p *= d
    return p


def first_shape(shapes):
    if isinstance(shapes, list):
        return shapes[0] if shapes else ()
    return shapes


def get_elem_size(dtype):
    return dtype_size.get(dtype, 4)


def latency_us(compute_flops, mem_bytes, chip_specs=None, peak_key='2d_peak_flops', sfu_flops=0):
    cs = chip_specs or default_chip_specs
    compute_time = compute_flops / cs[peak_key]
    if sfu_flops > 0:
        compute_time += sfu_flops / cs['sfu_peak_flops']
    mem_time = mem_bytes / cs['memory_bandwidth']
    return max(compute_time, mem_time) * 1e6


class BaseModeler:

    def __init__(self, chip_specs=None):
        self.chip_specs = chip_specs or default_chip_specs

    def mem_bytes(self, *shapes, dtype='torch.float32'):
        es = get_elem_size(dtype)
        total = 0
        for sh in shapes:
            total += prod(sh) * es
        return total

    def estimate(self, name, args):
        raise NotImplementedError
