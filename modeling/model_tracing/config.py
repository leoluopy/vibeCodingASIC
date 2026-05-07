chip_specs = {
    '2d_peak_flops': 1e12,
    '1d_peak_flops': 5e11,
    'sfu_peak_flops': 2e11,
    'memory_bandwidth': 1e11,
}

dtype_size = {
    'torch.float32': 4,
    'torch.bfloat16': 2,
    'torch.float16': 2,
    'torch.int64': 8,
    'torch.int32': 4,
    'torch.int8': 1,
}
