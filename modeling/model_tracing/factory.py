from .linear import LinearModeler
from .normalization import NormModeler
from .activation import SiluAndMulModeler
from .embedding import EmbeddingModeler
from .rotary import RotaryModeler
from .attention import CoreAttentionModeler, CompositeAttentionModeler, MLAAttentionModeler
from .mlp import MLPModeler
from .moe import FusedMoEModeler, MoEModeler


def is_linear(name):
    linear_keywords = [
        'QKVParallelLinear', 'MergedColumnParallelLinear', 'RowParallelLinear',
        'ColumnParallelLinear', 'ParallelLinear', 'FusedQkvAProjLinear',
    ]
    for kw in linear_keywords:
        if kw in name:
            return True
    return name.endswith('Linear') or name.endswith('Projection')


def is_attention_core(name):
    return name == 'Attention'


def is_attention_composite(name):
    attn_names = [
        'LlamaAttention', 'Qwen2Attention', 'Qwen2SdpaAttention',
        'DeepseekV2MLAAttention', 'DeepseekV4Attention',
    ]
    return name in attn_names or 'AttentionWrapper' in name


def is_mla_attention(name):
    return name == 'MLAAttention'


def is_mlp(name):
    mlp_names = ['LlamaMLP', 'Qwen2MLP', 'DeepseekV2MLP', 'DeepseekV4MLP']
    return name in mlp_names or (name.endswith('MLP') and 'MoE' not in name)


def is_moe(name):
    return 'MoE' in name and 'MoE' != name


def is_fused_moe(name):
    return 'FusedMoE' in name or name == 'FusedMoE'


def is_decoder_layer(name):
    return 'DecoderLayer' in name


def get_modeler(name, chip_specs=None):
    if name == 'VocabParallelEmbedding':
        return EmbeddingModeler(chip_specs)
    if name == 'RMSNorm':
        return NormModeler(chip_specs)
    if name == 'SiluAndMul':
        return SiluAndMulModeler(chip_specs)
    if 'RotaryEmbedding' in name:
        return RotaryModeler(chip_specs)
    if is_fused_moe(name):
        return FusedMoEModeler(chip_specs)
    if is_moe(name):
        return MoEModeler(chip_specs)
    if is_mla_attention(name):
        return MLAAttentionModeler(chip_specs)
    if is_attention_composite(name):
        return CompositeAttentionModeler(chip_specs)
    if is_attention_core(name):
        return CoreAttentionModeler(chip_specs)
    if is_mlp(name):
        return MLPModeler(chip_specs)
    if is_decoder_layer(name):
        return DecoderLayerModeler(chip_specs)
    if is_linear(name):
        return LinearModeler(chip_specs)
    return None
