from .base import BaseModeler, parse_shape, first_shape, get_elem_size, prod, latency_us

"""
================================================================================
FusedMoE — DeepSeek 共享专家层 (Shared Expert)
================================================================================

DeepSeek V2/V3/V4 的每层 MoE 包含两部分:


  ┌─────────────────────────────────────────────────────────────────┐
  │ 共享专家 (FusedMoE in trace):                                   │
  │   一个稠密 SwiGLU MLP, 所有 token 必经                          │
  │   等价于标准 LLM 的 MLP 模块                                    │
  ├─────────────────────────────────────────────────────────────────┤
  │ 路由专家 (DeepseekV2MoE / DeepseekV4MoE in trace):              │
  │   稀疏 MoE, 每个 token 通过 Gate 选出 top_k 个专家处理           │
  └─────────────────────────────────────────────────────────────────┘

  MoE 层最终输出 = shared_out + routed_out  (逐元素相加)

共享专家网络结构:

      x (B, H)          B = batch_size, H = hidden_size
          │
          ▼
   ┌───────────────┐     gate_up_proj:  W_gu: (2*I, H)  (I = moe_intermediate_size)
   │ gate_up_proj  │     y = x @ W_gu^T
   └───────┬───────┘     → (B, 2*I)
           │
           ▼ split
     ┌─────┴─────┐
     │           │
  gate (B,I)   up (B,I)
     │           │
  silu( )        │        act = silu(gate) * up
     │           │        → (B, I)
     └────×──────┘
           │
           ▼
   ┌───────────────┐     down_proj:  W_down: (H, I)
   │  down_proj    │     z = act @ W_down^T
   └───────┬───────┘     → (B, H)
           │
           ▼
      y (B, H)


FusedMoE FLOPs 推导 (一次前向):

  Step 1: gate_up_proj
    形状: (B, H) × (H, 2*I) → (B, 2*I)
    FLOPs = B * H * (2*I) * 2 = 4 * B * H * I
    解释: B 个 token, 每个做一次维度 H→2*I 的矩阵乘法,
         每次乘加运算计 2 FLOPs (乘+加)

  Step 2: silu_and_mul
    形状: gate=(B,I), up=(B,I), act=(B,I)
    FLOPs = B*I (sig/silu, SFU) + B*I (mul, 1D)
    相对 matmul 可忽略, 不在主公式中体现

  Step 3: down_proj
    形状: (B, I) × (I, H) → (B, H)
    FLOPs = B * I * H * 2 = 2 * B * H * I

  Total FLOPs = 4*B*H*I + 2*B*H*I = 6*B*H*I


FusedMoE 记忆体访问 (一次前向):

  输入:  x → B*H*es 字节 (读)
  权重:  W_gu → 2*I*H*es 字节 (加载)
         W_down → H*I*es 字节 (加载)
  输出:  y → B*H*es 字节 (写)

  es = element_size (fp32=4B, bf16=2B, fp8=1B)

  注: 中间激活 gate+up+act 在现代融合 kernel 中保留在 SRAM,
      不产生 HBM 访问, 不计入记忆体瓶颈.

================================================================================
MoE — DeepSeek 路由专家层 (Routed Experts)
================================================================================

DeepSeek V3 路由专家完整计算流程:

  ┌──────────────────────────────────────────────────────────────────┐
  │ Step 1: Gate (Router, 打分)                                     │
  │  scores = sigmoid(x @ W_gate^T)                                 │
  │  形状: (B, H) × (H, E) → (B, E)     E = n_routed_experts        │
  │  FLOPs_gate = 2 * B * H * E                                     │
  │  说明: 每个 token 对 E 个专家分别打分, 得到 logit 向量           │
  └──────────────────────────┬───────────────────────────────────────┘
                             ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ Step 2: Grouped Top-K 选择                                       │
  │                                                                  │
  │  将 E 个专家均匀分为 G 组 (V3.2: E=256, G=8, 每组 32 个专家)     │
  │                                                                  │
  │  2a) 组分数计算:                                                 │
  │      对组 g 内的 E/G 个专家分数从高到低排序,                     │
  │      取前 m 个 (通常 m=E/G, 即全取) 求和 → group_score_g         │
  │                                                                  │
  │  2b) 组选择:                                                     │
  │      对所有 G 个 group_score 排序, 选前 topk_group 个组           │
  │      → 候选专家池 = topk_group * (E/G) 个专家                    │
  │        (V3.2: 选 topk_group=4 组 → 4*32=128 候选)               │
  │                                                                  │
  │  2c) 最终选择:                                                   │
  │      在候选池中取分数最高的 top_k 个专家                          │
  │        (V3.2: top_k=8, 从 128 候选中选 8)                        │
  │                                                                  │
  │  注: 此过程为 routing 决策, 仅有排序/求和操作,                   │
  │      无矩阵乘法, FLOPs 相对 matmul 可忽略.                       │
  │      G=0 或 topk_group=0 时跳过分组, 直接全局 top-k.             │
  └──────────────────────────┬───────────────────────────────────────┘
                             ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ Step 3: Expert Computation (每个选中的专家内执行 SwiGLU)         │
  │                                                                  │
  │  对于每个 token t, 在其选中的 top_k 个专家 e ∈ top_k(t) 上:     │
  │                                                                  │
  │    a) gate_up: 同共享专家                                        │
  │       隐藏: x[t] @ W_gu_e^T  →  4*H*I FLOPs/e                   │
  │       实际: vLLM 融合所有选中 token-expert 对为一个 batched GEMM │
  │       总 token-专家对 = B * top_k                                │
  │       gate_up FLOPs = B * top_k * (4*H*I) = 4*B*top_k*H*I       │
  │                                                                  │
  │    b) silu_and_mul: 2*B*top_k*I FLOPs (可忽略)                  │
  │                                                                  │
  │    c) down:                                                      │
  │       down FLOPs = B * top_k * (2*H*I) = 2*B*top_k*H*I          │
  │                                                                  │
  │  关键洞察: 总专家计算 FLOPs 仅取决于 B*top_k 对 token-专家,      │
  │            与总专家数 E 无关.                                    │
  └──────────────────────────┬───────────────────────────────────────┘
                             ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ Step 4: Weighted Sum (加权求和, 合并各专家输出)                  │
  │                                                                  │
  │  output[t] = Σ  routing_prob[t][e] * expert_out[t][e]           │
  │              e∈top_k(t)                                          │
  │                                                                  │
  │  形状: (B, top_k, H) × 权重 → (B, H)                            │
  │  FLOPs_sum = B * top_k * H * 2  (mul + add)                     │
  └──────────────────────────────────────────────────────────────────┘
                             ▼
                     y_routed (B, H)

                   + y_shared (B, H)  ← 来自 FusedMoE
                   = final output (B, H)

路由专家总 FLOPs:

  Total = 2*B*H*E                (gate)
        + 6*B*top_k*H*I           (expert compute)
        + 2*B*top_k*H             (weighted sum, 通常忽略)

  = 2*B*H*(E + 3*top_k*I + top_k)

路由专家记忆体 (权重, 所有 token 共享, 一次加载):

  W_gate:       E * H * es
  W_gate_up_e:  E * (2*I) * H * es   (每个专家的 gate_up 权重)
  W_down_e:     E * I * H * es        (每个专家的 down 权重)
  总计: E*H*es + 3*E*H*I*es = E*H*es*(1 + 3*I)

  以 DeepSeek-V3.2 (fp8) 为例:
    E=256, H=7168, I=2048, es=1 (fp8)
    权重记忆体 = 256*7168*(1 + 3*2048) = 11.28 GB ← 单层!
    这是典型 memory-bound 场景的关键瓶颈.

路由专家记忆体 (激活, 每次前向读写):

  输入 x:           B*H*es
  gate scores:      B*E*es
  中间:             B*top_k*(2*I + I + H)*es
  输出 y:           B*H*es

  注: 在 fused kernel 中, 各 expert 的中间激活可保留在 SRAM,
      不影响 HBM 带宽瓶颈. 但对 memory-bound 场景, HBM 带宽
      主要被权重加载 + 输入输出读写占据.

================================================================================
数值对比 (DeepSeek-V3.2, prefill B=128, H=7168, I=2048, E=256, top_k=8):

  ┌────────────┬─────────────────────────────┬──────────────────────────┐
  │ 模块       │ FLOPs                       │ 权重记忆体 (fp8)         │
  ├────────────┼─────────────────────────────┼──────────────────────────┤
  │ FusedMoE   │ 6*128*7168*2048 = 11.3 G   │ 3*7168*2048 = 43.0 MB    │
  │ (共享专家) │                             │                          │
  ├────────────┼─────────────────────────────┼──────────────────────────┤
  │ DeepseekV2 │ 2*128*7168*256             │ 256*7168 = 1.8 MB        │
  │ MoE        │ + 6*128*8*7168*2048         │ + 3*256*7168*2048        │
  │ (路由专家) │ = 469.8 G + 90.2 T         │ = 11.28 GB               │
  │            │ = 90.7 T (gate 占比 ~0.5%) │ ← 绝对瓶颈               │
  └────────────┴─────────────────────────────┴──────────────────────────┘

  路由专家 FLOPs = 90.7 TFLOPs @ 1 PFLOPS → 90.7 ms compute
  路由专家权重 = 11.28 GB @ 100 GB/s → 112.8 ms memory
  → Roofline 瓶颈在 memory (weight loading dominate)

  FusedMoE FLOPs = 11.3 GFLOPs (计算极轻)
  → 延迟由记忆体 (43 MB权重 + 激活) 主导
================================================================================
"""


class FusedMoEModeler(BaseModeler):
    """
    共享专家模型 (DeepSeek FusedMoE)

    结构: SwiGLU MLP, 等价于 LlamaMLP:
      gate_up_proj → silu_and_mul → down_proj

    输入/输出形状: (B, H) → (B, H)
      I = moe_intermediate_size (共享专家中间维度)

    FLOPs = 6 * B * H * I     (gate_up + down, 忽略 silu_and_mul)

    权重记忆体 = 3 * H * I * es
    激活记忆体 = 2 * B * H * es (输入+输出)

    DeepSeek 配置参考:
      V3.2: H=7168, I=2048  (权重 ~43MB @ fp8)
      V4:   H=7168, I=3072  (权重 ~64MB @ fp8)
    """

    # 基于 hidden_size 查表获取共享专家 intermediate_size
    # Key = hidden_size, Value = moe_intermediate_size
    SHARED_INTERMEDIATE_MAP = {
        7168: 2048,    # DeepSeek-V3.2 / V2
    }
    DEFAULT_INTERMEDIATE = 2048

    def _get_intermediate(self, H, args):
        """获取共享专家中间维度 I

        优先级:
          1. args['intermediate_size']   — trace 显式传入
          2. SHARED_INTERMEDIATE_MAP 查表 — 基于 H 匹配已知模型
          3. DEFAULT_INTERMEDIATE — 兜底默认值
        """
        if 'intermediate_size' in args:
            return int(args['intermediate_size'])
        return self.SHARED_INTERMEDIATE_MAP.get(H, self.DEFAULT_INTERMEDIATE)

    def estimate(self, name, args):
        input_shape = first_shape(parse_shape(args['input_shape']))
        output_shape = first_shape(parse_shape(args['output_shape']))
        dtype = args.get('input_dtype', 'torch.float32')

        B = input_shape[0]
        H = input_shape[-1]
        I = self._get_intermediate(H, args)

        # ── FLOPs ────────────────────────────────────────────────────
        # gate_up_proj: 2 * B * H * (2*I) = 4*B*H*I
        # down_proj:    2 * B * I * H    = 2*B*H*I
        # 合计: 6*B*H*I
        flops = 6 * B * H * I

        # ── 记忆体 ──────────────────────────────────────────────────
        es = get_elem_size(dtype)
        in_bytes = prod(input_shape) * es
        out_bytes = prod(output_shape) * es
        weight_bytes = (2 * I * H + H * I) * es    # W_gu + W_down

        return latency_us(flops, in_bytes + out_bytes + weight_bytes,
                          self.chip_specs, '2d_peak_flops')


class MoEModeler(BaseModeler):
    """
    路由专家模型 (DeepseekV2MoE / DeepseekV4MoE)

    流程:
      Gate[matmul] → Grouped Top-K[routing] → Expert Compute[SwiGLU × top_k per token] → Weighted Sum

    FLOPs:
      gate:          2*B*H*E
      expert_gate_up: 4*B*top_k*H*I
      expert_down:    2*B*top_k*H*I
      weighted_sum:   2*B*top_k*H  (可忽略)

      Total ≈ 2*B*H*E + 6*B*top_k*H*I

    关键参数 (优先级: trace args > hidden_size 查表 > 默认值):
      n_routed_experts (E):    总专家数
      num_experts_per_tok (K): 每 token 激活数
      moe_intermediate_size (I): 专家中间维度
      n_group (G):             分组数 (DeepSeek V3 独有)
      topk_group:              选中的组数

    注意:
      当前 trace 不包含上述参数. 使用 H 查表匹配已知模型配置.
      如需支持新模型, 在 ROUTED_CONFIG_MAP 中添加条目.
    """

    # (hidden_size,) → (n_experts, top_k, intermediate, n_group, topk_group)
    ROUTED_CONFIG_MAP = {
        7168: (256, 8, 2048, 8, 4),    # DeepSeek-V3.2
    }

    def _get_config(self, H, args):
        """获取路由专家配置 (E, K, I, G, topk_G)"""
        E = int(args.get('n_routed_experts', 0))
        K = int(args.get('num_experts_per_tok', 0))
        I = int(args.get('moe_intermediate_size', 0))

        if E and K and I:
            G = int(args.get('n_group', 0))
            topk_G = int(args.get('topk_group', 0))
            return E, K, I, G, topk_G

        if H in self.ROUTED_CONFIG_MAP:
            return self.ROUTED_CONFIG_MAP[H]

        # 未知模型, 使用 Llama-like MoE 默认值
        E = 8
        K = 2
        I = int(round(H * 16 / 3 / 2) * 2)
        return E, K, I, 0, 0

    def _gate_flops(self, B, H, E):
        """Gate (Router) FLOPs

        公式推导:
          scores = x @ W_gate^T
          形状轨迹:  x: (B, H), W_gate: (E, H), W_gate^T: (H, E), scores: (B, E)
          matmul: (B, H) @ (H, E) → 求和维度 H
          FLOPs = B * E * H * 2 = 2*B*H*E

          注: 此处的 @ W^T 符合 LinearModeler 的约定 W: (out_features, in_features)
        """
        return 2 * B * H * E

    def _expert_flops(self, B, K, H, I):
        """专家内部 SwiGLU 计算 FLOPs

        公式推导:
          每个 token-专家对 (总共 B*K 对):
            gate_up: 2 * H * (2*I) = 4*H*I
            down:    2 * I * H = 2*H*I
            小计: 6*H*I

          总计: B*K * 6*H*I = 6*B*K*H*I

        解释:
          当 K << E (如 V3.2: K=8, E=256), 专家计算的总 FLOPs
          (6*B*K*H*I) 可能远小于 gate 的 2*B*H*E.
          以 V3.2 为例:
            gate: 2*128*7168*256 = 469.8 MFLOPs
            expert: 6*128*8*7168*2048 = 90.2 TFLOPs
          → 专家计算占绝对主导 (192x gate)
        """
        return 6 * B * K * H * I

    def _weighted_sum_flops(self, B, K, H):
        """加权求和 FLOPs

        公式推导:
          output[t][h] = Σ_{e∈top_k(t)} prob[t][e] * expert_out[t][e][h]
          对每个 token t, 每个维度 h:
            K 次 mul + K-1 次 add ≈ 2*K FLOPs
          总计: B * H * 2*K = 2*B*K*H

          相比 matmul, 此部分占比极小 (V3.2: 2*128*8*7168=14.7 MFLOPs
           vs 90TFLOPs expert), 估算时可忽略.
        """
        return 2 * B * K * H

    def _weight_bytes(self, E, H, I, es):
        """路由专家权重记忆体 (HBM 加载量)

        三组权重:
          1. W_gate:      E × H           — router 打分
          2. W_gate_up:   E × (2*I) × H   — 各专家 gate+up 融合
          3. W_down:      E × I × H       — 各专家 down

          总计字节 = E*H*es + E*2*I*H*es + E*I*H*es
                   = E*H*es * (1 + 3*I)

          此权重在每层 MoE 前向时一次性加载到 HBM→SRAM,
          是 memory-bound 场景下的主要瓶颈.
        """
        return (E * H + 3 * E * H * I) * es

    def estimate(self, name, args):
        input_shape = first_shape(parse_shape(args.get('input_shape',
                                        args.get('output_shape', '(1, 1)'))))
        output_shape = first_shape(parse_shape(args['output_shape']))
        dtype = args.get('output_dtype', 'torch.float32')

        B = input_shape[0]
        H = output_shape[-1]

        # ── 读取模型配置 ─────────────────────────────────────────────
        E, top_k, I, G, topk_G = self._get_config(H, args)
        es = get_elem_size(dtype)

        # ── FLOPs ────────────────────────────────────────────────────
        gate_flops = self._gate_flops(B, H, E)
        expert_flops = self._expert_flops(B, top_k, H, I)
        weighted_sum_flops = self._weighted_sum_flops(B, top_k, H)
        total_flops = gate_flops + expert_flops + weighted_sum_flops

        # ── 记忆体 ──────────────────────────────────────────────────
        in_bytes = prod(input_shape) * es
        out_bytes = prod(output_shape) * es
        weight_bytes = self._weight_bytes(E, H, I, es)

        cs = self.chip_specs
        compute_time = total_flops / cs['2d_peak_flops']
        mem_time = (in_bytes + out_bytes + weight_bytes) / cs['memory_bandwidth']
        return max(compute_time, mem_time) * 1e6
