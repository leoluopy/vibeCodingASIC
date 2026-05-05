# hw — 硬件设计

## 内容

| 子目录 | 说明 |
|---|---|
| `rtl/` | RISC-V AI 加速器 RTL 源码 (Verilog/SystemVerilog) |
| `verif/` | IC 验证环境 (UVM / 形式验证 / 覆盖率驱动) |
| `pd/` | 物理设计脚本 (综合、布局布线、时序收敛) |

## 职责

- 根据 `docs/spec/` 中的硬件规格完成 RTL 编码（VibeCoding 辅助）
- 搭建 UVM 验证环境，达到 ASIC 100% 验证目标
- 物理设计，确保时序收敛

## 里程碑对应

- M1: 根据 spec 输出 RTL 开发清单
- M2: RTL 逐模块开发 + 单元验证
- M3: ASIC 100% 验证通过
- M4: 流片
