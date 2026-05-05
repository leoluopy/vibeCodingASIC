# sw — 软件开发

## 内容

| 子目录 | 说明 |
|---|---|
| `compiler/` | AI 编译器 (Triton / TVM / MLIR 接入) |
| `runtime/` | Runtime 与驱动 |
| `models/` | 典型模型推理参考实现 |
| `ecosystem/` | 生态适配 (PyTorch / ONNX 接入) |

## 职责

- AI 编译器全栈编译优化，生成高质量 RISC-V 目标码
- 软件生态适配：对接主流 AI 框架
- Qemu 环境下典型模型软件栈走通与性能验证

## 里程碑对应

- M1: 输出 Software Feature 开发清单
- M2: 逐 Feature 开发编译器 Pass / Runtime 组件
- M3: Qemu 上跑通典型模型 + 性能读出
- M4: 芯片上点亮模型
