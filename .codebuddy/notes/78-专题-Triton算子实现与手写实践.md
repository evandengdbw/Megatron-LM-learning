# 专题：Triton 算子实现

> 实践程序：`.codebuddy/triton_tutorial/learn_triton.py`（6 个渐进练习，需 GPU）
> 仓库范例：`fusions/fused_pad_routing_map.py`、`fusions/fused_mla_yarn_rope_apply.py`

## 1. 仓库里的 Triton 版图

`@triton.jit` 分布（main：39 个文件；r0.19.x 分支：28 个文件 / 85 个 kernel）：

| 区域 | 文件 | 用途 |
|---|---|---|
| **SSM / Mamba** | `core/ssm/ops/mamba2/*`、`gdp/*`、`common/*` | Mamba2 状态扫描、Gated DeltaNet 块并行、因果卷积，仓库最复杂 |
| **推理** | `core/inference/moe/*`、`communication/torch_symm_triton/*`、`mtp_utils_triton.py` 等 | MoE permute、**对称内存 + PTX 汇编写集合通信**、KV cache、量化 |
| **训练侧融合** | `core/fusions/fused_mhc_kernels.py`、`fused_pad_routing_map.py`、`fused_mla_yarn_rope_apply.py`、`fused_indices_converter.py` 等 | 真正进训练主循环的算子 |

定位：训练主循环的性能算子优先由 TE / cuDNN / FlashAttention 提供（笔记 75），Triton 填补 TE 未覆盖的缝隙（SSM、MoE permute、推理 KV）。

选读顺序：`fused_pad_routing_map.py`（100 行，入门）→ `fused_mla_yarn_rope_apply.py`（完整形态）→ `inference/moe/permute.py` → `ssm/ops/gdp/*`（进阶）→ `torch_symm_triton/*`（硬核）。

## 2. 标准实现模式（三段式）

```
① 可选依赖守卫    没装 triton 也能 import
② @triton.jit     纯计算，只认指针和标量
③ host wrapper    校验 + 算 grid + 起 kernel
      ↓ 需要梯度
④ torch.autograd.Function
      ↓
⑤ 填进 ModuleSpec.submodules
```

### 可选依赖守卫（`fused_pad_routing_map.py:11-27`）

```python
try:
    import triton, triton.language as tl
    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False
    from unittest import mock
    triton = mock.MagicMock()   # @triton.jit 退化为 no-op 装饰器
    tl = mock.MagicMock()
```

Megatron 是库，不能因用户未装 triton 而 import 失败。`fused_mhc_kernels.py:49` 更进一步：backend 策略（`"auto"/"native"/"triton"/"cutile"`）+ 确定性声明（`:43-47`）。

### kernel 写法要点

| 概念 | 要点 |
|---|---|
| `tl.program_id(axis)` | grid 内坐标，可多维 |
| `tl.arange(0, BLOCK)` | 长度必须 2 的幂，`tl.constexpr` 标注 |
| `mask=` | 越界保护，`tl.load(..., other=0.0)` 配合 |
| 指针算术 | 只认指针 + 显式 stride，`x_ptr + row * N + cols` |
| `tl.dot` | 矩阵乘 tile，累加器 fp32 |
| `num_warps` / `num_stages` | warps 数 / 软件流水级数 |

### host wrapper 职责

校验（is_cuda / contiguous / dtype）→ `triton.next_power_of_2(N)` 算 BLOCK → 算 grid（引用 `meta["BLOCK"]` 时用 lambda）。

### autograd 封装（对照 `fused_bias_swiglu.py:121`）

- `ctx.save_for_backward` 存中间结果而非输入（如 `x+b`，反向省一次加法）
- backward 常需 2 个 kernel：dx（elementwise）+ db（归约）
- 归约优先"分块 + 循环"而非 `atomic_add`（确定性）

## 3. 范例

### `fused_pad_routing_map.py`（100 行，最小骨架）

MoE routing map 补列对齐。仓库唯一"单 kernel、无 autograd、无 autotune"的文件，隔离出三段式最小结构。测试在 `tests/unit_tests/fusions/`。

### `fused_mla_yarn_rope_apply.py`（完整形态）

MLA 的 RoPE 应用（含 YARN），叠加四个进阶要素：`@triton.autotune`（配置列表 + `key` 缓存粒度）、`torch.autograd.Function`、stride 显式传参（非连续张量）、原地写入。测试 `tests/unit_tests/fusions/test_mla_yarn_rope_apply.py`。

## 4. 实践程序

`.codebuddy/triton_tutorial/learn_triton.py`，GPU 环境运行：

```bash
python learn_triton.py       # 全部
python learn_triton.py 3     # 只跑练习 3
```

| 练习 | 学什么 | 对应仓库文件 |
|---|---|---|
| Ex1 向量加 | program_id / block / mask | — |
| Ex2 融合 bias+GELU | 2D 行式加载、广播、融合省带宽 | `fused_pad_routing_map.py` |
| Ex3 autograd | 反向 kernel + 归约 kernel | `fused_bias_swiglu.py:121` |
| Ex4 autotune | Config 列表 + key | `fused_mla_yarn_rope_apply.py` |
| Ex5 矩阵乘 | 二维 tile、tl.dot、num_stages | `ssm/ops/gdp/chunk_h.py` |
| Ex6 Megatron 封装 | HAVE_TRITON 守卫 + spec 接入 | `fused_pad_routing_map.py:11-27` |

每个练习与 torch 参考数值比对并打印加速比。

常见坑：`tl.arange` 必须 2 的幂（`next_power_of_2`）；grid 引用 meta 用 lambda；`tl.dot` 累加器 fp32；超参常数放模块级；归约别急用 `atomic_add`。

调试：`TRITON_INTERPRET=1` 纯解释模式逐行看中间值。
