#!/usr/bin/env python3
"""Triton 算子手写实践（渐进式 6 个练习）。

对应笔记：.codebuddy/notes/78-专题-Triton算子实现与手写实践.md

运行（需要 NVIDIA GPU + torch + triton）：
    python learn_triton.py            # 全部练习
    python learn_triton.py 3          # 只跑练习 3

无 GPU 时脚本会打印提示并退出（Triton 必须有 CUDA）。

练习路线：
    Ex1 向量加       —— 掌握 program_id / 一维 block / mask 边界
    Ex2 融合 bias+gelu —— 掌握 2D 行式加载、广播、融合多个 torch op 的意义
    Ex3 autograd 封装 —— 掌握反向 kernel + torch.autograd.Function（Megatron 的标准套路）
    Ex4 autotune    —— 掌握 @triton.autotune 配置搜索
    Ex5 矩阵乘       —— 掌握二维 tile、tl.dot、累加器、stride 指针算术
    Ex6 Megatron 风格封装 —— 把 Ex2/Ex3 包装成可直接替换进 Megatron 的模块
"""

import sys
import time

import torch
import triton
import triton.language as tl

# GELU(tanh 近似) 用的常数：c = sqrt(2/pi)
GELU_C = 0.7978845608028654
GELU_CUBIC = 0.044715


# ===========================================================================
# 工具
# ===========================================================================
def check(name, got, want, atol=1e-5, print_shape=True):
    """与 torch 参考实现比对。"""
    got = got.float()
    want = want.float()
    diff = (got - want).abs().max().item()
    ok = diff <= atol and torch.isfinite(got).all()
    shape = f"{tuple(got.shape)} " if print_shape else ""
    print(f"  {'[OK]  ' if ok else '[FAIL]'} {name}: {shape}max_abs_diff={diff:.3e}")
    return ok


def bench(fn, *args, n_warmup=10, n_repeat=100):
    """GPU 计时（毫秒）。"""
    for _ in range(n_warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_repeat):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_repeat


def title(n, text):
    print(f"\n{'=' * 66}\n练习 {n}: {text}\n{'=' * 66}")


def gelu_tanh(x):
    """torch 侧参考：与 kernel 里的 tanh 近似逐位一致。"""
    return torch.nn.functional.gelu(x, approximate="tanh")


# ===========================================================================
# Ex1: 向量加 —— Triton 的 "hello world"
# ===========================================================================
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 1) 我是哪个程序实例？（grid 里的第几个 block）
    pid = tl.program_id(axis=0)
    # 2) 我负责哪些元素？
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 3) 越界保护：n 不是 BLOCK_SIZE 整数倍时，最后一块要 mask
    mask = offsets < n_elements
    # 4) 捞数据 -> 算 -> 写回
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = out.numel()
    # grid 是"启动多少个程序实例"。写成 lambda 才能引用 meta['BLOCK_SIZE']
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024, num_warps=4)
    return out


def ex1():
    title(1, "向量加 —— program_id / block / mask 三件套")
    n = 98473  # 故意不是 1024 的整数倍，用来验证 mask
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")

    out = add(x, y)
    ok = check("add vs torch", out, x + y)

    ms_tri = bench(add, x, y)
    ms_ref = bench(torch.add, x, y)
    print(f"  triton {ms_tri:.4f} ms | torch {ms_ref:.4f} ms | speedup {ms_ref / ms_tri:.2f}x")
    print("  （注意：向量加是访存瓶颈，Triton 通常和 torch 打平——它的价值在 Ex2 之后的融合）")
    return ok


# ===========================================================================
# Ex2: 融合 bias + GELU —— 2D 行式处理与"融合"的真正意义
# ===========================================================================
# 目标：out = gelu(x + bias)，bias 是长度 N 的向量，按列广播。
# torch 要 3 个 kernel（add / gelu / 各一次读写）；Triton 一个 kernel 一次读写。


@triton.jit
def _gelu(u):
    """tanh 近似 GELU，只用 tl.exp 实现 tanh，避免依赖 libdevice 命名空间。"""
    # tanh(u) = (e^{2u} - 1) / (e^{2u} + 1)
    e = tl.exp(2.0 * u)
    t = (e - 1.0) / (e + 1.0)
    return 0.5 * u * (1.0 + t)


@triton.jit
def bias_gelu_fwd_kernel(
    x_ptr, bias_ptr, out_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(axis=0)          # 每个程序实例处理矩阵的一整行
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N                       # N 可能不是 BLOCK_N 整数倍
    offs = row * N + cols                 # 行主序展平：row * N + col

    x = tl.load(x_ptr + offs, mask=mask)
    b = tl.load(bias_ptr + cols, mask=mask)   # 同一行的所有元素共享 bias
    y = _gelu(x + b)
    tl.store(out_ptr + offs, y, mask=mask)


def bias_gelu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(N)   # tl.arange 的长度必须是 2 的幂
    grid = (M,)                            # 一行为一个程序实例
    bias_gelu_fwd_kernel[grid](x, bias, out, M, N, BLOCK_N=BLOCK_N, num_warps=4)
    return out


def ex2():
    title(2, "融合 bias+GELU —— 一次读写干完三个 op")
    M, N = 4096, 1024
    x = torch.randn(M, N, device="cuda")
    b = torch.randn(N, device="cuda")

    out = bias_gelu(x, b)
    ok = check("bias_gelu vs torch", out, gelu_tanh(x + b), atol=2e-5)

    ms_tri = bench(bias_gelu, x, b)

    def torch_ref(x, b):
        return gelu_tanh(x + b)

    ms_ref = bench(torch_ref, x, b)
    print(f"  triton {ms_tri:.4f} ms | torch {ms_ref:.4f} ms | speedup {ms_ref / ms_tri:.2f}x")
    print("  省的是显存带宽：torch 读 x 写中间结果再读再写，Triton 只读一次写一次。")
    return ok


# ===========================================================================
# Ex3: autograd 封装 —— 反向 kernel + torch.autograd.Function
# 这是 Megatron 里每个融合算子的标准写法（对照 fusions/fused_bias_swiglu.py）
# ===========================================================================
# 前向:  s = x + b;  y = gelu(s)
# 反向:  dx = dy * gelu'(s)
#        db = sum_over_rows(dy * gelu'(s))


@triton.jit
def _dgelu(u):
    """gelu'(u)（tanh 近似解析导数）。
    y = 0.5*u*(1+t),  t = tanh(c*(u + k*u^3))
    y' = 0.5*(1+t) + 0.5*u*(1-t^2)*c*(1+3k*u^2)
    """
    inner = GELU_C * (u + GELU_CUBIC * u * u * u)
    e = tl.exp(2.0 * inner)
    t = (e - 1.0) / (e + 1.0)
    return 0.5 * (1.0 + t) + 0.5 * u * (1.0 - t * t) * GELU_C * (1.0 + 3.0 * GELU_CUBIC * u * u)


@triton.jit
def bias_gelu_bwd_dx_kernel(dy_ptr, s_ptr, dx_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    dy = tl.load(dy_ptr + offs, mask=mask)
    s = tl.load(s_ptr + offs, mask=mask)     # 前向保存的 x+b，避免重算
    tl.store(dx_ptr + offs, dy * _dgelu(s), mask=mask)


@triton.jit
def bias_gelu_bwd_db_kernel(
    dy_ptr, s_ptr, db_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """db = 沿行维归约。每个程序实例负责 BLOCK_N 列，循环遍历所有行。"""
    pid_n = tl.program_id(axis=0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for m0 in range(0, M, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        mask2 = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        offs = offs_m[:, None] * N + offs_n[None, :]
        dy = tl.load(dy_ptr + offs, mask=mask2, other=0.0)
        s = tl.load(s_ptr + offs, mask=mask2, other=0.0)
        acc += tl.sum(dy * _dgelu(s), axis=0)   # 沿 axis=0（行）求和

    tl.store(db_ptr + offs_n, acc, mask=mask_n)


class BiasGeluFunction(torch.autograd.Function):
    """torch.autograd.Function 是 Triton kernel 接入 PyTorch 自动微分的标准桥梁。"""

    @staticmethod
    def forward(ctx, x, bias):
        M, N = x.shape
        out = torch.empty_like(x)
        s = torch.empty_like(x)          # 保存给反向用
        BLOCK_N = triton.next_power_of_2(N)

        @triton.jit
        def _fwd(x_ptr, bias_ptr, out_ptr, s_ptr, M, N, BLOCK_N: tl.constexpr):
            row = tl.program_id(axis=0)
            cols = tl.arange(0, BLOCK_N)
            mask = cols < N
            offs = row * N + cols
            xv = tl.load(x_ptr + offs, mask=mask)
            bv = tl.load(bias_ptr + cols, mask=mask)
            sv = xv + bv
            tl.store(s_ptr + offs, sv, mask=mask)
            tl.store(out_ptr + offs, _gelu(sv), mask=mask)

        _fwd[(M,)](x, bias, out, s, M, N, BLOCK_N=BLOCK_N, num_warps=4)
        ctx.save_for_backward(s)
        ctx.shape = (M, N)
        return out

    @staticmethod
    def backward(ctx, dy):
        (s,) = ctx.saved_tensors
        M, N = ctx.shape
        dy = dy.contiguous()
        dx = torch.empty_like(dy)
        db = torch.empty(N, device=dy.device, dtype=torch.float32)

        n = M * N
        grid_dx = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
        bias_gelu_bwd_dx_kernel[grid_dx](dy, s, dx, n, BLOCK=1024, num_warps=4)

        BLOCK_N = min(triton.next_power_of_2(N), 1024)
        grid_db = (triton.cdiv(N, BLOCK_N),)
        bias_gelu_bwd_db_kernel[grid_db](
            dy, s, db, M, N, BLOCK_N=BLOCK_N, BLOCK_M=128, num_warps=4
        )
        return dx, db.to(s.dtype)


def bias_gelu_autograd(x, bias):
    return BiasGeluFunction.apply(x, bias)


def ex3():
    title(3, "autograd 封装 —— 反向 kernel + torch.autograd.Function")
    M, N = 1024, 512
    x = torch.randn(M, N, device="cuda", requires_grad=True)
    b = torch.randn(N, device="cuda", requires_grad=True)
    dy = torch.randn(M, N, device="cuda")

    y = bias_gelu_autograd(x, b)
    y.backward(dy)

    # torch 参考
    x2 = x.detach().clone().requires_grad_(True)
    b2 = b.detach().clone().requires_grad_(True)
    y2 = gelu_tanh(x2 + b2)
    y2.backward(dy)

    ok_f = check("forward", y, y2, atol=2e-5)
    ok_dx = check("grad wrt x", x.grad, x2.grad, atol=2e-4)
    ok_db = check("grad wrt bias", b.grad, b2.grad, atol=2e-3)
    print("  要点：ctx.save_for_backward(s) 保存的是 x+b（而非 x），反向直接省一次加法。")
    print("  db 的归约用「列分块 + 行循环」，避免 atomic_add 的非确定性。")
    return ok_f and ok_dx and ok_db


# ===========================================================================
# Ex4: autotune —— 让 Triton 自己搜 BLOCK_SIZE / num_warps
# ===========================================================================
def _bias_gelu_configs():
    """Triton 官方推荐写法：小 block 多 warps、大 block 少 warps 都要试。"""
    return [
        triton.Config({"BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_N": 1024}, num_warps=8),
        triton.Config({"BLOCK_N": 2048}, num_warps=8),
    ]


@triton.autotune(configs=_bias_gelu_configs(), key=["N"])
@triton.jit
def bias_gelu_autotune_kernel(
    x_ptr, bias_ptr, out_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    offs = row * N + cols
    x = tl.load(x_ptr + offs, mask=mask)
    b = tl.load(bias_ptr + cols, mask=mask)
    tl.store(out_ptr + offs, _gelu(x + b), mask=mask)


def bias_gelu_tuned(x, bias):
    M, N = x.shape
    out = torch.empty_like(x)
    # autotune 会临时把 BLOCK_N 写进 meta，所以 grid 仍用 lambda
    bias_gelu_autotune_kernel[(M,)](x, bias, out, M, N)
    return out


def ex4():
    title(4, "autotune —— key 决定缓存粒度")
    M, N = 4096, 1024
    x = torch.randn(M, N, device="cuda")
    b = torch.randn(N, device="cuda")

    out = bias_gelu_tuned(x, b)
    ok = check("autotuned vs torch", out, gelu_tanh(x + b), atol=2e-5)

    best = bias_gelu_autotune_kernel.best_config
    print(f"  选中的配置: {best}")
    print(f"  key=['N'] 的含义：按 N 缓存最优配置；换 N 会重新搜索一次。")
    print("  代价：首次调用会跑一遍所有配置（几百 ms），Megatron 里通常配合 warmup。")
    return ok


# ===========================================================================
# Ex5: 矩阵乘 —— 二维 tile、tl.dot、累加器、stride 指针算术
# ===========================================================================
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 指针块：一次算好首地址，循环里只做偏移量递增（比每次重算便宜）
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)   # 累加器必须 fp32
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def matmul(a, b, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))   # 二维 grid
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    return c


def ex5():
    title(5, "矩阵乘 —— 二维 tile + tl.dot + 归约维循环")
    M, N, K = 512, 512, 1024
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    c = matmul(a, b)
    ref = torch.mm(a, b)
    ok = check("matmul vs torch.mm", c, ref, atol=5e-2)

    ms_tri = bench(matmul, a, b, n_repeat=50)
    ms_ref = bench(torch.mm, a, b, n_repeat=50)
    tflops = 2 * M * N * K / (ms_tri * 1e-3) / 1e12
    print(f"  triton {ms_tri:.4f} ms ({tflops:.1f} TFLOPS) | cuBLAS {ms_ref:.4f} ms")
    print("  教学版没有做 shared memory 流水线优化，打不过 cuBLAS 是正常的。")
    print("  试着改 BLOCK_M/N/K 和 num_stages 看变化——这就是 Megatron/TE 调优的日常。")
    return ok


# ===========================================================================
# Ex6: Megatron 风格封装 —— 可选依赖 + 原样接入
# 对照 megatron/core/fusions/fused_pad_routing_map.py 的 HAVE_TRITON 模式
# ===========================================================================
def ex6():
    title(6, "Megatron 风格封装 —— 把算子变成可插拔模块")
    print("""
  Megatron 里每个 Triton 算子都遵循同一个三段式（以 fused_pad_routing_map.py 为模板）：

    # ① 可选依赖守卫：没装 triton 也能 import
    try:
        import triton
        import triton.language as tl
        HAVE_TRITON = True
    except ImportError:
        HAVE_TRITON = False
        from unittest import mock
        triton = mock.MagicMock()      # 让 @triton.jit 变成 no-op 装饰器
        tl = mock.MagicMock()

    # ② @triton.jit kernel（本文件 Ex2/Ex3 已写）

    # ③ host wrapper：做 dtype/连续性校验 + 算 grid + 起 kernel
    def fused_op(x, bias):
        assert x.is_cuda
        ...
        return out

  接入路径（笔记 30 的 spec 机制）：
    fused_op 写好后，在 xxx_layer_specs.py 里把它填进 ModuleSpec 的
    submodules（如 self_attn_bda / mlp_bda），或包成 torch.autograd.Function
    后放进 ModuleSpec(module=YourFunction)。整棵树不需要改一行。

  验证（对照 tests/unit_tests/fusions/）：
    1. 与 torch 参考实现比对前向 + 反向（本脚本 Ex2/Ex3 已做）
    2. golden values 固化，防回归
    3. 只在 HAVE_TRITON 且 torch.cuda.is_available() 时才 skip 掉测试
""")
    return True


# ===========================================================================
def main():
    if not torch.cuda.is_available():
        print("本脚本需要 NVIDIA GPU（Triton 只支持 CUDA/ROCm）。")
        print(f"  当前 torch={torch.__version__}, triton={triton.__version__}, cuda=False")
        print("请在带 GPU 的环境（或本仓库的 docker/ 容器）里运行。")
        sys.exit(1)

    print(f"torch {torch.__version__} | triton {triton.__version__} | "
          f"{torch.cuda.get_device_name(0)}")

    exercises = {1: ex1, 2: ex2, 3: ex3, 4: ex4, 5: ex5, 6: ex6}
    if len(sys.argv) > 1:
        only = int(sys.argv[1])
        exercises = {only: exercises[only]}

    results = {}
    for i in sorted(exercises):
        try:
            results[i] = exercises[i]()
        except Exception as e:
            print(f"  练习 {i} 报错: {type(e).__name__}: {e}")
            results[i] = False

    print(f"\n{'=' * 66}")
    print("汇总: " + "  ".join(
        f"Ex{i}={'PASS' if v else 'FAIL'}" for i, v in sorted(results.items())
    ))
    print("=" * 66)


if __name__ == "__main__":
    main()
