"""
Puzzle 02: Vector Add
==============
This puzzle asks you to implement a vector addition operation.

Category: ["official"]
Difficulty: ["easy"]
"""

import tilelang
import tilelang.language as T
import torch

from common.utils import bench_puzzle, test_puzzle

"""
Vector addition is our first step towards computation. Tilelang provides basic arithmetic
operations like add, sub, mul, div, etc. But these operations are element-wise (They are not
TileOps like T.copy). So we need a loop abstraction to iterate over elements in the tensor.
Inside the loop body, we can perform whatever computation we want.

02-1: 1-D vector addition.

Inputs:
    A: Tensor([N,], float16)  # input tensor
    B: Tensor([N,], float16)  # input tensor
    N: int   # size of the tensor. 1 <= N <= 1024*1024

Output:
    C: Tensor([N,], T.float16)  # output tensor

Definition:
    for i in range(N):
        C[i] = A[i] + B[i]
"""


def ref_add_1d(A: torch.Tensor, B: torch.Tensor):
    assert len(A.shape) == 1
    assert len(B.shape) == 1
    assert A.shape[0] == B.shape[0]
    assert A.dtype == B.dtype == torch.float16
    return A + B


@tilelang.jit
def tl_add_1d(A, B, BLOCK_N: int):
    N = T.const("N")
    A: T.Tensor((N,), T.float16)
    B: T.Tensor((N,), T.float16)
    C = T.empty((N,), T.float16)

    with T.Kernel(N // BLOCK_N, threads=256) as bx:
        base_idx = bx * BLOCK_N
        for i in T.Parallel(BLOCK_N):
            C[base_idx + i] = A[base_idx + i] + B[base_idx + i]

    return C


def run_add_1d():
    print("\n=== Vector Add 1D ===\n")
    N = 1024 * 256
    BLOCK_N = 1024
    test_puzzle(tl_add_1d, ref_add_1d, {"N": N, "BLOCK_N": BLOCK_N})


"""
We can fuse more elementwise operations into this kernel.
Now that's do an element-wise multiplication with a ReLU activation.

HINT: We can use T.if_then_else(cond, true_value, false_value) to implement conditional logic.

02-2: 1-D vector multiplication with ReLU activation

Inputs:
    A: Tensor([N,], float16)  # input tensor
    B: Tensor([N,], float16)  # input tensor
    N: int   # size of the tensor. 1 <= N <= 1024*1024

Output:
    C: Tensor([N,], T.float16)  # output tensor

Output:
    C: [N,]  # output tensor

Definition:
    for i in range(N):
        C[i] = max(0, A[i] * B[i])
"""


def ref_mul_relu_1d(A: torch.Tensor, B: torch.Tensor):
    assert len(A.shape) == 1
    assert len(B.shape) == 1
    assert A.shape[0] == B.shape[0]
    assert A.dtype == B.dtype == torch.float16
    return (A * B).relu_()


@tilelang.jit
def tl_mul_relu_1d(A, B, BLOCK_N: int):
    N = T.const("N")
    A: T.Tensor((N,), T.float16)
    B: T.Tensor((N,), T.float16)
    C = T.empty((N,), T.float16)

    with T.Kernel(N // BLOCK_N, threads=256) as bx:
        base_idx = bx * BLOCK_N
        for i in T.Parallel(BLOCK_N):
            C[base_idx + i] = T.if_then_else(
                    A[base_idx + i] * B[base_idx + i] > 0,
                    A[base_idx + i] * B[base_idx + i],
                    0,
                    )

    return C


def run_mul_relu_1d():
    print("\n=== Vector Multiplication with ReLU 1D ===\n")
    N = 1024 * 256
    BLOCK_N = 1024
    test_puzzle(tl_mul_relu_1d, ref_mul_relu_1d, {"N": N, "BLOCK_N": BLOCK_N})


"""
NOTE: This section needs some understanding of GPU memory hierarchy and basic CUDA
programming knowledge.

We can further optimize the previous example. Here, we introduce a common optimization technique
used in kernel programming. If you have experience with CUDA or other GPU programming frameworks,
you are likely aware of the memory hierarchy on GPUs.

Typically, there are three main levels of memory: global memory (DRAM), shared memory, and
registers. Registers are the fastest but also the smallest form of memory. In CUDA, registers are
allocated when you declare local variables within a kernel.

Our previous implementation loads data directly from A and B and stores the result to C, where A, B,
and C are all passed as global memory pointers. This is inefficient because it requires accessing
global memory for every single element. You can use print_source_code() to inspect the generated
CUDA code.

Here, we consider using registers to optimize the kernel. The key idea is to copy multiple data
elements between registers and global memory in a single operation. For example, CUDA often uses
ldg128 to load 128 bits of data from global memory into registers at once, which can theoretically
reduce the number of memory accesses by 4x.

In our fused kernel example, intermediate results from A * B can also be stored in registers. When
applying the ReLU operation, we can read directly from registers instead of global memory. (In
practice, this may not need to be done explicitly—it can often be optimized automatically by
NVCC through common subexpression elimination, or CSE.)
"""

"""
TileLang explicitly exposes these memory levels to users. You can use `T.alloc_fragment`
to allocate a fragment of registers. Note that when you write CUDA, registers are thread-local.
So when you write programs, you usually need to handle some logics to make sure each thread load
certain part of the data into registers. But in TileLang, you don't need to do such mappings.
A fragment is an abstraction of registers in all threads in a block. We can manipulate this
fragment in a unified way as we do to a T.Buffer.
"""


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_mul_relu_1d_mem(A, B, BLOCK_N: int):
    """
    Essentially write from GMEM straight away to register file

    Use T.alloc_fragment(shape, dtype) to first allocate BLOCK_N space in register
    Perform required sum + ReLU
    Accumulate in register
    Write back to GMEM
    """

    N = T.const("N")
    dtype = T.float16
    A: T.Tensor((N,), dtype)
    B: T.Tensor((N,), dtype)
    C = T.empty((N,), dtype)

    with T.Kernel(N // BLOCK_N, threads=256) as bx:
        base_idx = bx * BLOCK_N
        A_register = T.alloc_fragment((BLOCK_N), dtype) # Allocate BLOCK_N space in register
        B_register = T.alloc_fragment((BLOCK_N), dtype) # Allocate BLOCK_N space in register
        C_register = T.alloc_fragment((BLOCK_N), dtype) # Allocate BLOCK_N space in register

        T.copy(A[base_idx], A_register)
        T.copy(B[base_idx], B_register)

        for i in T.Parallel(BLOCK_N):
            C_register[i] = A_register[i] * B_register[i]
            C_register[i] = T.if_then_else(C_register[i] > 0, C_register[i], 0)

        T.copy(C_register, C[base_idx])


    return C


def run_mul_relu_1d_mem():
    print("\n=== Vector Multiplication with ReLU 1D (Memory Optimized) ===\n")
    N = 1024 * 4096
    BLOCK_N = 1024

    print("Naive TL Implementation: ")
    tl_mul_relu_kernel = tl_mul_relu_1d.compile(N=N, BLOCK_N=BLOCK_N)
    tl_mul_relu_kernel.print_source_code()

    print("Optimized Version")
    tl_mul_relu_kernel_opt = tl_mul_relu_1d_mem.compile(N=N, BLOCK_N=BLOCK_N)
    tl_mul_relu_kernel_opt.print_source_code()

    test_puzzle(tl_mul_relu_1d_mem, ref_mul_relu_1d, {"N": N, "BLOCK_N": BLOCK_N})
    bench_puzzle(
        tl_mul_relu_1d,
        ref_mul_relu_1d,
        {"N": N, "BLOCK_N": BLOCK_N},
        bench_name="TL Naive",
        bench_torch=True,
    )
    bench_puzzle(
        tl_mul_relu_1d_mem,
        ref_mul_relu_1d,
        {"N": N, "BLOCK_N": BLOCK_N},
        bench_name="TL OPT",
        bench_torch=False,
    )


if __name__ == "__main__":
    run_add_1d()
    run_mul_relu_1d()
    run_mul_relu_1d_mem()

# Results
# === Vector Add 1D ===

# 2026-04-28 07:36:30  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:133): TileLang begins to compile kernel `tl_add_1d` with `out_idx=[-1]`
# 2026-04-28 07:36:33  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:141): TileLang completes to compile kernel `tl_add_1d`
# ✅ Results match: True

# === Vector Multiplication with ReLU 1D ===

# 2026-04-28 07:36:34  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:133): TileLang begins to compile kernel `tl_mul_relu_1d` with `out_idx=[-1]`
# 2026-04-28 07:36:37  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:141): TileLang completes to compile kernel `tl_mul_relu_1d`
# ✅ Results match: True

# === Vector Multiplication with ReLU 1D (Memory Optimized) ===

# Naive TL Implementation:
# 2026-04-28 07:36:38  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:133): TileLang begins to compile kernel `tl_mul_relu_1d` with `out_idx=[-1]`
# 2026-04-28 07:36:42  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:141): TileLang completes to compile kernel `tl_mul_relu_1d`
# 2026-04-28 07:36:42  [TileLang:tilelang.jit.kernel:WARNING] (kernel.py:562): print_source_code is deprecated; use show_source() or export_sources() instead.
# #include <tl_templates/cuda/gemm.h>
# #include <tl_templates/cuda/copy.h>
# #include <tl_templates/cuda/reduce.h>
# #include <tl_templates/cuda/ldsm.h>
# #include <tl_templates/cuda/threadblock_swizzle.h>
# #include <tl_templates/cuda/debug.h>
# #ifdef ENABLE_BF16
# #include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
# #endif

# extern "C" __global__ void tl_mul_relu_1d_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C);
# extern "C" __global__ void __launch_bounds__(256, 1) tl_mul_relu_1d_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C) {
#   for (int i_s = 0; i_s < 4; ++i_s) {
#     half_t condval;
#     if ((half_t(0x0p+0f/*0.000000e+00*/) < (A[(((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 4)) + i_s)] * B[(((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 4)) + i_s)]))) {
#       condval = (A[(((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 4)) + i_s)] * B[(((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 4)) + i_s)]);
#     } else {
#       condval = half_t(0x0p+0f/*0.000000e+00*/);
#     }
#     C[(((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 4)) + i_s)] = condval;
#   }
# }


# Optimized Version
# 2026-04-28 07:36:42  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:133): TileLang begins to compile kernel `tl_mul_relu_1d_mem` with `out_idx=[-1]`
# 2026-04-28 07:36:45  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:141): TileLang completes to compile kernel `tl_mul_relu_1d_mem`
# 2026-04-28 07:36:45  [TileLang:tilelang.jit.kernel:WARNING] (kernel.py:562): print_source_code is deprecated; use show_source() or export_sources() instead.
# #include <tl_templates/cuda/gemm.h>
# #include <tl_templates/cuda/copy.h>
# #include <tl_templates/cuda/reduce.h>
# #include <tl_templates/cuda/ldsm.h>
# #include <tl_templates/cuda/threadblock_swizzle.h>
# #include <tl_templates/cuda/debug.h>
# #ifdef ENABLE_BF16
# #include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
# #endif

# extern "C" __global__ void tl_mul_relu_1d_mem_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C);
# extern "C" __global__ void __launch_bounds__(256, 1) tl_mul_relu_1d_mem_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C) {
#   half_t A_register[4];
#   half_t B_register[4];
#   half_t C_register[4];
#   *(uint2*)(A_register + 0) = *(uint2*)(A + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 4)));
#   *(uint2*)(B_register + 0) = *(uint2*)(B + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 4)));
#   #pragma unroll
#   for (int i = 0; i < 4; ++i) {
#     C_register[i] = (A_register[i] * B_register[i]);
#     half_t condval;
#     if ((half_t(0x0p+0f/*0.000000e+00*/) < C_register[i])) {
#       condval = C_register[i];
#     } else {
#       condval = half_t(0x0p+0f/*0.000000e+00*/);
#     }
#     C_register[i] = condval;
#   }
#   *(uint2*)(C + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 4))) = *(uint2*)(C_register + 0);
# }


# 2026-04-28 07:36:45  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:133): TileLang begins to compile kernel `tl_mul_relu_1d_mem` with `out_idx=[-1]`
# 2026-04-28 07:36:49  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:141): TileLang completes to compile kernel `tl_mul_relu_1d_mem`
# ✅ Results match: True
# 2026-04-28 07:36:49  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:133): TileLang begins to compile kernel `tl_mul_relu_1d` with `out_idx=[-1]`
# 2026-04-28 07:36:53  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:141): TileLang completes to compile kernel `tl_mul_relu_1d`
# Torch time: 0.010 ms
# TL Naive time: 0.007 ms
# 2026-04-28 07:36:53  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:133): TileLang begins to compile kernel `tl_mul_relu_1d_mem` with `out_idx=[-1]`
# 2026-04-28 07:36:57  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:141): TileLang completes to compile kernel `tl_mul_relu_1d_mem`
# TL OPT time: 0.005 ms
