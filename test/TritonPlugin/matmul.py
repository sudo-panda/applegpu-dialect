# apple-accelerate-matmul on a plain GEMM (fp16 in, f32 accumulate, K loop).
# The dot and its accumulator take the simdgroup_matrix layout; A and B reach
# it through threadgroup memory, and the only layout conversion left is the
# one before the store. The TTGIR, with the plugin's layout in it, survives a
# round trip through applegpu-opt unchanged.
#
# RUN: %triton_to_ttgir %s matmul_kernel --plugin %apple_triton_plugin \
# RUN:   --signature 'a_ptr=*fp16,b_ptr=*fp16,c_ptr=*fp16,M=i32,N=i32,K=i32,stride_am=i32,stride_bk=i32,stride_cm=i32' \
# RUN:   --constexpr 'stride_ak=1,stride_bn=1,stride_cn=1,BLOCK_M=64,BLOCK_N=64,BLOCK_K=32,GROUP_M=8' \
# RUN:   --divisible M,N,K,stride_am,stride_bk,stride_cm > %t.ttgir
# RUN: FileCheck %s < %t.ttgir
# RUN: grep -c '"ttg.convert_layout"' %t.ttgir | FileCheck %s --check-prefix=CONVERSIONS
# RUN: applegpu-opt --allow-unregistered-dialect --mlir-print-op-generic \
# RUN:   --mlir-print-local-scope --mlir-print-debuginfo %t.ttgir | diff %t.ttgir -
#
# CHECK: "ttg.local_load"{{.*}} -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #apple_tt.simdgroup_mma<{warpsPerCTA = [2, 2]}>}>>
# CHECK: "ttg.local_load"{{.*}} -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #apple_tt.simdgroup_mma<{warpsPerCTA = [2, 2]}>}>>
# CHECK: "tt.dot"{{.*}} -> tensor<64x64xf32, #apple_tt.simdgroup_mma<{warpsPerCTA = [2, 2]}>>
# CHECK: "scf.yield"({{.*}}) : (tensor<64x64xf32, #apple_tt.simdgroup_mma<{warpsPerCTA = [2, 2]}>>
# CHECK: "ttg.convert_layout"{{.*}} : (tensor<64x64xf16, #apple_tt.simdgroup_mma<{warpsPerCTA = [2, 2]}>>) -> tensor<64x64xf16, #ttg.blocked
# CHECK: "tt.store"
# CONVERSIONS: 1

import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  GROUP_M: tl.constexpr):
    # Programs in groups of GROUP_M rows of blocks, so that neighbours reuse B.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    group = pid // (GROUP_M * num_pid_n)
    first_m = group * GROUP_M
    group_size = min(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group_size)
    pid_n = (pid % (GROUP_M * num_pid_n)) // group_size

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_left = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_left, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_left, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float16)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(rows[:, None] < M) & (cols[None, :] < N))
