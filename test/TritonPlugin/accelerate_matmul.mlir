// apple-accelerate-matmul on hand-written TTGIR, in the style of Triton's
// test/TritonGPU/accelerate-matmul.mlir: which dots take the
// #apple_tt.simdgroup_mma layout, with which SIMD-group grid, and which stay
// FMA dots.
//
// RUN: env TRITON_PLUGIN_PATHS=%apple_triton_plugin %triton_opt %s \
// RUN:   -split-input-file -apple-accelerate-matmul | FileCheck %s

// A 64x64 f16 dot with 4 SIMD-groups: 8x8 tiles per side, so the grid is
// 2x2. A and B are converted to dot operands of the new layout, the
// accumulator to the layout itself, and the result back to the old layout.
// CHECK: #[[MMA:.+]] = #apple_tt.simdgroup_mma<{warpsPerCTA = [2, 2]}>
// CHECK-LABEL: @f16_64x64
// CHECK: %[[A:.+]] = ttg.convert_layout %arg0 : {{.*}} -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[MMA]]}>>
// CHECK: %[[B:.+]] = ttg.convert_layout %arg1 : {{.*}} -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #[[MMA]]}>>
// CHECK: %[[C:.+]] = ttg.convert_layout %arg2 : {{.*}} -> tensor<64x64xf32, #[[MMA]]>
// CHECK: %[[D:.+]] = tt.dot %[[A]], %[[B]], %[[C]] : {{.*}} -> tensor<64x64xf32, #[[MMA]]>
// CHECK: %[[R:.+]] = ttg.convert_layout %[[D]] : tensor<64x64xf32, #[[MMA]]> -> tensor<64x64xf32, #blocked>
// CHECK: tt.return %[[R]]
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.target" = "apple:m1", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @f16_64x64(%a: tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
                     %b: tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
                     %c: tensor<64x64xf32, #blocked>) -> tensor<64x64xf32, #blocked> {
    %d = tt.dot %a, %b, %c : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    tt.return %d : tensor<64x64xf32, #blocked>
  }
}

// -----

// bf16, 16x64: 2 tiles along M and 8 along N, so all 4 SIMD-groups go
// along N.
// CHECK: #[[MMA:.+]] = #apple_tt.simdgroup_mma<{warpsPerCTA = [1, 4]}>
// CHECK-LABEL: @bf16_16x64
// CHECK: tt.dot {{.*}} -> tensor<16x64xf32, #[[MMA]]>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.target" = "apple:m1", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @bf16_16x64(%a: tensor<16x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
                      %b: tensor<32x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
                      %c: tensor<16x64xf32, #blocked>) -> tensor<16x64xf32, #blocked> {
    %d = tt.dot %a, %b, %c : tensor<16x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x64xf32, #blocked>
    tt.return %d : tensor<16x64xf32, #blocked>
  }
}

// -----

// f32 inputs, 64x16: 8 tiles along M and 2 along N, so all 4 SIMD-groups
// go along M.
// CHECK: #[[MMA:.+]] = #apple_tt.simdgroup_mma<{warpsPerCTA = [4, 1]}>
// CHECK-LABEL: @f32_64x16
// CHECK: tt.dot {{.*}} -> tensor<64x16xf32, #[[MMA]]>
#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.target" = "apple:m1", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @f32_64x16(%a: tensor<64x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
                     %b: tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
                     %c: tensor<64x16xf32, #blocked>) -> tensor<64x16xf32, #blocked> {
    %d = tt.dot %a, %b, %c : tensor<64x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x16xf32, #blocked>
    tt.return %d : tensor<64x16xf32, #blocked>
  }
}

// -----

// Two chained dots, as in attention (S = Q.K^T, then S.V). The first, 32x16,
// is longer along M, so all SIMD-groups go along M: [4, 1]. The second,
// 32x64, reuses that grid. On its own it would get [2, 2], and as a chained
// dot choosing for itself [1, 4], so only reuse gives [4, 1].
// CHECK: #[[MMA:.+]] = #apple_tt.simdgroup_mma<{warpsPerCTA = [4, 1]}>
// CHECK-LABEL: @chained
// CHECK: %[[S:.+]] = tt.dot {{.*}} -> tensor<32x16xf32, #[[MMA]]>
// CHECK: tt.dot {{.*}} -> tensor<32x64xf32, #[[MMA]]>
// CHECK-NOT: apple_tt
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.target" = "apple:m1", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @chained(%q: tensor<32x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
                   %k: tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
                   %v: tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<32x64xf32, #blocked> {
    %zero_s = arith.constant dense<0.000000e+00> : tensor<32x16xf32, #blocked>
    %zero_o = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #blocked>
    %s = tt.dot %q, %k, %zero_s : tensor<32x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x16xf32, #blocked>
    %p = arith.truncf %s : tensor<32x16xf32, #blocked> to tensor<32x16xf16, #blocked>
    %p_op = ttg.convert_layout %p : tensor<32x16xf16, #blocked> -> tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %o = tt.dot %p_op, %v, %zero_o : tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x64xf32, #blocked>
    tt.return %o : tensor<32x64xf32, #blocked>
  }
}

// -----

// Dots that stay FMA dots: an f16 accumulator, fp8 inputs, M not a multiple
// of 8, and a batched 3-D dot.
// CHECK-NOT: apple_tt
// CHECK-LABEL: @f16_accumulator
// CHECK: tt.dot {{.*}} -> tensor<64x64xf16, #blocked>
// CHECK-LABEL: @fp8_inputs
// CHECK: tt.dot {{.*}} -> tensor<64x64xf32, #blocked>
// CHECK-LABEL: @m_not_multiple_of_8
// CHECK: tt.dot {{.*}} -> tensor<4x64xf32, #blocked>
// CHECK-LABEL: @batched
// CHECK: tt.dot {{.*}} -> tensor<2x16x16xf32, #blocked1>
// CHECK-NOT: apple_tt
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1, 4], threadsPerWarp = [1, 4, 8], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
module attributes {"ttg.target" = "apple:m1", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @f16_accumulator(%a: tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
                           %b: tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
                           %c: tensor<64x64xf16, #blocked>) -> tensor<64x64xf16, #blocked> {
    %d = tt.dot %a, %b, %c : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf16, #blocked>
    tt.return %d : tensor<64x64xf16, #blocked>
  }
  tt.func @fp8_inputs(%a: tensor<64x32xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
                      %b: tensor<32x64xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
                      %c: tensor<64x64xf32, #blocked>) -> tensor<64x64xf32, #blocked> {
    %d = tt.dot %a, %b, %c : tensor<64x32xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x64xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    tt.return %d : tensor<64x64xf32, #blocked>
  }
  tt.func @m_not_multiple_of_8(%a: tensor<4x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
                               %b: tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
                               %c: tensor<4x64xf32, #blocked>) -> tensor<4x64xf32, #blocked> {
    %d = tt.dot %a, %b, %c : tensor<4x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<4x64xf32, #blocked>
    tt.return %d : tensor<4x64xf32, #blocked>
  }
  tt.func @batched(%a: tensor<2x16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>,
                   %b: tensor<2x16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>,
                   %c: tensor<2x16x16xf32, #blocked1>) -> tensor<2x16x16xf32, #blocked1> {
    %d = tt.dot %a, %b, %c : tensor<2x16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<2x16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<2x16x16xf32, #blocked1>
    tt.return %d : tensor<2x16x16xf32, #blocked1>
  }
}
