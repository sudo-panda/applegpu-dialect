// RUN: env TRITON_PLUGIN_PATHS=%apple_triton_plugin %triton_opt %s \
// RUN:   -split-input-file -verify-diagnostics

// expected-error @+1 {{simdgroup_mma supports rank 2 only, got warpsPerCTA of size 3}}
#mma = #apple_tt.simdgroup_mma<{warpsPerCTA = [1, 2, 2]}>

// -----

// expected-error @+1 {{warpsPerCTA must be powers of 2}}
#mma = #apple_tt.simdgroup_mma<{warpsPerCTA = [3, 1]}>

// -----

// Triton checks every tensor layout against its module through the layout's
// dialect; ours forwards that check to TritonGPU. Here the layout spreads
// over 2 x 4 = 8 SIMD-groups, but the kernel has 4.
#mma = #apple_tt.simdgroup_mma<{warpsPerCTA = [2, 4]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @wrong_warp_count(%x: tensor<16x16xf32, #mma>) -> tensor<16x16xf32, #mma> {
    %y = arith.addf %x, %x : tensor<16x16xf32, #mma>
    // expected-error @+1 {{Layout has 8 warps per CTA, but the context requires 4 warps per CTA.}}
    tt.return %y : tensor<16x16xf32, #mma>
  }
}
