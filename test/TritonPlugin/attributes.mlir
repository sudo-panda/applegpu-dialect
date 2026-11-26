// The layout attribute parses and prints back unchanged, here as the layout
// of a kernel argument (printed in place, without an alias).
// RUN: env TRITON_PLUGIN_PATHS=%apple_triton_plugin %triton_opt %s \
// RUN:   --mlir-print-local-scope | FileCheck %s

// CHECK-LABEL: tt.func @layout
// CHECK-SAME: tensor<16x16xf32, #apple_tt.simdgroup_mma<{warpsPerCTA = [2, 2]}>>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @layout(%x: tensor<16x16xf32, #apple_tt.simdgroup_mma<{warpsPerCTA = [2, 2]}>>) {
    tt.return
  }
}
