// Triton ops whose result layout follows from their operand's ask the
// operand layout's dialect, which forwards to TritonGPU: tt.reduce gives a
// #ttg.slice of the layout, and tt.broadcast checks that its operand and
// result layouts agree. A softmax row step: max over each row, broadcast
// back, subtract.
// RUN: env TRITON_PLUGIN_PATHS=%apple_triton_plugin %triton_opt %s \
// RUN:   --mlir-print-local-scope | FileCheck %s

// CHECK-LABEL: tt.func @row_max
// CHECK: "tt.reduce"
// CHECK: -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #apple_tt.simdgroup_mma<{warpsPerCTA = [2, 2]}>}>>
// CHECK: tt.expand_dims {{.*}} -> tensor<16x1xf32, #apple_tt.simdgroup_mma<{warpsPerCTA = [2, 2]}>>
// CHECK: tt.broadcast
// CHECK: arith.subf
#mma = #apple_tt.simdgroup_mma<{warpsPerCTA = [2, 2]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @row_max(%x: tensor<16x16xf32, #mma>) -> tensor<16x16xf32, #mma> {
    %max = "tt.reduce"(%x) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %m = arith.maxnumf %a, %b : f32
      tt.reduce.return %m : f32
    }) : (tensor<16x16xf32, #mma>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %col = tt.expand_dims %max {axis = 1 : i32} : tensor<16xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<16x1xf32, #mma>
    %row = tt.broadcast %col : tensor<16x1xf32, #mma> -> tensor<16x16xf32, #mma>
    %y = arith.subf %x, %row : tensor<16x16xf32, #mma>
    tt.return %y : tensor<16x16xf32, #mma>
  }
}
