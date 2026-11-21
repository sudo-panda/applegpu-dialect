// test/runners/triton_to_ttgir.py on vector add: Triton's passes choose the
// layout, and the TTGIR survives a round trip through applegpu-opt unchanged.
//
// RUN: %triton_to_ttgir %S/../Integration/Triton/vector_add.py add_kernel \
// RUN:   --signature 'x_ptr=*fp32,y_ptr=*fp32,out_ptr=*fp32,n=i32' \
// RUN:   --constexpr BLOCK=512 --num-warps 4 > %t.ttgir
// RUN: FileCheck %s < %t.ttgir
// RUN: applegpu-opt --allow-unregistered-dialect --mlir-print-op-generic \
// RUN:   --mlir-print-local-scope --mlir-print-debuginfo %t.ttgir | diff %t.ttgir -

// Pointers are marked 16-byte aligned, so coalesce gives each of the 128
// threads 4 adjacent floats (one 128-bit access) of the 512-element block.
// CHECK: arg_attrs = [{tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {}]
// CHECK: "tt.load"
// CHECK-SAME: #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK: "ttg.num-warps" = 4 : i32, ttg.target = "apple:m1", "ttg.threads-per-warp" = 32 : i32
