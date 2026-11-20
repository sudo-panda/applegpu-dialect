// RUN: applegpu-translate %s --mlir-to-msl > %t.metal
// RUN: xcrun -sdk macosx metal -c %t.metal -o %t.air
// RUN: xcrun -sdk macosx metallib %t.air -o %t.metallib
// RUN: %run_grid %t.metallib reduce 1 32 f32:32:iota f32:32:fill=0 | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @reduce(%in: memref<32xf32>, %out: memref<32xf32>) kernel {
      %tid = gpu.thread_id x
      %v = memref.load %in[%tid] : memref<32xf32>
      %sum = applegpu.simdgroup_sum %v : f32
      memref.store %sum, %out[%tid] : memref<32xf32>
      gpu.return
    }
  }
}

// The input holds 0..31, so every lane gets 0 + 1 + ... + 31 = 496.
// CHECK-COUNT-32: arg1[{{[0-9]+}}] = 496.0