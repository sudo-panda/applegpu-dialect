// RUN: applegpu-translate %s --mlir-to-msl > %t.metal
// RUN: xcrun -sdk macosx metal -c %t.metal -o %t.air
// RUN: xcrun -sdk macosx metallib %t.air -o %t.metallib
// RUN: xcrun -sdk macosx swiftc %S/../runners/run_kernel.swift -o %t.runner
// RUN: %t.runner %t.metallib reduce | FileCheck %s

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

// CHECK: GPU:
// CHECK-COUNT-32: = 528.0