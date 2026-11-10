// RUN: applegpu-opt %s | applegpu-opt | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @kernels {
    // CHECK-LABEL: gpu.func @reduce
    gpu.func @reduce(%in: memref<32xf32>, %out: memref<32xf32>) kernel {
      %tid = gpu.thread_id x
      %v = memref.load %in[%tid] : memref<32xf32>
      // CHECK: applegpu.simdgroup_sum %{{.*}} : f32
      %sum = applegpu.simdgroup_sum %v : f32
      memref.store %sum, %out[%tid] : memref<32xf32>
      gpu.return
    }
  }
}