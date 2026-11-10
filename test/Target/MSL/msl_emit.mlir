// RUN: applegpu-translate %s --mlir-to-msl | FileCheck %s

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

// CHECK: kernel void reduce(
// CHECK: device float* buf0
// CHECK: device float* buf1
// CHECK: float [[IN:v[0-9]+]] = buf0[
// CHECK: simd_sum([[IN]])