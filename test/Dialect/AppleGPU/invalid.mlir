// RUN: applegpu-opt %s -verify-diagnostics

func.func @outside_kernel(%x: f32) -> f32 {
  // expected-error @+1 {{must be inside a gpu.func kernel}}
  %0 = applegpu.simdgroup_sum %x : f32
  return %0 : f32
}