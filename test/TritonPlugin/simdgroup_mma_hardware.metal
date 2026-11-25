// Checks #apple_tt.simdgroup_mma against the GPU, for float, half and bfloat.
// The kernel below runs on one SIMD-group and writes three buffers; the
// checker compares them with the layout the plugin computes, read with
// triton-tensor-layout, so the expected positions are never typed by hand:
//
// 1. loaded: a matrix holding 0..63 is loaded with simdgroup_load, and each
//    lane writes its two thread_elements(). This is where the hardware puts
//    each element; it must match the plugin's result layout and both of its
//    operand layouts.
// 2. positions: where this kernel believes each lane's elements are, from the
//    same bit mapping as fragment() in AppleTTDialect.cpp. It must also match
//    the plugin, so that part 3 tests the plugin's mapping.
// 3. product: each lane writes A and B straight into its fragments at those
//    positions, with no simdgroup_load, as the lowering will do for values
//    computed in registers (attention's P), then multiplies. The stored
//    product must equal A x B computed on the CPU, which holds only if the
//    multiply reads A and B where the operand layouts say they are.
//
// REQUIRES: metal
// RUN: xcrun -sdk macosx metal -std=metal3.1 -c %s -o %t.air
// RUN: xcrun -sdk macosx metallib %t.air -o %t.metallib
// RUN: %run_grid %t.metallib fragments_float 1 32 f32:64:fill=-1 f32:64:fill=-1 f32:64:fill=-1 > %t.float.out
// RUN: %run_grid %t.metallib fragments_half 1 32 f32:64:fill=-1 f32:64:fill=-1 f32:64:fill=-1 > %t.half.out
// RUN: %run_grid %t.metallib fragments_bfloat 1 32 f32:64:fill=-1 f32:64:fill=-1 f32:64:fill=-1 > %t.bfloat.out
// RUN: %triton_python %S/../runners/check_simdgroup_fragments.py \
// RUN:   --tensor-layout %triton_tensor_layout --plugin %apple_triton_plugin \
// RUN:   float=%t.float.out half=%t.half.out bfloat=%t.bfloat.out | FileCheck %s
// CHECK: float: loaded 64/64, positions 64/64, operand A 64/64, operand B 64/64, product 64/64
// CHECK: half: loaded 64/64, positions 64/64, operand A 64/64, operand B 64/64, product 64/64
// CHECK: bfloat: loaded 64/64, positions 64/64, operand A 64/64, operand B 64/64, product 64/64

#include <metal_stdlib>
using namespace metal;

// (row, col) of a lane's register r in an 8x8 fragment, as fragment() in
// AppleTTDialect.cpp defines it: register bit 0 -> col + 1; lane bits 0-4 ->
// col + 2, row + 1, row + 2, col + 4, row + 4.
static uint2 layoutPosition(uint lane, uint r) {
  uint row = ((lane >> 1) & 1) + (((lane >> 2) & 1) << 1) + (((lane >> 4) & 1) << 2);
  uint col = r + ((lane & 1) << 1) + (((lane >> 3) & 1) << 2);
  return uint2(row, col);
}

// Small integers, so every product and sum is exact in half and bfloat. The
// checker computes the same values.
static float aValue(uint m, uint k) { return float(int((m + 2 * k) % 5) - 2); }
static float bValue(uint k, uint n) { return float(int((3 * k + n) % 7) - 3); }

template <typename T>
static void fragments(device float *loaded, device float *positions,
                      device float *product, threadgroup T *tile, uint lane) {
  // 1. The hardware's placement.
  for (uint i = lane; i < 64; i += 32)
    tile[i] = T(float(i));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_matrix<T, 8, 8> m;
  simdgroup_load(m, tile, 8);
  for (uint r = 0; r < 2; ++r)
    loaded[lane * 2 + r] = float(m.thread_elements()[r]);

  // 2. The positions this kernel assumes.
  for (uint r = 0; r < 2; ++r) {
    uint2 p = layoutPosition(lane, r);
    positions[lane * 2 + r] = float(p.x * 8 + p.y);
  }

  // 3. Operands placed by those positions, multiplied.
  simdgroup_matrix<T, 8, 8> a, b;
  simdgroup_matrix<T, 8, 8> c = make_filled_simdgroup_matrix<T, 8, 8>(T(0.0f));
  for (uint r = 0; r < 2; ++r) {
    uint2 p = layoutPosition(lane, r);
    a.thread_elements()[r] = T(aValue(p.x, p.y)); // A: (m, k)
    b.thread_elements()[r] = T(bValue(p.x, p.y)); // B: (k, n)
  }
  simdgroup_multiply_accumulate(c, a, b, c);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_store(c, tile, 8);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = lane; i < 64; i += 32)
    product[i] = float(tile[i]);
}

kernel void fragments_float(device float *loaded [[buffer(0)]],
                            device float *positions [[buffer(1)]],
                            device float *product [[buffer(2)]],
                            uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tile[64];
  fragments<float>(loaded, positions, product, tile, lane);
}

kernel void fragments_half(device float *loaded [[buffer(0)]],
                           device float *positions [[buffer(1)]],
                           device float *product [[buffer(2)]],
                           uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half tile[64];
  fragments<half>(loaded, positions, product, tile, lane);
}

kernel void fragments_bfloat(device float *loaded [[buffer(0)]],
                             device float *positions [[buffer(1)]],
                             device float *product [[buffer(2)]],
                             uint lane [[thread_index_in_simdgroup]]) {
  threadgroup bfloat tile[64];
  fragments<bfloat>(loaded, positions, product, tile, lane);
}
