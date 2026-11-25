// Which lane holds which element of #apple_tt.simdgroup_mma, printed by Triton's triton-tensor-layout ("T<lane>:<register>",
// "|" between SIMD-groups that hold the same element). The 8x8 fragment is
// the mapping measured on the GPU: each lane holds 2 adjacent elements of
// one row.

// RUN: env TRITON_PLUGIN_PATHS=%apple_triton_plugin %triton_tensor_layout \
// RUN:   -l "#apple_tt.simdgroup_mma<{warpsPerCTA = [1, 1]}>" -t "tensor<8x8xf32>" \
// RUN:   | FileCheck %s --check-prefix=FRAG
// FRAG:      {{\[\[}} T0:0,  T0:1,  T1:0,  T1:1,  T8:0,  T8:1,  T9:0,  T9:1]
// FRAG-NEXT: [  T2:0,  T2:1,  T3:0,  T3:1, T10:0, T10:1, T11:0, T11:1]
// FRAG-NEXT: [  T4:0,  T4:1,  T5:0,  T5:1, T12:0, T12:1, T13:0, T13:1]
// FRAG-NEXT: [  T6:0,  T6:1,  T7:0,  T7:1, T14:0, T14:1, T15:0, T15:1]
// FRAG-NEXT: [ T16:0, T16:1, T17:0, T17:1, T24:0, T24:1, T25:0, T25:1]
// FRAG-NEXT: [ T18:0, T18:1, T19:0, T19:1, T26:0, T26:1, T27:0, T27:1]
// FRAG-NEXT: [ T20:0, T20:1, T21:0, T21:1, T28:0, T28:1, T29:0, T29:1]
// FRAG-NEXT: [ T22:0, T22:1, T23:0, T23:1, T30:0, T30:1, T31:0, T31:1]]

// 2x2 SIMD-groups, row-major: SIMD-group 1 (threads 32-63) holds columns
// 8-15, SIMD-group 2 (threads 64-95) rows 8-15.
// RUN: env TRITON_PLUGIN_PATHS=%apple_triton_plugin %triton_tensor_layout \
// RUN:   -l "#apple_tt.simdgroup_mma<{warpsPerCTA = [2, 2]}>" -t "tensor<16x16xf32>" \
// RUN:   | FileCheck %s --check-prefix=WARPS
// WARPS: {{\[\[}} T0:0, {{.*}} T9:1, T32:0, {{.*}} T41:1]
// WARPS: [ T64:0, {{.*}} T73:1, T96:0, {{.*}} T105:1]

// A tensor bigger than the pattern repeats it in registers, in the order
// getRepOrder() gives: columns first. Thread 0's register 2 is column 16,
// register 4 is row 16.
// RUN: env TRITON_PLUGIN_PATHS=%apple_triton_plugin %triton_tensor_layout \
// RUN:   -l "#apple_tt.simdgroup_mma<{warpsPerCTA = [2, 2]}>" -t "tensor<32x32xf32>" \
// RUN:   | FileCheck %s --check-prefix=REPEAT
// REPEAT: {{\[\[}} T0:0, {{.*}} T41:1, T0:2, T0:3, {{.*}} T41:3]
// REPEAT: [ T0:4, T0:5, {{.*}} T41:6, T41:7]
