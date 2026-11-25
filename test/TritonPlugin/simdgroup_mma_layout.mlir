// Which lane holds which element of #apple_tt.simdgroup_mma and its dot
// operands, printed by Triton's triton-tensor-layout ("T<lane>:<register>",
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

// Operand A (M x K) is shared by the SIMD-groups of one row (0 and 1),
// operand B (K x N) by those of one column (0 and 2).
// RUN: env TRITON_PLUGIN_PATHS=%apple_triton_plugin %triton_tensor_layout \
// RUN:   -l "#ttg.dot_op<{opIdx = 0, parent = #apple_tt.simdgroup_mma<{warpsPerCTA = [2, 2]}>}>" \
// RUN:   -t "tensor<16x8xf16>" | FileCheck %s --check-prefix=OPA
// OPA: {{\[\[}} T0:0| T32:0,   T0:1| T32:1,   T1:0| T33:0,
// RUN: env TRITON_PLUGIN_PATHS=%apple_triton_plugin %triton_tensor_layout \
// RUN:   -l "#ttg.dot_op<{opIdx = 1, parent = #apple_tt.simdgroup_mma<{warpsPerCTA = [2, 2]}>}>" \
// RUN:   -t "tensor<8x16xf16>" | FileCheck %s --check-prefix=OPB
// OPB: {{\[\[}} T0:0| T64:0,   T0:1| T64:1,   T1:0| T65:0,

// A tensor bigger than the pattern repeats it in registers, in the order
// getRepOrder() and getRepOrderForOperand() give. For the result, columns
// first: thread 0's register 2 is column 16, register 4 is row 16.
// RUN: env TRITON_PLUGIN_PATHS=%apple_triton_plugin %triton_tensor_layout \
// RUN:   -l "#apple_tt.simdgroup_mma<{warpsPerCTA = [2, 2]}>" -t "tensor<32x32xf32>" \
// RUN:   | FileCheck %s --check-prefix=REPEAT
// REPEAT: {{\[\[}} T0:0, {{.*}} T41:1, T0:2, T0:3, {{.*}} T41:3]
// REPEAT: [ T0:4, T0:5, {{.*}} T41:6, T41:7]

// Operands repeat along K first: K is the column for A, the row for B. A's
// pattern is 16 x 8 here, so it repeats along both; register 2 is the next
// 8 columns (K), and only register 8 moves down to row 16.
// RUN: env TRITON_PLUGIN_PATHS=%apple_triton_plugin %triton_tensor_layout \
// RUN:   -l "#ttg.dot_op<{opIdx = 0, parent = #apple_tt.simdgroup_mma<{warpsPerCTA = [2, 2]}>}>" \
// RUN:   -t "tensor<32x32xf16>" | FileCheck %s --check-prefix=OPA_K
// OPA_K: {{\[\[}} T0:0| T32:0, {{.*}} T9:1| T41:1, T0:2| T32:2, {{.*}} T9:3| T41:3, T0:4| T32:4,
// OPA_K: [ T0:8| T32:8, T0:9| T32:9,
// RUN: env TRITON_PLUGIN_PATHS=%apple_triton_plugin %triton_tensor_layout \
// RUN:   -l "#ttg.dot_op<{opIdx = 1, parent = #apple_tt.simdgroup_mma<{warpsPerCTA = [2, 2]}>}>" \
// RUN:   -t "tensor<32x16xf16>" | FileCheck %s --check-prefix=OPB_K
// OPB_K: [ T0:2| T64:2, T0:3| T64:3,
