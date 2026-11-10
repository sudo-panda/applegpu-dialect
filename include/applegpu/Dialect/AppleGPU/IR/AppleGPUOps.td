//===- AppleGPUOps.td - AppleGPU dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef APPLEGPU_OPS
#define APPLEGPU_OPS

include "applegpu/Dialect/AppleGPU/IR/AppleGPUDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def AppleGPU_SimdgroupSumOp : AppleGPU_Op<"simdgroup_sum",
    [Pure, SameOperandsAndResultType]> {
  let summary = "Sums a value across all threads in a SIMD-group";
  let description = [{
    Every thread in the SIMD-group provides a value, and every thread
    receives the total. Corresponds to `simd_sum` in Metal Shading Language.
  }];

  let arguments = (ins AnyTypeOf<[F16, F32, I32]>:$value);
  let results = (outs AnyTypeOf<[F16, F32, I32]>:$result);

  let assemblyFormat = "$value attr-dict `:` type($value)";
  
  let hasVerifier = 1;
}

#endif // APPLEGPU_OPS
