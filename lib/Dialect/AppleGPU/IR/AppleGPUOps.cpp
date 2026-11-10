//===- AppleGPUOps.cpp - AppleGPU dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/IR/GPUDialect.h"

#include "applegpu/Dialect/AppleGPU/IR/AppleGPUOps.h"
#include "applegpu/Dialect/AppleGPU/IR/AppleGPUDialect.h"

#define GET_OP_CLASSES
#include "applegpu/Dialect/AppleGPU/IR/AppleGPUOps.cpp.inc"

mlir::LogicalResult mlir::applegpu::SimdgroupSumOp::verify() {
  if (!(*this)->getParentOfType<mlir::gpu::GPUFuncOp>())
    return emitOpError("must be inside a gpu.func kernel");
  return mlir::success();
}