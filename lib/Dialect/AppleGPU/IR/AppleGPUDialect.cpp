//===- AppleGPUDialect.cpp - AppleGPU dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "applegpu/Dialect/AppleGPU/IR/AppleGPUDialect.h"
#include "applegpu/Dialect/AppleGPU/IR/AppleGPUOps.h"

using namespace mlir;
using namespace mlir::applegpu;

#include "applegpu/Dialect/AppleGPU/IR/AppleGPUOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// AppleGPU dialect.
//===----------------------------------------------------------------------===//

void AppleGPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "applegpu/Dialect/AppleGPU/IR/AppleGPUOps.cpp.inc"
      >();
}
