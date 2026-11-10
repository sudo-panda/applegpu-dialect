//===- AppleGPUOps.h - AppleGPU dialect ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef APPLEGPU_APPLEGPUOPS_H
#define APPLEGPU_APPLEGPUOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#define GET_OP_CLASSES
#include "applegpu/Dialect/AppleGPU/IR/AppleGPUOps.h.inc"

#endif // APPLEGPU_APPLEGPUOPS_H
