//===- applegpu-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/ArithDialect.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "applegpu/Dialect/AppleGPU/IR/AppleGPUDialect.h"

int main(int argc, char **argv) {
  // No passes are registered yet: the current tests only parse, print and
  // verify. TODO: Register applegpu passes here.

  mlir::DialectRegistry registry;
  registry.insert<mlir::applegpu::AppleGPUDialect,
                  mlir::arith::ArithDialect,
                  mlir::func::FuncDialect,
                  mlir::gpu::GPUDialect,
                  mlir::memref::MemRefDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "AppleGPU optimizer driver\n", registry));
}
