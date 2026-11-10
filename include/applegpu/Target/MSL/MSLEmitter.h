//===- MSLEmitter.h - Metal Shading Language emitter ---------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef APPLEGPU_TARGET_MSL_MSLEMITTER_H
#define APPLEGPU_TARGET_MSL_MSLEMITTER_H

#include "mlir/Support/LLVM.h"

namespace mlir {
class Operation;

namespace applegpu {

/// Translates the gpu.func kernels nested in `op` to Metal Shading Language
/// and writes the result to `os`.
LogicalResult translateToMSL(Operation *op, raw_ostream &os);

/// Registers the `mlir-to-msl` translation.
void registerToMSLTranslation();

} // namespace applegpu
} // namespace mlir

#endif // APPLEGPU_TARGET_MSL_MSLEMITTER_H
