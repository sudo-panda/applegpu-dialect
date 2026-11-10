//===- AppleGPUDialect.td - AppleGPU dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef APPLEGPU_DIALECT
#define APPLEGPU_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// AppleGPU dialect definition.
//===----------------------------------------------------------------------===//

def AppleGPU_Dialect : Dialect {
    let name = "applegpu";
    let summary = "A applegpu out-of-tree MLIR dialect.";
    let description = [{
        This dialect is a dialect to support Metal GPU on Apple devices.
    }];
    let cppNamespace = "::mlir::applegpu";
}

//===----------------------------------------------------------------------===//
// Base applegpu operation definition.
//===----------------------------------------------------------------------===//

class AppleGPU_Op<string mnemonic, list<Trait> traits = []> :
        Op<AppleGPU_Dialect, mnemonic, traits>;

#endif // APPLEGPU_DIALECT
