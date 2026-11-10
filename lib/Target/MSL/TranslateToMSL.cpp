//===- TranslateToMSL.cpp - Metal Shading Language emitter ---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "applegpu/Target/MSL/MSLEmitter.h"
#include "applegpu/Dialect/AppleGPU/IR/AppleGPUDialect.h"
#include "applegpu/Dialect/AppleGPU/IR/AppleGPUOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace {

// Walks gpu.func kernels and prints them as Metal Shading Language.
class MSLEmitter {
public:
  explicit MSLEmitter(raw_ostream &os) : os(os) {}

  LogicalResult emitModule(Operation *top) {
    os << "#include <metal_stdlib>\nusing namespace metal;\n\n";
    LogicalResult result = success();
    top->walk([&](gpu::GPUFuncOp func) {
      if (func.isKernel() && failed(emitKernel(func)))
        result = failure();
    });
    return result;
  }

private:
  raw_ostream &os;
  llvm::DenseMap<Value, std::string> names; // MLIR value -> MSL variable name
  unsigned counter = 0;

  // Gives an MLIR value a fresh MSL variable name: v0, v1, v2, ...
  std::string newName(Value v) {
    std::string name = "v" + std::to_string(counter++);
    names[v] = name;
    return name;
  }

  // Maps an MLIR type to the matching MSL type.
  FailureOr<std::string> mslType(Type t) {
    if (t.isF32())       return std::string("float");
    if (t.isF16())       return std::string("half");
    if (t.isInteger(32)) return std::string("int");
    if (t.isIndex())     return std::string("uint");
    return failure();
  }

  LogicalResult emitKernel(gpu::GPUFuncOp func) {
    names.clear();
    counter = 0;

    // Signature: one device buffer per memref argument.
    os << "kernel void " << func.getName() << "(\n";
    unsigned i = 0;
    for (BlockArgument arg : func.getArguments()) {
      auto memref = dyn_cast<MemRefType>(arg.getType());
      if (!memref || memref.getRank() != 1)
        return func.emitError("only 1-D memref arguments are supported");
      FailureOr<std::string> elem = mslType(memref.getElementType());
      if (failed(elem))
        return func.emitError("unsupported memref element type");
      std::string name = "buf" + std::to_string(i);
      names[arg] = name;
      os << "    device " << *elem << "* " << name
         << " [[buffer(" << i << ")]],\n";
      ++i;
    }
    os << "    uint3 tid [[thread_position_in_threadgroup]]) {\n";

    // Body: one MSL line per operation.
    for (Operation &op : func.getBody().front())
      if (failed(emitOp(op)))
        return failure();

    os << "}\n\n";
    return success();
  }

  LogicalResult emitOp(Operation &op) {
    return llvm::TypeSwitch<Operation *, LogicalResult>(&op)
        .Case<gpu::ThreadIdOp>([&](gpu::ThreadIdOp tidOp) -> LogicalResult {
          std::string n = newName(tidOp.getResult());
          os << "  uint " << n << " = tid."
             << gpu::stringifyDimension(tidOp.getDimension()) << ";\n";
          return success();
        })
        .Case<memref::LoadOp>([&](memref::LoadOp load) -> LogicalResult {
          if (load.getIndices().size() != 1)
            return load.emitError("only 1-D loads are supported");
          FailureOr<std::string> t = mslType(load.getResult().getType());
          if (failed(t))
            return load.emitError("unsupported load type");
          std::string n = newName(load.getResult());
          os << "  " << *t << " " << n << " = " << names[load.getMemRef()]
             << "[" << names[load.getIndices()[0]] << "];\n";
          return success();
        })
        .Case<applegpu::SimdgroupSumOp>(
            [&](applegpu::SimdgroupSumOp sum) -> LogicalResult {
              FailureOr<std::string> t = mslType(sum.getResult().getType());
              if (failed(t))
                return sum.emitError("unsupported simdgroup_sum type");
              std::string n = newName(sum.getResult());
              os << "  " << *t << " " << n << " = simd_sum("
                 << names[sum.getValue()] << ");\n";
              return success();
            })
        .Case<memref::StoreOp>([&](memref::StoreOp store) -> LogicalResult {
          if (store.getIndices().size() != 1)
            return store.emitError("only 1-D stores are supported");
          os << "  " << names[store.getMemRef()] << "["
             << names[store.getIndices()[0]] << "] = "
             << names[store.getValueToStore()] << ";\n";
          return success();
        })
        .Case<gpu::ReturnOp>([&](gpu::ReturnOp) -> LogicalResult {
          os << "  return;\n";
          return success();
        })
        .Default([&](Operation *other) -> LogicalResult {
          return other->emitError("MSL emitter does not support this operation yet");
        });
  }
};

} // namespace

LogicalResult mlir::applegpu::translateToMSL(Operation *op, raw_ostream &os) {
  return MSLEmitter(os).emitModule(op);
}

void mlir::applegpu::registerToMSLTranslation() {
  TranslateFromMLIRRegistration mslTranslation(
      "mlir-to-msl", "Translate gpu kernels to Metal Shading Language",
      [](Operation *op, raw_ostream &os) { return translateToMSL(op, os); },
      [](DialectRegistry &registry) {
        registry.insert<applegpu::AppleGPUDialect, arith::ArithDialect,
                        func::FuncDialect, gpu::GPUDialect,
                        memref::MemRefDialect>();
      });
}
