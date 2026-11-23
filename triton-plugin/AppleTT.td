// The apple_tt dialect: Apple GPU layouts that Triton loads as a plugin.
// Built against Triton's headers and Triton's LLVM.

#ifndef APPLE_TT_TD
#define APPLE_TT_TD

include "mlir/IR/DialectBase.td"

def AppleTT_Dialect : Dialect {
  let name = "apple_tt";
  let cppNamespace = "::mlir::apple_tt";
  let summary = "Apple GPU layouts for Triton";
}

#endif // APPLE_TT_TD
