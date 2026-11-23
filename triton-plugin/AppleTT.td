// The apple_tt dialect: Apple GPU layouts that Triton loads as a plugin.
// Built against Triton's headers and Triton's LLVM.

#ifndef APPLE_TT_TD
#define APPLE_TT_TD

include "mlir/IR/AttrTypeBase.td"

def AppleTT_Dialect : Dialect {
  let name = "apple_tt";
  let cppNamespace = "::mlir::apple_tt";
  let summary = "Apple GPU layouts for Triton";
  let useDefaultAttributePrinterParser = 1;
}

def AppleTT_SimdgroupMmaEncodingAttr
    : AttrDef<AppleTT_Dialect, "SimdgroupMmaEncoding"> {
  let mnemonic = "simdgroup_mma";
  let summary = "Result layout of a dot computed with simdgroup_matrix<T, 8, 8>";
  let parameters = (ins ArrayRefParameter<"unsigned">:$warpsPerCTA);
  let assemblyFormat = "`<` `{` `warpsPerCTA` `=` `[` $warpsPerCTA `]` `}` `>`";
  let genVerifyDecl = 1;
}

#endif // APPLE_TT_TD
