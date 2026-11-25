// The apple_tt dialect: Apple GPU layouts that Triton loads as a plugin.
// Built against Triton's headers and Triton's LLVM.

#ifndef APPLE_TT_TD
#define APPLE_TT_TD

include "mlir/IR/AttrTypeBase.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrInterfaces.td"

def AppleTT_Dialect : Dialect {
  let name = "apple_tt";
  let cppNamespace = "::mlir::apple_tt";
  let summary = "Apple GPU layouts for Triton";
  let useDefaultAttributePrinterParser = 1;
}

def AppleTT_SimdgroupMmaEncodingAttr
    : AttrDef<AppleTT_Dialect, "SimdgroupMmaEncoding", [
        DeclareAttrInterfaceMethods<LayoutEncodingTrait, ["getCGALayout"]>,
        DeclareAttrInterfaceMethods<DistributedEncodingTrait,
                                    ["getRepOrder", "toLinearLayout"]>,
        DeclareAttrInterfaceMethods<MmaEncodingTrait,
                                    ["getRepOrderForOperand",
                                     "dotOperandToLinearLayout"]>]> {
  let mnemonic = "simdgroup_mma";
  let summary = "Result layout of a dot computed with simdgroup_matrix<T, 8, 8>";
  let description = [{
    The tensor is tiled into 8x8 simdgroup_matrix fragments. Within a
    fragment, each lane holds 2 adjacent elements of one row: register bit 0
    is column bit 0, and lane bits 0-4 are column bit 1, row bits 0 and 1,
    column bit 2 and row bit 2 (measured on the GPU; the same mapping as
    MLX's get_coord). SIMD-groups are arranged as warpsPerCTA, row-major, and
    the pattern repeats over the tensor in registers.

    Dot operands with this parent use the same fragment mapping, with rows
    and columns being (M, K) for operand 0 and (K, N) for operand 1. Each
    operand is replicated across the SIMD-groups that share its rows (A) or
    columns (B).
  }];
  let parameters = (ins ArrayRefParameter<"unsigned">:$warpsPerCTA);
  let assemblyFormat = "`<` `{` `warpsPerCTA` `=` `[` $warpsPerCTA `]` `}` `>`";
  let genVerifyDecl = 1;
}

#endif // APPLE_TT_TD
