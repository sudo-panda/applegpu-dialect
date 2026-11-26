#ifndef APPLE_TT_H
#define APPLE_TT_H

#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LinearLayout.h"

// The generated declarations name these unqualified.
namespace mlir::apple_tt {
using ::mlir::triton::LinearLayout;
using ::mlir::triton::gpu::CGAEncodingAttr;
} // namespace mlir::apple_tt

#include "AppleTTDialect.h.inc"
#define GET_ATTRDEF_CLASSES
#include "AppleTTAttrs.h.inc"

namespace mlir::apple_tt {
#define GEN_PASS_DECL
#include "AppleTTPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "AppleTTPasses.h.inc"
} // namespace mlir::apple_tt

#endif // APPLE_TT_H
