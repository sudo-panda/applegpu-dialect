#ifndef APPLE_TT_H
#define APPLE_TT_H

#include "mlir/IR/Dialect.h"
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

#endif // APPLE_TT_H
