#include "AppleTT.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

#include "AppleTTDialect.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "AppleTTAttrs.cpp.inc"

namespace mlir::apple_tt {
namespace {

// One 8x8 simdgroup_matrix fragment, rows on `rowDim` and columns on
// `colDim`: 32 lanes x 2 registers. See the attribute's description.
LinearLayout fragment(MLIRContext *ctx, StringAttr rowDim, StringAttr colDim) {
  StringAttr kRegister = StringAttr::get(ctx, "register");
  StringAttr kLane = StringAttr::get(ctx, "lane");
  // Each basis is (row, col).
  return LinearLayout({{kRegister, {{0, 1}}},
                       {kLane, {{0, 2}, {1, 0}, {2, 0}, {0, 4}, {4, 0}}}},
                      {rowDim, colDim});
}

// The tensor's output dimensions listed in a rep order (fastest first).
// combineCtaCgaWithShape adds the register bits that repeat the pattern over
// a larger tensor in the order of the layout's outputs, so listing them in
// the rep order makes registers follow getRepOrder(), as for NVIDIA's layouts.
SmallVector<StringAttr> repOrderedDims(MLIRContext *ctx,
                                       ArrayRef<unsigned> repOrder) {
  auto dims = standardOutDimNames(ctx, repOrder.size());
  SmallVector<StringAttr> ordered;
  for (unsigned d : repOrder)
    ordered.push_back(dims[d]);
  return ordered;
}

} // namespace

void AppleTTDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "AppleTTAttrs.cpp.inc"
      >();
}

LogicalResult
SimdgroupMmaEncodingAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<unsigned> warpsPerCTA) {
  if (warpsPerCTA.size() != 2)
    return emitError()
           << "simdgroup_mma supports rank 2 only, got warpsPerCTA of size "
           << warpsPerCTA.size();
  for (unsigned w : warpsPerCTA)
    if (!llvm::isPowerOf2_32(w))
      return emitError() << "warpsPerCTA must be powers of 2";
  return success();
}

CGAEncodingAttr SimdgroupMmaEncodingAttr::getCGALayout() const {
  return CGAEncodingAttr::get1CTALayout(getContext(), getWarpsPerCTA().size());
}

SmallVector<unsigned> SimdgroupMmaEncodingAttr::getRepOrder() const {
  return getMatrixOrder(getWarpsPerCTA().size(), /*rowMajor=*/true);
}

LinearLayout
SimdgroupMmaEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  MLIRContext *ctx = getContext();
  auto dims = standardOutDimNames(ctx, 2);
  auto repDims = repOrderedDims(ctx, getRepOrder());
  StringAttr kWarp = StringAttr::get(ctx, "warp");
  LinearLayout tile = fragment(ctx, dims[0], dims[1]).transposeOuts(repDims);
  // Row-major SIMD-groups: the low warp bits walk along N.
  LinearLayout warps =
      identityStandardND(kWarp, getWarpsPerCTA(), /*order=*/{1, 0})
          .transposeOuts(repDims);
  return combineCtaCgaWithShape(tile * warps, getCGALayout(), shape);
}

} // namespace mlir::apple_tt
