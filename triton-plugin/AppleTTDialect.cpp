#include "AppleTT.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
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

// Triton asks the dialect of a tensor's layout how layouts change through
// ops (reduce, expand_dims, trans, ...). TritonGPU's answers work from the
// LinearLayout of any distributed layout, so every question is forwarded.
struct InferLayout : public triton::DialectInferLayoutInterface {
  using DialectInferLayoutInterface::DialectInferLayoutInterface;

  const triton::DialectInferLayoutInterface &ttg() const {
    auto *dialect =
        getDialect()->getContext()->getLoadedDialect<TritonGPUDialect>();
    return *dialect
                ->getRegisteredInterface<triton::DialectInferLayoutInterface>();
  }

  LogicalResult
  inferTransOpEncoding(Attribute operandEncoding, ArrayRef<int64_t> shape,
                       ArrayRef<int32_t> order, Attribute &resultEncoding,
                       std::optional<Location> loc) const override {
    return ttg().inferTransOpEncoding(operandEncoding, shape, order,
                                      resultEncoding, loc);
  }
  LogicalResult
  inferReduceOpEncoding(Attribute operandEncoding, unsigned axis,
                        Attribute &resultEncoding,
                        std::optional<Location> loc) const override {
    return ttg().inferReduceOpEncoding(operandEncoding, axis, resultEncoding,
                                       loc);
  }
  LogicalResult
  inferExpandDimsOpEncoding(Attribute operandEncoding, unsigned axis,
                            Attribute &resultEncoding,
                            std::optional<Location> loc) const override {
    return ttg().inferExpandDimsOpEncoding(operandEncoding, axis,
                                           resultEncoding, loc);
  }
  LogicalResult
  verifyBroadcastOpEncoding(RankedTensorType srcType,
                            RankedTensorType dstType) const override {
    return ttg().verifyBroadcastOpEncoding(srcType, dstType);
  }
  LogicalResult inferDotOpEncoding(Attribute operandEncoding, unsigned opIdx,
                                   Attribute retEncoding,
                                   std::optional<Location> loc) const override {
    return ttg().inferDotOpEncoding(operandEncoding, opIdx, retEncoding, loc);
  }
  LogicalResult
  inferReshapeOpEncoding(ArrayRef<int64_t> srcShape, Attribute srcEnc,
                         ArrayRef<int64_t> dstShape, Attribute &dstEnc,
                         bool allowReorder,
                         std::optional<Location> loc) const override {
    return ttg().inferReshapeOpEncoding(srcShape, srcEnc, dstShape, dstEnc,
                                        allowReorder, loc);
  }
  LogicalResult verifyLayoutsAreEqual(ArrayRef<int64_t> shape,
                                      Attribute expected, Attribute got,
                                      std::optional<Location> loc,
                                      bool ignoreRegBroadcast) const override {
    return ttg().verifyLayoutsAreEqual(shape, expected, got, loc,
                                       ignoreRegBroadcast);
  }
  LogicalResult
  inferDefaultJoinOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                             ArrayRef<int64_t> shape,
                             std::optional<Location> loc) const override {
    return ttg().inferDefaultJoinOpEncoding(srcEnc, dstEnc, shape, loc);
  }
  LogicalResult
  inferSplitOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                       ArrayRef<int64_t> shape,
                       std::optional<Location> loc) const override {
    return ttg().inferSplitOpEncoding(srcEnc, dstEnc, shape, loc);
  }
  LogicalResult
  verifyDotOpEncodingCompatibility(Operation *op, Attribute operandEncodingA,
                                   Attribute operandEncodingB) const override {
    return ttg().verifyDotOpEncodingCompatibility(op, operandEncodingA,
                                                  operandEncodingB);
  }
  LogicalResult verifyDotScaledOpEncodingCompatibility(
      Operation *op, Attribute operandEncodingA, Attribute operandEncodingB,
      Attribute scaleEncodingA, Attribute scaleEncodingB) const override {
    return ttg().verifyDotScaledOpEncodingCompatibility(
        op, operandEncodingA, operandEncodingB, scaleEncodingA, scaleEncodingB);
  }
  LogicalResult verifyCatOpEncodingCompatibility(Operation *op) const override {
    return ttg().verifyCatOpEncodingCompatibility(op);
  }
  LogicalResult
  inferFp4ToFpOpEncoding(ArrayRef<int64_t> shape, int axis, Attribute inEnc,
                         Attribute &outEnc, bool fwdInference,
                         std::optional<Location> loc) const override {
    return ttg().inferFp4ToFpOpEncoding(shape, axis, inEnc, outEnc,
                                        fwdInference, loc);
  }
};

// Triton also asks the layout's dialect to check each tensor layout against
// its module: rank, threads per warp, ttg.num-warps and CTAs. TritonGPU's
// checks work from the LinearLayout too, so both questions are forwarded.
struct VerifyLayout : public triton::DialectVerifyTensorLayoutInterface {
  using DialectVerifyTensorLayoutInterface::DialectVerifyTensorLayoutInterface;

  const triton::DialectVerifyTensorLayoutInterface &ttg() const {
    auto *dialect =
        getDialect()->getContext()->getLoadedDialect<TritonGPUDialect>();
    return *dialect->getRegisteredInterface<
        triton::DialectVerifyTensorLayoutInterface>();
  }

  LogicalResult verifyTensorLayout(
      Attribute layout, RankedTensorType type, Operation *op,
      function_ref<InFlightDiagnostic()> emitError) const override {
    return ttg().verifyTensorLayout(layout, type, op, emitError);
  }
  LogicalResult verifyMemDescLayout(
      Attribute layout, Type type, Operation *op,
      function_ref<InFlightDiagnostic()> emitError) const override {
    return ttg().verifyMemDescLayout(layout, type, op, emitError);
  }
};

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
  addInterfaces<InferLayout, VerifyLayout>();
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

SmallVector<unsigned>
SimdgroupMmaEncodingAttr::getRepOrderForOperand(int opIdx) const {
  return getOrderForDotOperand(opIdx, getWarpsPerCTA().size(),
                               /*kContig=*/true);
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

LinearLayout SimdgroupMmaEncodingAttr::dotOperandToLinearLayout(
    Attribute dotOp, ArrayRef<int64_t> shape) const {
  auto dotEnc = cast<DotOperandEncodingAttr>(dotOp);
  MLIRContext *ctx = getContext();
  auto dims = standardOutDimNames(ctx, 2);
  // K first for both operands, as getRepOrderForOperand says.
  auto repDims = repOrderedDims(ctx, getRepOrderForOperand(dotEnc.getOpIdx()));
  StringAttr kWarp = StringAttr::get(ctx, "warp");
  unsigned warpsM = getWarpsPerCTA()[0], warpsN = getWarpsPerCTA()[1];
  LinearLayout tile = fragment(ctx, dims[0], dims[1]).transposeOuts(repDims);
  // Same warp bit order as the result: N bits first, then M bits. A (M x K)
  // is the same for every SIMD-group in a row, B (K x N) in a column.
  LinearLayout warps =
      dotEnc.getOpIdx() == 0
          ? LinearLayout::zeros1D(warpsN, kWarp, dims[1]) *
                LinearLayout::identity1D(warpsM, kWarp, dims[0])
          : LinearLayout::identity1D(warpsN, kWarp, dims[1]) *
                LinearLayout::zeros1D(warpsM, kWarp, dims[0]);
  return combineCtaCgaWithShape(tile * warps.transposeOuts(repDims),
                                dotEnc.getCGALayout(), shape);
}

} // namespace mlir::apple_tt
