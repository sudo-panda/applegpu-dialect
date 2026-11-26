#include "AppleTT.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Builders.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace mlir::apple_tt {
#define GEN_PASS_DEF_APPLEACCELERATEMATMUL
#include "AppleTTPasses.h.inc"

namespace {

constexpr int64_t kTile = 8;

bool isSupportedInput(Type t) { return t.isF16() || t.isBF16() || t.isF32(); }

bool canUseSimdgroupMatrix(DotOp dot) {
  auto retType = dot.getType();
  auto aType = dot.getA().getType();
  auto bType = dot.getB().getType();
  if (retType.getRank() != 2 ||
      !isa<BlockedEncodingAttr>(retType.getEncoding()))
    return false;
  Type aElt = aType.getElementType(), bElt = bType.getElementType();
  if (aElt != bElt || !isSupportedInput(aElt) ||
      !retType.getElementType().isF32())
    return false;
  int64_t m = retType.getShape()[0], n = retType.getShape()[1];
  int64_t k = aType.getShape()[1];
  return m % kTile == 0 && n % kTile == 0 && k % kTile == 0;
}

// NVIDIA's getMmaV2WarpsPerCTA with 8x8 tiles: double the SIMD-groups along
// the dimension with more tiles left per SIMD-group, ties toward M.
SmallVector<unsigned> balancedWarps(ArrayRef<int64_t> shape, int numWarps) {
  SmallVector<int64_t> reps = {(shape[0] + kTile - 1) / kTile,
                               (shape[1] + kTile - 1) / kTile};
  SmallVector<unsigned> warps = {1, 1};
  while (warps[0] * warps[1] < static_cast<unsigned>(numWarps)) {
    if (reps[0] >= reps[1]) {
      warps[0] *= 2;
      if (reps[0] != 1)
        reps[0] /= 2;
    } else {
      warps[1] *= 2;
      reps[1] /= 2;
    }
  }
  return warps;
}

// NVIDIA's warpsPerTileV2: a dot that feeds or is fed by another dot (as in
// attention) puts all SIMD-groups on one axis, so that row reductions
// between the two dots stay inside one SIMD-group, and reuses the placement
// of a chained dot that already has one.
SmallVector<unsigned> warpsPerTile(DotOp dot, ArrayRef<int64_t> shape,
                                   int numWarps) {
  auto filter = [&dot](Operation *op) {
    return op->getParentRegion() == dot->getParentRegion() && !isa<TransOp>(op);
  };
  bool hasChainedDot = false;
  for (Operation *op : mlir::getSlice(dot, {filter}, {filter})) {
    auto other = dyn_cast<DotOp>(op);
    if (!other || other == dot || other.getType().getRank() != 2)
      continue;
    if (auto mma =
            dyn_cast<SimdgroupMmaEncodingAttr>(other.getType().getEncoding()))
      return llvm::to_vector(mma.getWarpsPerCTA());
    hasChainedDot = true;
  }
  if (hasChainedDot)
    return shape[0] >= shape[1] ? SmallVector<unsigned>{unsigned(numWarps), 1}
                                : SmallVector<unsigned>{1, unsigned(numWarps)};
  return balancedWarps(shape, numWarps);
}

Value convertTo(OpBuilder &b, Value v, Attribute encoding) {
  auto type = cast<RankedTensorType>(v.getType());
  return ConvertLayoutOp::create(b, v.getLoc(),
                                 type.cloneWithEncoding(encoding), v);
}

struct AppleAccelerateMatmulPass
    : impl::AppleAccelerateMatmulBase<AppleAccelerateMatmulPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    SmallVector<DotOp> dots;
    getOperation().walk([&](DotOp dot) {
      if (canUseSimdgroupMatrix(dot))
        dots.push_back(dot);
    });
    for (DotOp dot : dots) {
      auto retType = dot.getType();
      auto warps = warpsPerTile(dot, retType.getShape(), lookupNumWarps(dot));
      auto mma = SimdgroupMmaEncodingAttr::get(ctx, warps);
      OpBuilder b(dot);
      Value a =
          convertTo(b, dot.getA(), DotOperandEncodingAttr::get(ctx, 0, mma, 0));
      Value bv =
          convertTo(b, dot.getB(), DotOperandEncodingAttr::get(ctx, 1, mma, 0));
      Value c = convertTo(b, dot.getC(), mma);
      auto newDot = DotOp::create(
          b, dot.getLoc(), retType.cloneWithEncoding(mma), a, bv, c,
          dot.getInputPrecision(), dot.getMaxNumImpreciseAcc());
      dot.replaceAllUsesWith(
          convertTo(b, newDot.getResult(), retType.getEncoding()));
      dot.erase();
    }
  }
};

} // namespace
} // namespace mlir::apple_tt
