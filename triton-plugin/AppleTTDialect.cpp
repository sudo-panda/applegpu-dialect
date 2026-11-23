#include "AppleTT.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

#include "AppleTTDialect.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "AppleTTAttrs.cpp.inc"

namespace mlir::apple_tt {

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

} // namespace mlir::apple_tt
