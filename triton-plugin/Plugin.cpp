// Entry point Triton calls when it loads this library (TRITON_PLUGIN_PATHS,
// or passes.plugin.extend_with and ir.extend_dialects_with from Python).

#include "AppleTT.h"

#include "mlir/Pass/PassManager.h"
#include "triton/Tools/PluginUtils.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

void addAccelerateMatmul(PassManager *pm, const std::vector<std::string> &) {
  pm->addPass(apple_tt::createAppleAccelerateMatmul());
}

void registerAccelerateMatmul() { apple_tt::registerAppleAccelerateMatmul(); }

void registerDialect(DialectRegistry *registry) {
  registry->insert<apple_tt::AppleTTDialect>();
}

} // namespace

TRITON_PLUGIN_API plugin::PluginInfo *tritonGetPluginInfo() {
  static plugin::PassInfo passes[] = {{"apple_accelerate_matmul",
                                       TRITON_VERSION, addAccelerateMatmul,
                                       registerAccelerateMatmul}};
  static plugin::DialectInfo dialects[] = {
      {"apple_tt", TRITON_VERSION, registerDialect}};
  static plugin::PluginInfo info = {TRITON_PLUGIN_API_VERSION,
                                    "AppleTritonPlugin",
                                    TRITON_VERSION,
                                    passes,
                                    1,
                                    dialects,
                                    1,
                                    nullptr,
                                    0,
                                    TRITON_VERSION};
  return &info;
}
