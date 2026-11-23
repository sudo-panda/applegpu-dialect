// Entry point Triton calls when it loads this library (TRITON_PLUGIN_PATHS,
// or ir.extend_dialects_with from Python).

#include "AppleTT.h"

#include "triton/Tools/PluginUtils.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

void registerDialect(DialectRegistry *registry) {
  registry->insert<apple_tt::AppleTTDialect>();
}

} // namespace

TRITON_PLUGIN_API plugin::PluginInfo *tritonGetPluginInfo() {
  static plugin::DialectInfo dialects[] = {
      {"apple_tt", TRITON_VERSION, registerDialect}};
  static plugin::PluginInfo info = {TRITON_PLUGIN_API_VERSION,
                                    "AppleTritonPlugin",
                                    TRITON_VERSION,
                                    nullptr,
                                    0,
                                    dialects,
                                    1,
                                    nullptr,
                                    0,
                                    TRITON_VERSION};
  return &info;
}
