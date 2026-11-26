// Triton loads the plugin from TRITON_PLUGIN_PATHS and registers its dialect.
// RUN: env TRITON_PLUGIN_PATHS=%apple_triton_plugin %triton_opt --show-dialects \
// RUN:   | FileCheck %s
// RUN: %triton_opt --show-dialects | FileCheck %s --check-prefix=NO-PLUGIN
// CHECK: Available Dialects:
// CHECK-SAME: apple_tt
// NO-PLUGIN: Available Dialects:
// NO-PLUGIN-NOT: apple_tt
