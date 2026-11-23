// RUN: env TRITON_PLUGIN_PATHS=%apple_triton_plugin %triton_opt %s \
// RUN:   -split-input-file -verify-diagnostics

// expected-error @+1 {{simdgroup_mma supports rank 2 only, got warpsPerCTA of size 3}}
#mma = #apple_tt.simdgroup_mma<{warpsPerCTA = [1, 2, 2]}>

// -----

// expected-error @+1 {{warpsPerCTA must be powers of 2}}
#mma = #apple_tt.simdgroup_mma<{warpsPerCTA = [3, 1]}>
