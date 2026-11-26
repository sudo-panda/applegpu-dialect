// Without Triton support (APPLEGPU_ENABLE_TRITON off), lit does not list any
// Triton test; with it, it lists them, as unsupported if Triton is missing.
// Each run only lists the tests (--show-tests), without running them.
// RUN: %lit --show-tests -D applegpu_enable_triton=0 %applegpu_test_dir \
// RUN:   | FileCheck %s --check-prefix=OFF
// RUN: %lit --show-tests -D applegpu_enable_triton=1 %applegpu_test_dir \
// RUN:   | FileCheck %s --check-prefix=ON

// OFF: APPLEGPU :: Dialect/AppleGPU/ops.mlir
// OFF-NOT: Triton/
// OFF-NOT: TritonPlugin/

// ON: APPLEGPU :: Integration/Triton/vector_add.py
// ON: APPLEGPU :: Triton/plugin_extend_with.py
// ON: APPLEGPU :: TritonPlugin/commandline.mlir
