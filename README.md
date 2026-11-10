# AppleGPU: An MLIR Dialect and Backend for Apple GPUs

AppleGPU is an out-of-tree [MLIR](https://mlir.llvm.org/) dialect and backend for Apple GPUs. Upstream MLIR supports NVIDIA, AMD and Intel GPUs through dedicated dialects, but has no equivalent for Apple silicon. This project aims to provide one: a reusable backend that any MLIR-based compiler can target, exposing Apple-specific features such as SIMD-group operations and `simdgroup_matrix`.

It provides two tools: `applegpu-opt`, which runs the lowering and optimization passes, and `applegpu-translate`, which emits Metal Shading Language (MSL).

## Why a separate dialect

- **No direct path today:** MLIR reaches Apple GPUs only indirectly, through SPIR-V and a translator, and upstream Triton has no Apple backend.
- **Apple-specific operations:** SIMD-group operations and `simdgroup_matrix` need their own ops, as NVIDIA's do in NVGPU.
- **Decisions in passes:** Apple-specific choices live in the IR, so MSL and AIR get the same optimizations.
- **One target for every frontend:** Triton and standard MLIR lower to the same operations.
- **Learning MLIR:** building a dialect, its passes and its emitters end to end is how I am learning MLIR.

## Design

![AppleGPU compilation pipeline](docs/applegpu-pipeline.svg)

Triton and standard MLIR share the same dialect, passes and emitters. The backend never depends on a specific frontend, and all performance decisions are made before the emitters so that every output format benefits from them.

## Roadmap

We support MSL first, since it is readable and validated by Apple's Metal compiler. AIR will follow once the Triton and generic paths are ready.

1. **MSL foundation:** the dialect, its first operations and an MSL emitter, tested end to end on an Apple GPU.
2. **Triton and generic MLIR:** lowering Triton kernels and standard MLIR GPU code to the dialect, checked against CPU references and NVIDIA GPUs.
3. **AIR:** a second emitter that produces AIR directly, validated against the MSL path.

Performance work, such as `simdgroup_matrix` support and benchmarks against MLX, runs alongside phases 2 and 3.

## Requirements

An Apple silicon Mac with full Xcode and the Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`), CMake and Ninja. Without the Metal toolchain, the dialect still builds but the GPU tests are skipped.

## Building - Component Build

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir \
    -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit \
    -DMLIR_INCLUDE_TESTS=ON
cmake --build . --target check-applegpu
```
To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```
**Note**: Pass `-DMLIR_INCLUDE_TESTS=ON` as above, otherwise the `check-applegpu` target is not created. Also make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM in order to install `FileCheck` to the chosen installation prefix.

## Building - Monolithic Build

This setup assumes that you build the project as part of a monolithic LLVM build via the `LLVM_EXTERNAL_PROJECTS` mechanism.
To build LLVM, MLIR, the dialect and launch the tests run
```sh
mkdir build && cd build
cmake -G Ninja `$LLVM_SRC_DIR/llvm` \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_EXTERNAL_PROJECTS=applegpu-dialect -DLLVM_EXTERNAL_APPLEGPU_DIALECT_SOURCE_DIR=../
cmake --build . --target check-applegpu
```
Here, `$LLVM_SRC_DIR` needs to point to the root of the monorepo.