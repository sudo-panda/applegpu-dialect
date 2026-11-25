# AGENTS.md

Guidance for coding agents, and for people, working in this repository: what the project is, its rules, how to build and test it, and the plan. The reasons behind the rules are in `docs/design-faq.md`. Paths are relative to the repository root.

## AppleGPU

An out-of-tree MLIR dialect and backend for Apple GPUs. Frontends are lowered into `gpu` + `applegpu` + standard dialects, then emitted by `applegpu-translate`: standard MLIR by `applegpu-opt`, Triton kernels by our Triton plugin inside Triton's process. Triton support is optional: without it, nothing of Triton is fetched, built or tested. MSL is the main output format. AIR comes later, as a learning exercise rather than for speed: Apple's compiler optimizes MSL, and MSL already reaches `simdgroup_matrix` and M5's tensor ops (MetalPerformancePrimitives). AIR is undocumented, so the MSL path stays the default and the reference that AIR output is checked against.

## Rules

- Never edit `deps/`: it holds LLVM and Triton as set up by `utils/setup.sh`.
- Use `git mv` for every move or rename, so history is kept.
- A refactor must not change behavior. The same tests must pass before and after, except tests deleted together with the code they cover.

## Design rules

- The backend never depends on a frontend. `include/`, `lib/` and `tools/` never include, link or mention Triton. Triton code may depend on the `applegpu` dialect, never the other way round.
- Triton is optional. The CMake option `APPLEGPU_ENABLE_TRITON` (off by default; `utils/setup.sh` turns it on unless `--no-triton`) decides whether `triton-plugin/` is built and whether lit sees the Triton test folders at all. With it off, a user is affected by Triton only through the LLVM version.
- One LLVM: Triton's. Its commit is the `llvm_hash` in Triton's `cmake/llvm-info.json` at `TRITON_TAG`, never pinned separately. Code uses only APIs that this LLVM has; where newer LLVMs renamed something, use the name both have (for example `mlir/Dialect/Arith/IR/Arith.h`, `mlir/Dialect/Func/IR/FuncOps.h`).
- All decisions, including performance ones, happen in passes. For Triton kernels, Triton's reused passes and our plugin choose layouts, and the plugin's `convert-triton-to-applegpu` lowers them to per-thread `applegpu` IR inside Triton's process, using Triton's own analyses. Everything after that happens in `applegpu` passes. Emitters only print already-lowered IR.
- An emitter reports unsupported operations with `emitError`. It never prints a guess.
- `applegpu-opt` and `applegpu-translate` never link Triton. For Triton kernels, the whole compile happens inside Triton's process: the plugin runs Triton's passes, our lowering, our `applegpu` passes and the MSL emitter, and stores the MSL and the kernel's metadata on the module as string attributes for Triton's Python backend to read. Nothing crosses as text. Text is for tests, debugging and other frontends: the plugin can print its per-thread IR (`gpu` + `applegpu` + standard dialects), which `applegpu-opt` and `applegpu-translate` parse and verify, and Triton's IR is printed only for tests and debugging.
- Reuse Triton's passes instead of rewriting them. They run in Triton's process with target `apple:<family>` and 32 threads per warp:
  - TTIR: `inliner`, `rewrite_tensor_descriptor_to_pointer`, `canonicalizer`, `combine`, `reorder_broadcast`, `cse`, `symbol_dce`, `loop_unroll`.
  - TTGIR: `convert_to_ttgpuir`, `coalesce`, `f32_dot_tc(False)`, `remove_layout_conversions`, `optimize_thread_locality`, *Apple accelerate-matmul (plugin)*, `remove_layout_conversions`, `optimize_dot_operands`, `loop_aware_cse`, `fuse_nested_loops`, `canonicalizer`, `triton_licm`, `canonicalizer`, `combine_tensor_select_and_if`, *Apple pipelining (later, only if measured)*, `canonicalizer`, `loop_aware_cse`, `optimize_dot_operands`, `remove_layout_conversions`, `reduce_data_duplication`, `reorder_instructions`, `loop_aware_cse`, `symbol_dce`, `sccp`, `cse`, `canonicalizer`.
  - All of these run without errors with `apple:m1` on vector add, a matmul, 68 vLLM worker kernels and float-atomic kernels, and give the same TTGIR as `cuda:80` except for float atomics.
  - Never use NVIDIA's or AMD's `accelerate_matmul` (it asserts with an Apple target), pipelining (it emits async copies), `prefetch`, or the cluster, TMA, tensor-memory and warp-specialization passes.
  - Pointer arguments need `tt.divisibility = 16`, and unit strides specialized to a constexpr 1; otherwise `coalesce` keeps one element per thread.
- Everything Apple-specific on the Triton side lives in `triton-plugin/`, never in patches to Triton: the Apple counterparts of Triton's hardware passes, and the lowering to `applegpu`. The plugin defines an Apple matrix layout attribute implementing Triton's `LayoutEncodingTrait`, `DistributedEncodingTrait` and `MmaEncodingTrait`, registers `DialectInferLayoutInterface` and `DialectVerifyTensorLayoutInterface` by forwarding to TritonGPU's implementations, and adds `apple-accelerate-matmul`. Triton's own `remove_layout_conversions` then propagates the layout, and Triton's verifier checks it against the module (rank, threads per warp, `ttg.num-warps`). The plugin is tied to the exact Triton release and needs Triton built with `TRITON_EXT_ENABLED=1`; rebuild it, and expect small fixes, with every Triton update. It links our `applegpu` libraries (dialect, passes and MSL emitter) but takes MLIR and LLVM from the process that loads it (`libtriton` or `triton-opt`): a second copy of MLIR in one process breaks its global registries.
- Float atomics: Triton has no target information for `apple`, so `coalesce` gives float `atomic_rmw` up to 128 bits per thread. `convert-triton-to-applegpu` splits them into scalar atomics.
- When Triton IR is printed (tests, debugging), it is printed with `--mlir-print-op-generic --mlir-print-local-scope --mlir-print-debuginfo`, so that tools without Triton's dialects can read and reprint it unchanged. Without `--mlir-print-local-scope`, a `#ttg.slice` refers to its parent through an alias such as `#blocked`, which such a tool drops (3 of 68 vLLM kernels broke). With the flag, Triton → text → `applegpu-opt` → text → Triton was identical on 137 files.
- Launching: never copy data (unified memory), encode on the caller's stream (PyTorch's MPS stream), never commit or wait per kernel. These matter more for vLLM's small kernels than anything inside them.
- Hardware facts (SIMD-group width, threadgroup memory, matrix shapes, atomics, `bfloat` support, tile sizes) come from one table keyed by Apple GPU family, never from constants spread across passes.
- `simdgroup_matrix<T, 8, 8>` fragments are a known layout: each lane holds 2 adjacent elements of one row. As a linear layout: register bit → column bit 0; lane bits 0–4 → column bit 1, row bit 0, row bit 1, column bit 2, row bit 2. Measured on M4 for `float` and `half`; MLX uses the same formula (`get_coord` in `steel/gemm/mma.h`) on every generation. A GPU test fails if it ever changes. M5's `cooperative_tensor` fragments (Metal 4 tensor ops) are opaque and device specific and get a separate, opaque layout.

## Current state

- Dialect: one op, `applegpu.simdgroup_sum` (f16/f32/i32, `Pure`, MSL `simd_sum`). Its verifier rejects it outside a `gpu.func` kernel.
- `applegpu-opt` registers no passes yet: it only parses, verifies and prints the applegpu, arith, func, gpu and memref dialects.
- `applegpu-translate` has one translation, `--mlir-to-msl`, implemented in `lib/Target/MSL/TranslateToMSL.cpp` (`MSLEmitter` class). It handles 1-D memref kernel arguments, `gpu.thread_id`, 1-D `memref.load`/`memref.store` and `simdgroup_sum`. Anything else fails with `emitError`.
- Integration tests (for example `test/Integration/simdgroup_sum_run.mlir`): `applegpu-translate` → `xcrun metal` → `metallib` → `%run_grid`, then FileCheck or `diff`. `test/Integration/lit.local.cfg` marks them unsupported when `xcrun -sdk macosx metal --version` fails.
- GPU runner: `test/runners/run_grid.mm` (Objective-C++), built once by CMake on macOS as `build/test/runners/run-grid` and used in tests as `%run_grid`. It is a test helper, not installed. Usage: `run-grid <metallib> <kernel> <threadgroups> <threads-per-group> <arg>...`, where each arg binds the next `[[buffer(i)]]`: `f32:<count>:iota`, `f32:<count>:iota*<k>`, `f32:<count>:fill=<v>` or `i32:<value>` (scalar, `setBytes`). It prints every float buffer as `arg<i>[<j>] = <value>` (whole numbers as `3.0`, like Python) and the GPU name to stderr.
- TTGIR capture: `test/runners/triton_to_ttgir.py <kernel.py> <kernel> --signature ... [--constexpr ...] [--divisible ...] [--num-warps 4] [--target apple:m1] [--plugin PATH] [--vllm] --triton-opt PATH` (in lit: `%triton_to_ttgir`) runs the reused TTIR and TTGIR passes from the design rules and prints generic TTGIR through `triton-opt` with the three print flags (Triton's Python API cannot print the generic form). Pointer arguments get `tt.divisibility = 16`; `--constexpr` turns arguments into constants (pass unit strides as `=1`); `--divisible` marks integer arguments that are multiples of 16, as Triton's launcher does; parameters left out take their default when it is `None` or they are `tl.constexpr`. `--vllm` registers an inactive `cpu` Triton backend in the script's process, because vLLM's macOS build otherwise turns Triton off and its kernels become placeholders. With `--plugin`, the plugin's passes run after `optimize_thread_locality`, and `triton-opt` loads it too, to print its attributes. Triton's NVIDIA backend is used only by the front end that builds TTIR (libdevice table, dot size checks); no NVIDIA pass runs. `test/Triton/triton_to_ttgir.mlir` checks vector add's layout and the round trip through `applegpu-opt`.
- Triton end-to-end test: `test/Integration/Triton/vector_add.py` holds the kernel, its RUN lines and a reference printed when the file is run. Pipeline: `triton_to_ttgir.py` (`BLOCK = 512`, 4 warps: 128 threads of 4 elements) → `applegpu-opt --convert-ttgir-to-applegpu --applegpu-distribute` → `applegpu-translate` → metallib → `%run_grid` (2 programs of 128 threads) → `diff`. It is `XFAIL`; its RUN lines still name an earlier design's passes, and plan step 1.11 replaces them with the plugin's `convert-triton-to-applegpu`. Remove the `XFAIL` line when it passes. Kernel arguments take buffer slots in argument order; scalars pass by value.
- `test/Triton/plugin_extend_with.py` checks that Triton loads pass plugins: it loads Triton's example plugin with `passes.plugin.extend_with` and runs its pass, which renames every function to `foo`.
- Triton plugin: `triton-plugin/` (`AppleTT.td`, `AppleTTDialect.cpp`, `AppleAccelerateMatmul.cpp`, `Plugin.cpp`) builds `libAppleTritonPlugin` with the `apple_tt` dialect and the `apple_accelerate_matmul` pass. The layout `#apple_tt.simdgroup_mma<{warpsPerCTA = [m, n]}>` has the measured 8x8 fragment mapping, SIMD-groups row-major (N first), repeated over the tensor; dot operands use the same fragment with (M, K) or (K, N), shared by the SIMD-groups of a row (A) or column (B). The dialect forwards Triton's layout inference and layout verification to TritonGPU, so a wrong SIMD-group count is reported. The pass handles 2-D dots with a blocked result, f16/bf16/f32 inputs, an f32 accumulator and M, N, K multiples of 8, and places SIMD-groups as `warpsPerTileV2` does. Its CMake project reads Triton's source folder, LLVM, compiler and version from Triton's `CMakeCache.txt` (`-DTRITON_BUILD_DIR`); the top-level `CMakeLists.txt` builds it as an `ExternalProject` into `build/triton-plugin/` when it finds a Triton build (`APPLEGPU_TRITON_BUILD_DIR`, default `deps/triton/build/*`), and `check-applegpu` rebuilds it. Tests in `test/TritonPlugin/`: loading the plugin (`--show-dialects`), the attribute's syntax, the layout as printed by `triton-tensor-layout`, the layout against the GPU (`simdgroup_mma_hardware.metal`: where `simdgroup_load` puts each element, and the product of operands written straight into fragments at the operand layouts' positions, for `float`, `half` and `bfloat`, compared with the plugin's layouts by `test/runners/check_simdgroup_fragments.py`; needs the `metal` feature), the verifiers, layout inference through TritonGPU (`infer_layout.mlir`: `tt.reduce` and `tt.broadcast` on the layout), the pass on hand-written TTGIR (`accelerate_matmul.mlir`: SIMD-group grids, chained dots, and dots that stay FMA dots), a GEMM (`matmul.py`) and vLLM's `kernel_unified_attention` (needs the `vllm` lit feature): both dots, the softmax and the loop-carried accumulator stay in the Apple layout, and the only conversion left is before the store.
- `test/Triton/lit.local.cfg`, `test/TritonPlugin/lit.local.cfg` and `test/Integration/Triton/lit.local.cfg` accept `.py` tests and marks them unsupported unless lit has the `triton` feature (set in `test/lit.cfg.py` when the Triton Python can `import triton` and `triton-opt` exists).

## Build and test

`utils/setup.sh` sets up LLVM, Triton and this project, then runs the tests. Versions come from `utils/pins.txt`. Every step is skipped or incremental when already done, so rerunning it after changing a pin updates only what changed. Options: `--no-triton`, `--with-vllm` (PyTorch, vLLM and vllm-metal in `.venv`; macOS only), `--llvm-build DIR` (use an existing LLVM/MLIR build), `--jobs N`, and `PYTHON=<python3.13>` for the venv.

```text
build/              this project's build (ignored by git)
deps/llvm-project/  LLVM sources at the pinned commit
deps/llvm-build/    LLVM and MLIR build
deps/triton/        Triton sources and build at the pinned tag
deps/vllm-metal/    vllm-metal sources at the pinned tag (--with-vllm)
.venv/              Python venv with Triton installed (editable)
```

After setup, rebuild and test with `ninja -C build check-applegpu`. By hand:

```sh
cmake -G Ninja -S . -B build \
    -DMLIR_DIR=<llvm-build>/lib/cmake/mlir \
    -DLLVM_EXTERNAL_LIT=<llvm-build>/bin/llvm-lit \
    -DMLIR_INCLUDE_TESTS=ON
ninja -C build check-applegpu
```

Run one test (build the tools first):

```sh
ninja -C build applegpu-opt applegpu-translate
<llvm-build>/bin/llvm-lit -v build/test/Target/MSL/msl_emit.mlir
```

- Without `-DMLIR_INCLUDE_TESTS=ON` the `check-applegpu` target does not exist.
- Re-run `cmake` after moving files or changing any `CMakeLists.txt`.
- Removing a target leaves its old binary in `build/bin`. Delete `build/` and re-run `cmake` when that matters.
- `ninja -C build mlir-doc` builds the dialect docs from TableGen.
- The top-level `CMakeLists.txt` has an untested branch for monolithic LLVM builds (`LLVM_EXTERNAL_PROJECTS`), inherited from MLIR's `standalone` example. It is not documented or supported, and may be removed.
- `docs/applegpu-pipeline.svg` is the pipeline diagram used by `README.md`, and `docs/design-faq.md` gives the reason for each design decision in short answers. Keep both in step with the design rules and the plan.
- `test/Integration/` runs kernels on the GPU. It needs full Xcode and the Metal toolchain, and is reported as unsupported otherwise.
- Triton tests use two optional CMake cache variables, `APPLEGPU_TRITON_PYTHON` (a Python that can `import triton`) and `APPLEGPU_TRITON_OPT` (`triton-opt` from the same build). When unset, `test/CMakeLists.txt` looks for `.venv/bin/python` and `triton-opt` in `APPLEGPU_TRITON_BUILD_DIR`. The `cmake` output prints the paths it found (`Triton build: ...`, `Triton for tests: ...`). Lit features: `metal` (the Metal toolchain works), `triton`, `apple-triton-plugin` and `vllm` (the Triton Python finds `vllm`); substitutions `%triton_python`, `%triton_opt`, `%triton_tensor_layout`, `%triton_to_ttgir`, `%apple_triton_plugin` and `%vllm_dir`. Without them, the Triton tests are unsupported. With them, `test/Triton/plugin_extend_with.py` fails if Triton was built without `TRITON_EXT_ENABLED=1`.
- The venv uses Python 3.13, because vllm-metal requires `>=3.12,<3.14`. Triton's build is tied to one Python version (`build/cmake.<platform>-cpython-3.13`), so changing Python means a new `.venv` and a full Triton rebuild. Keep only one build folder in `deps/triton/build/`: the tests take the first `build/*/bin/triton-opt`.
- With 16 GB of memory or less, pass `--jobs 4`: Triton's default (2 × cores) can run the machine out of memory.
- `git clean -fdx` deletes `deps/` and `.venv/` with it; use `git clean -fdx -e deps/ -e .venv/`.

## Layout

```text
include/applegpu/
  Dialect/AppleGPU/IR/     AppleGPUDialect.td, AppleGPUOps.td and headers
                           (AppleGPUTypes.td when types are added)
  Conversion/              Passes.td, one folder per pass
  Target/MSL/              MSLEmitter.h
lib/                       mirrors include/
  Dialect/AppleGPU/IR/
  Conversion/
  Target/MSL/
tools/
  applegpu-opt/
  applegpu-translate/      only main(): registers translations
runtime/                   later: Metal runtime library (C interface, applegpu_* functions)
triton-plugin/             Triton plugin (only with APPLEGPU_ENABLE_TRITON): Apple matrix layout,
                           apple-accelerate-matmul and convert-triton-to-applegpu; links the
                           applegpu libraries, never linked into applegpu-opt
test/
  Dialect/AppleGPU/        ops.mlir, invalid.mlir, commandline.mlir
  Conversion/              one folder per pass
  Target/MSL/              emitter output tests
  Triton/                  needs Triton, no GPU (plugin loading, TTGIR capture)
  TritonPlugin/            tests of triton-plugin/, in Triton's tools
  Integration/             GPU runs (Triton/ for Triton kernels, .py tests)
  runners/                 run_grid.mm, triton_to_ttgir.py, run_kernel.py (helpers, not tests)
utils/                     setup.sh, pins.txt and other workflow scripts
docs/
```

Conventions:

- C++ is formatted with `clang-format` in LLVM style (`.clang-format` at the root), as MLIR and Triton are.
- Includes use the full path, for example `#include "applegpu/Dialect/AppleGPU/IR/AppleGPUOps.h"`.
- The C++ namespace is `mlir::applegpu`. Library names are `MLIRAppleGPU...`.
- Dialect libraries use `add_mlir_dialect_library`, passes `add_mlir_conversion_library` and emitters `add_mlir_translation_library`.
- Capitalized test folders mirror code layers. Lowercase folders hold helpers, not tests.
- The `--show-dialects` check lives in `test/Dialect/AppleGPU/commandline.mlir`, named after upstream's `mlir/test/mlir-opt/commandline.mlir`.

Upstream MLIR (`mlir/` in LLVM) is the reference for structure: `lib/Target/Cpp` for the emitter, and `NVGPU`, `GPUToNVVM` and `test/Integration/GPU/CUDA` for the NVIDIA equivalents of each layer.

## Plan

1. Kernels through Triton, in this order. The goal is that vLLM's own Triton attention and fused MoE run on this backend, not only its small worker kernels, so matrix multiplication (`tt.dot`) is on the critical path. TTGIR capture and the vector-add end-to-end test exist (`test/runners/triton_to_ttgir.py`, `test/Integration/Triton/vector_add.py`, `XFAIL`). Each step ends with the check that shows it is done.
   1. **Triton with plugins.** `utils/setup.sh` builds Triton with `TRITON_EXT_ENABLED=1`; without it, `libtriton` exports no symbols and no plugin loads. With the flag, Triton also builds its example pass plugin (`deps/triton/examples/plugins/TritonPlugin.cpp`) into `triton/plugins/`. *Done when* `test/Triton/plugin_extend_with.py` loads that plugin through `passes.plugin.extend_with` and its pass renames a function. Building a plugin out of tree is step 4.
   2. **PyTorch and vLLM in the venv.** `utils/setup.sh --with-vllm` installs vllm-metal's pinned torch nightly, vLLM and vllm-metal into `.venv`, pinned only by `VLLM_METAL_TAG` in `utils/pins.txt`: vllm-metal names the vLLM release, torch and mlx it needs. vllm-metal is installed from source into `deps/vllm-metal`, because its release wheels are for Python 3.12 only; the script then builds its Metal kernels as its own `install.sh` does. *Done when* `.venv/bin/python` can `import torch, vllm, vllm_metal` and `torch.backends.mps.is_available()` is true.
   3. **`triton_to_ttgir.py`** replaces `triton_to_ttir.py`. It runs the reused TTIR and TTGIR passes from the design rules and prints with the three flags. Options: `--signature`, `--constexpr`, `--divisible`, `--num-warps` (default 4), `--target` (default `apple:m1`), `--plugin PATH` and `--triton-opt PATH`; the plugin's slot stays empty until step 4. Pointer arguments get `tt.divisibility = 16`, integer arguments given the value 1 with `--constexpr` are specialized, and `--divisible` integers get `tt.divisibility = 16`, as Triton's launcher does. *Done when* a FileCheck test sees `sizePerThread = [4]` on vector add with `BLOCK = 512` and 4 warps, and a round trip through `applegpu-opt` leaves the TTGIR unchanged.
   4. **`triton-plugin/`**, a CMake project built against `deps/triton` (headers and generated `.inc` files) and Triton's LLVM, both read from Triton's `CMakeCache.txt`, linked with `-undefined dynamic_lookup`, with the plugin version set to the Triton release. It contains:
      - The layout `#apple_tt.simdgroup_mma<{warpsPerCTA = [m, n]}>`: 8×8 `simdgroup_matrix` tiles, one per SIMD-group at a time, SIMD-groups arranged by `warpsPerCTA`, repeated over the block. It implements `LayoutEncodingTrait` (a single CTA), `DistributedEncodingTrait` (`toLinearLayout` from the measured fragment mapping) and `MmaEncodingTrait` (the A and B operands use the same fragment mapping), and registers `DialectInferLayoutInterface` by forwarding to TritonGPU's implementation.
      - The pass `apple-accelerate-matmul`: gives `tt.dot` the Apple layout for fp16, bf16 and f32 inputs with f32 accumulators when M, N and K are multiples of 8, and places SIMD-groups the way NVIDIA's `warpsPerTileV2` places warps. Other dots stay FMA dots.
      - Lit tests that run Triton's `triton-opt` with `TRITON_PLUGIN_PATHS`.

      *Done when* on a GEMM and on `kernel_unified_attention`, both dots take the Apple layout, and after `remove_layout_conversions` the softmax and the accumulator stay in it. The only conversions left are into the second dot's operand and before the stores.
   5. **One LLVM, and Triton as an option.**
      - `utils/setup.sh` reads `llvm_hash` from Triton's `cmake/llvm-info.json` at `TRITON_TAG` (from `deps/triton` when it is cloned, otherwise fetched from GitHub, so `--no-triton` needs no Triton sources) and downloads Triton's prebuilt LLVM for it, as Triton's own build does, into `deps/llvm-<hash>/`. `--llvm-build DIR` still takes an LLVM built from that commit instead; the script checks the commit. `LLVM_COMMIT` leaves `utils/pins.txt`.
      - `lit` comes from pip into `.venv` (Triton's LLVM ships `FileCheck` and `not`, but not `lit` or `count`); `test/lit.cfg.py` uses only `FileCheck` and `not`.
      - The CMake option `APPLEGPU_ENABLE_TRITON`, off by default. When off: no Triton lookup, no `triton-plugin/`, and lit excludes `test/Triton`, `test/TritonPlugin` and `test/Integration/Triton`. `setup.sh` passes `ON` unless `--no-triton`.
      - The includes that only newer LLVMs have (`ArithDialect.h`, `FuncDialect.h`) change to names both have.

      *Done when* `setup.sh --no-triton` on a clean clone downloads only `llvm-info.json` and the LLVM build, builds no Triton code, and `check-applegpu` lists no Triton test; and with Triton, every test passes on Triton's LLVM.
   6. **Our code inside Triton's process.** `triton-plugin/` links the `applegpu` libraries from the main build (dialect, passes and MSL emitter), with MLIR and LLVM left to the loading process, and registers the `applegpu` dialect with Triton. It adds a pass, `apple_emit_msl`, that runs our `applegpu` passes and the emitter and stores the MSL on the module as a string attribute (`applegpu.msl`), and Triton's Python backend reads it with `get_str_attr`. *Done when* `triton-opt` with the plugin parses, verifies and prints `applegpu.simdgroup_sum` in a `gpu.func`; a Python test runs `apple_emit_msl` on that module through `passes.plugin` and reads the same MSL `applegpu-translate` prints; and `nm` shows that the plugin defines no MLIR or LLVM symbols.
   7. **Fragment mapping on more chips.** `test/TritonPlugin/simdgroup_mma_hardware.metal` (added with step 4) checks the result and operand fragments on the GPU it runs on; it passes on M4 for `float`, `half` and `bfloat`. *Done when* it has also passed on M1, M2, M3 and M5, and the GPU names are recorded here.
   8. **Per-thread ops in the `applegpu` dialect**, in MSL's terms and independent of Triton:
      - A pointer type `!applegpu.ptr<T, space>` (`device` or `threadgroup`) with `addptr`, `load` and `store` (scalar or vector, with an optional mask and fill value), `int_to_ptr`, `ptr_to_int` and `atomic_rmw`.
      - Threadgroup memory and SIMD-group ops: a threadgroup allocation, `simd_shuffle_xor` and the `simd_*` reductions besides `simdgroup_sum`.
      - `simdgroup_matrix` fragments: a type, `load`, `store`, `multiply_accumulate` and access to a lane's elements.

      *Done when* `test/Dialect/AppleGPU/ops.mlir` and `invalid.mlir` cover every new type and op, and every op has an emitter test (step 10).
   9. **`convert-triton-to-applegpu`**, a pass in `triton-plugin/`: TTGIR → per-thread `gpu` + `applegpu` + `arith`/`math`/`scf`, inside Triton's process. It uses Triton's target-independent analyses directly: each layout's `LinearLayout` for indices, `ModuleAxisInfoAnalysis` for vector widths and mask alignment, `ModuleAllocation` for threadgroup-memory offsets and `ModuleMembarAnalysis` for barriers. Reductions become in-thread, `simd_*` and threadgroup steps; dots become `simdgroup_matrix` operations or FMA loops; layout conversions go through threadgroup memory; float atomics are split. Threads per program are `ttg.num-warps` × 32. Triton's `TritonGPUToLLVM` patterns, which target the LLVM dialect, are the reference for each op. Its output feeds `apple_emit_msl` in the same process, and can also be printed as text. *Done when* FileCheck tests show the expected per-thread IR for vector add (4-wide loads), a row reduction, an FMA dot and a `simdgroup_matrix` dot, and `applegpu-opt` verifies each output.
   10. **The emitter** gains what these need: arithmetic, `scf.if`, `scf.for` and `scf.while`, `gpu.block_id`, pointers and vector loads, scalar arguments, threadgroup memory, barriers, SIMD-group ops and `simdgroup_matrix`. *Done when* every new op has an emitter test and every unsupported op still fails with `emitError`.
   11. **Vector add end to end.** `triton_to_ttgir.py` gains `--emit ttgir|applegpu|msl`: after Triton's passes it runs `convert-triton-to-applegpu` and, for `msl`, `apple_emit_msl`, and prints the stage asked for. The test compiles vector add to MSL in Triton's process, then builds the metallib and runs it with `run-grid`; it also prints the per-thread IR and checks that `applegpu-translate` gives the same MSL. *Done when* `vector_add.py` passes and its `XFAIL` line is removed.
   12. **Kernel metadata and a Python harness.** The emitter produces, next to the MSL, a JSON description of the kernel's ABI (argument → buffer slot, scalars, threads per program, grid; plan step 4.1): in the plugin as a second module attribute (`applegpu.metadata`), from `applegpu-translate` as a file. `test/runners/run_kernel.py` compiles a Triton kernel in Triton's process, loads the MSL with `torch.mps.compile_shader` or the metallib with `torch.mps.load_metallib`, runs it on MPS tensors and compares the results with a torch reference on the CPU. *Done when* vector add passes through the harness in f32, fp16 and bf16.
   13. **Matrix multiplication**, a plain Triton GEMM like Triton's matmul tutorial (fp16 or bf16 in, f32 accumulate, K loop). It goes through FMA dots first, then the plugin's `simdgroup_matrix` layout. *Done when* both match `torch.matmul` within tolerance, including sizes that are not a multiple of the block, and the harness reports their speed next to `torch.matmul` on MPS.
   14. **Attention**: `vllm/v1/attention/ops/triton_unified_attention.py`. Two dots per tile (`Q·K` 16×128·128×32 and `P·V` 16×32·32×128 for prefill with head size 128), an online softmax with row `max`/`sum` and `exp`, `scf.for` over tiles, and an `scf.while` binary search. *Done when* prefill and decode match a torch reference attention over the same paged KV cache.
   15. **Fused MoE**: `vllm/model_executor/layers/fused_moe/fused_moe.py`. One dot in a K loop, from 16×128·128×64 to 128×64·64×128 with the default configurations. *Done when* it matches a torch reference MoE layer for the default configurations.
   16. **vLLM's worker kernels**: loops and early returns, `tt.reduce`, pointer tables (`tt.int_to_ptr`) and integer atomics. Most of what they need exists by then. *Done when* each matches the torch replacement vllm-metal uses for it.

   Software pipelining (prefetching the next loop iteration into registers) is added only if measurements on steps 13–15 show a loss.

2. Test tiers, using lit features so each machine reports what it cannot run as unsupported:
   - Tier 1, no requirements and no Triton: per-thread `applegpu` IR → `applegpu-opt` → `applegpu-translate`, checked with FileCheck, including saved outputs of `convert-triton-to-applegpu` as fixtures.
   - Tier 2, `triton`: Triton kernels through Triton's passes and the plugin, checked with FileCheck and compared with the tier 1 fixtures, to catch Triton upgrades (layouts, Python pass names, plugin API).
   - Tier 3, Metal toolchain and GPU: the end-to-end run.
3. CI: GitHub-hosted runners for as much as possible. Linux runs tier 1 without Triton and tier 2 with it. LLVM is Triton's prebuilt download, so CI builds no LLVM; the Triton build is cached, keyed by `utils/pins.txt`. macOS also compiles the generated MSL to a `.metallib`. GPU runs (tier 3) happen on contributors' Macs. Later, a script (`utils/report-gpu-tests.sh`) runs tier 3 on a clean tree, posts the results as a PR comment and sets a `gpu-e2e` commit status that branch protection requires.
4. Plug and play: let Triton, or any package that builds for Metal, use the backend without building it. Start once `convert-triton-to-applegpu` works for vector add, except the metadata format, which is designed together with the pass because the pass decides the ABI.
   - Contract: MSL or `.metallib` out, from Triton kernels in Triton's process or from our IR as text (per-thread `gpu` + `applegpu` + standard dialects, from any frontend), plus a JSON metadata file per kernel: argument → `[[buffer(i)]]` slot, scalars by value, threads per program, grid mapping. Launchers read the metadata instead of hard-coding the ABI.
   - Order:
     1. Kernel ABI and metadata: a short doc, and the compiler writes the JSON (built in plan step 1.12).
     2. `applegpu-compile`, for frontends other than Triton: one driver, `applegpu-compile in.mlir -o k.metallib --emit-metadata k.json` (also `--emit=msl`). It runs the registered `applegpu-pipeline` and then the emitter. Ship this one binary rather than `-opt` and `-translate`, so LLVM is not packaged twice.
     3. Runtime library in `runtime/`: Objective-C++ with a C interface (`applegpu_load`, `applegpu_launch`, ...) driven by the metadata. `run-grid` becomes a thin wrapper around it; `mlir-runner` uses it later.
     4. `triton-applegpu`, a separate Python package registered under the `triton.backends` entry-point group (Triton 3.9 discovers out-of-tree backends there). Its `compiler.py` (a `BaseBackend`) loads the Triton plugin, runs the reused TTIR and TTGIR passes, `convert-triton-to-applegpu` and `apple_emit_msl` in Triton's process, and reads the MSL and metadata from the module's attributes; its `driver.py` (a `DriverBase`) launches through `torch.mps.load_metallib`, and its `is_active()` returns `torch.backends.mps.is_available()`. Its target is `GPUTarget("metal", "<GPU family>", 32)`. Its launcher takes the tensors themselves, not `data_ptr()`, so it can pass each `MTLBuffer` with its offset. It ships the plugin library built for the matching Triton release. It needs no `applegpu-compile`. `applegpu` installs and runs without it.
     5. vLLM integration, from `triton-applegpu` through `vllm.general_plugins` and `triton.backends` entry points, or as small pull requests to vllm-metal where cleaner:
        - Attention: vllm-metal's `get_attn_backend_cls` (`vllm_metal/platform.py`) always returns its own MPS backend; it must return vLLM's `TRITON_ATTN`. That backend needs KV-cache block sizes that are multiples of 16.
        - MoE models: vllm-metal's `configure_mps` rejects them (`model.is_moe`).
        - Triton detection: vLLM's macOS wheel is a `+cpu` build, which turns Triton off unless a `triton.backends` entry named `cpu` exists. Ship a `cpu` backend whose driver is always inactive, so ours is the single active driver.
        - KV-cache writes: `triton_reshape_and_cache_flash` calls `torch.cuda.get_device_capability`; patch it as vllm-metal already patches `torch.cuda.Stream`.
        - MoE backend: on an out-of-tree platform vLLM selects no MoE kernel; select the Triton one. The fused path also calls C++ ops the macOS wheel lacks (`moe_align_block_size`, `moe_sum`, `silu_and_mul`); replace them with Triton or torch code.
     6. CMake `install()`/export for `find_package(AppleGPU)`, then wheels (`applegpu` with the tool and runtime dylib, `triton-applegpu` with the plugin, which contains our dialect code) built on GitHub's macOS runners.
   - Size: LLVM is linked statically, only the parts used, and there is no LLVM codegen. Measured: `applegpu-opt` is 14 MB, 6.3 MB stripped, 2.1 MB gzipped (`libtriton.so` is 250 MB). Strip release binaries.
   - Packaging LLVM build: configure with `-DLLVM_ENABLE_ZSTD=OFF`, since a build may link Homebrew's `libzstd`, which a packaged binary cannot rely on. Check with `otool -L` that release binaries link only system libraries.
   - Launch with PyTorch: Triton's own launchers pass `data_ptr()`, which loses the `MTLBuffer`; ours takes the tensors. An MPS tensor's `MTLBuffer` is `tensor.storage().data()` bit-cast to `id<MTLBuffer>`, at byte offset `storage_offset() × element_size()`, encoded on `at::mps::getCurrentMPSStream()`. Two ways with the same speed: an Objective-C++ extension linked to libtorch (vllm-metal's paged attention), or `torch.mps.load_metallib` / `torch.mps.compile_shader` from Python (vllm-metal's GDN kernels). Prefer the Python route: no libtorch ABI coupling, which matters because vllm-metal pins a nightly torch. Known issue: fp16/bf16 scalar binding through it is reported unreliable, so pass such scalars as 32-bit values or in a small buffer (the ABI is ours).
   - Open question: compiling MSL at run time (`torch.mps.compile_shader` or `newLibraryWithSource:`) so end users need no Xcode (preferred for the plug-in; `.metallib` output stays for the tools).
5. Target: vLLM on Apple silicon, through vllm-metal.
   - Why: the goal is an LLM running through Triton on the Mac GPU. The popular laptop tools (Ollama, LM Studio, llama.cpp, MLX) do not use Triton. vLLM is the most popular framework built on Triton kernels, and `vllm-project/vllm-metal` is its official Apple silicon plug-in.
   - Where we plug in: vllm-metal's default backend (`VLLM_METAL_BACKEND=mlx`) runs models with MLX and uses no Triton. Its `mps` backend runs vLLM's own PyTorch model code and replaces vLLM's Triton kernels with hand-written code, through `vllm.triton_utils.dispatcher.register_kernels` and by patching functions (`vllm_metal/pytorch_backend/bindings.py`). The goal is to compile those kernels with our backend instead. Attention on that path is vllm-metal's own `pagedattention.metal`; the goal is to run vLLM's Triton attention backend (`vllm/v1/attention/backends/triton_attn.py`) and fused MoE through our backend instead, with the integration in plan step 4.5.
   - Versions: vllm-metal tracks vLLM 0.34.0; it needs Python `>=3.12,<3.14`, torch `2.16.0.dev20261013` (nightly) and `mlx==0.32.4`. `VLLM_METAL_TAG` is the only pin for these: vllm-metal names the rest. vLLM 0.34.0 pins torch 2.14.0, which requires Triton `~=3.8.0` on Linux. On macOS, torch does not depend on Triton at all.
   - Triton version: stay on v3.9.0. Check: 70 of the 83 launched kernels in vLLM 0.34.0's `vllm/v1/worker/gpu/` compile to TTIR with 3.9, including `_gather_block_tables_kernel`, `_compute_slot_mappings_kernel` and `_apply_write_kernel` with exact launch-site types. The 13 failures come from the checker (guessed types, names from torch and vLLM modules not loaded), not from Triton. The 3.8 → 3.9 API diff only drops `tl.advance` from the exports, and vLLM uses neither it nor `make_block_ptr`. Do not go back to 3.8: the plugin interfaces for out-of-tree layouts first appear in 3.9.
   - TTIR operations in the worker kernels (number of kernels out of 68): every kernel uses masked `tt.load`/`tt.store`, `tt.addptr`, `tt.splat`, `tt.make_range`, `tt.get_program_id` and integer `arith`. Then `scf.for` (32), `cf.cond_br` (32, from early `return`), `tt.reduce` (27), `scf.if` (22), int/float conversions and comparisons, `math.exp`/`math.log` (4), `tt.atomic_rmw` (3), `scf.while` (2), `tt.int_to_ptr` (2, pointer tables), and one each of `tt.scan`, `tt.bitcast`, `tt.call`. No `tt.dot`.
   - The 68 worker kernels as TTGIR use only `#ttg.blocked` and `#ttg.slice` layouts.
   - Kernels with `tt.dot` in vLLM 0.34.0: `kernel_unified_attention` in `vllm/v1/attention/ops/triton_unified_attention.py` and `fused_moe_kernel` in `vllm/model_executor/layers/fused_moe/fused_moe.py` (shapes in plan steps 1.14 and 1.15). With fp16 and bf16, all of them compile to TTIR with Triton 3.9. They use `tt.dot` with f32 accumulators, row reductions, `tl.where` masks and early returns, and no atomics, inline assembly, libdevice or block pointers. Not compiled yet: fp8/int8 KV caches, quantized MoE variants (w8a8, w8a16, GPTQ/AWQ) and the tensor-descriptor paths (CUDA and XPU only).
   - Related projects: `triton-msl` (community, Triton → MSL, claims Triton's `test_core.py` passes, built on Triton 3.7.0) and triton-ext PR #127 in `triton-lang` (a minimal Apple GPU backend, TTGIR → MSL). Both are Triton-only backends; this project is an MLIR dialect and backend usable by any frontend.
