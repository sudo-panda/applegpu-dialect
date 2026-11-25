# Design FAQ

Short answers to "why is it done this way?". The rules themselves are in [`AGENTS.md`](../AGENTS.md); this file gives the reasons.

## Goal and scope

**Why an MLIR dialect for Apple GPUs at all?**
Upstream MLIR has dialects for NVIDIA, AMD and Intel GPUs but none for Apple's. Its only route to Apple GPUs goes through SPIR-V and a translator, which cannot express Apple features such as SIMD-group reductions and `simdgroup_matrix`. A dialect gives every MLIR-based frontend one place to target.

**Why not just write a Triton backend?**
Triton is the first frontend, not the only one. Keeping the backend independent of Triton lets standard MLIR GPU code use it too, and keeps Triton's frequent internal changes out of the core.

**Why is vLLM the target?**
The goal is an LLM running through Triton on a Mac GPU. The popular laptop tools (Ollama, LM Studio, llama.cpp, MLX) do not use Triton. vLLM is the most popular framework built on Triton kernels, and vllm-metal is its official Apple silicon plug-in.

**Why must attention and MoE run through Triton, not only vLLM's small kernels?**
Otherwise Triton would only run bookkeeping, while the model's real work stays in hand-written Metal code. vLLM's Triton attention and fused MoE kernels are where the computation is, and both use matrix multiplication.

**Why does matrix multiplication come right after vector add?**
Attention and MoE need it, so it is on the critical path. Vector add comes first because it is the smallest kernel that exercises the whole pipeline; a matmul then builds on that working base instead of on nothing.

## Output format

**Why MSL and not AIR?**
MSL is documented and validated by Apple's compiler, which also does the low-level optimization. It already reaches everything we need, including `simdgroup_matrix` and M5's tensor operations. AIR is undocumented.

**Then why is AIR still planned?**
As a learning exercise, not for speed. The MSL path stays the default and the reference that AIR output is checked against.

**Why do emitters make no decisions?**
Decisions made in passes can be tested with FileCheck and are shared by every output format. An emitter that chose things would need the same logic again for AIR.

**Why does an emitter fail with `emitError` instead of printing its best guess?**
A guess produces kernels that compile and give wrong answers, which is the hardest kind of bug to find on a GPU.

## Triton boundary

**Why never link Triton into `applegpu-opt`?**
Triton pins its own LLVM, which differs from ours. Linking both into one binary means two copies of LLVM, symbol clashes, and breakage every time Triton changes its C++ API.

**How does Triton's IR reach us without linking?**
As MLIR text in generic form, which `applegpu-opt` parses with `--allow-unregistered-dialect`. Text is the format that best tolerates two different LLVM versions; MLIR bytecode is tied to the version that wrote it.

**Does text lose information?**
Not when printed with `--mlir-print-op-generic --mlir-print-local-scope --mlir-print-debuginfo`. Without `--mlir-print-local-scope`, a slice layout refers to its parent through an alias that `applegpu-opt` drops, because it cannot see inside Triton's attributes. With the flag, a round trip through `applegpu-opt` reproduces Triton's IR exactly.

**Why read TTGIR instead of TTIR?**
TTGIR carries Triton's layout decisions: which thread holds which elements. Reading TTIR would mean redoing all of those decisions ourselves.

**Why reuse Triton's passes instead of writing our own?**
They are mature and tested: contiguity analysis, coalescing and layout propagation alone are about 4,000 lines. Rewriting them would add bugs and take months. All of Triton's target-independent passes run unchanged with an Apple target.

**Why are some of Triton's passes left out?**
They target hardware Apple GPUs lack: clusters, TMA, tensor memory, warp specialization and async copies. Skipping them loses nothing on Apple GPUs. Only matrix multiplication and software pipelining need Apple versions.

**Why put the Apple matmul pass in a Triton plugin?**
Attention needs the matrix layout to flow through softmax, reductions and loops. Triton's own layout propagation already does that for any layout that implements its interfaces, and a plugin can provide one. The alternatives were worse: rewriting propagation ourselves, or copying about 10,000 lines of Triton code and keeping them in sync.

**Doesn't the plugin break "never link Triton"?**
It narrows it. The plugin is a separate library built against Triton's LLVM and loaded only into Triton's process. `applegpu-opt` and our LLVM still never meet Triton.

**What does the plugin cost?**
Triton must be built with `TRITON_EXT_ENABLED=1`. The plugin must be rebuilt for every Triton release, with small source fixes most of the time. The tier 2 tests catch breakage after an upgrade.

**How is the plugin built without mixing the two LLVMs?**
As a separate CMake project that takes Triton's source folder, LLVM, compiler and version from Triton's own build cache. Our build starts it as an external project, so it builds with `check-applegpu`, but nothing of ours links it.

**Why not patch Triton?**
A patched Triton is a fork to maintain forever. The plugin interface exists so that out-of-tree backends do not need one.

## Layouts and lowering

**Why keep a tile level instead of lowering Triton straight to threads?**
Every performance decision is a layout decision: wide loads, coalescing, reductions inside one SIMD-group, matrix units. Splitting into threads first fixes the worst layout and throws away the information needed to choose a better one.

**Why mirror Triton's blocked and slice layouts instead of designing our own format?**
Our input is Triton, so a layout in our IR then reads the same as in TTGIR, and Triton's tools work on it. It also covers everything vLLM's kernels need.

**Why our own tile-level ops instead of keeping Triton's ops?**
Triton's ops arrive as opaque generic ops that `applegpu-opt` cannot verify, and the rest of our pipeline should not depend on Triton's op set. We define ops only for what is specific to Triton (pointer arithmetic, masked loads and stores, dots, reductions, layout conversions); standard `arith`, `math` and `scf` ops work on our tensors directly.

**Why a pointer type of our own?**
Triton computes on tensors of pointers (`tt.addptr`), which matches MSL's `device` pointer arithmetic directly. `memref` would force every address into a base plus index form that Triton kernels do not have.

**How do we read Triton's layouts without Triton's dialect?**
With a small hand-written parser for the few attribute and type forms we accept. Anything else fails with `emitError`, so a Triton change that alters the text is reported, never misread.

**Why copy Triton's `LinearLayout` instead of writing our own coordinate math?**
Every layout (blocked, slice, dot operand, the Apple matrix layout) needs "which register of which lane holds which element", and `LinearLayout` answers that for all of them in one representation. It is self-contained, compiles against our LLVM unchanged, and is stable upstream. Copying it is the cheap part of vendoring; the expensive part, layout propagation, stays in Triton.

**Why C++20?**
The copied `LinearLayout` code uses it, and Triton builds as C++20. Switching the project keeps the copy unchanged, so upstream fixes apply cleanly.

**Why one distribution pass instead of several stages?**
Intel's Triton backend removed its staged lowering once the single-step path matched its performance. Stages can be added later if they pay off.

**Why is the `simdgroup_matrix` fragment layout treated as known, not opaque?**
It was measured on the GPU: each lane holds two adjacent elements of one row, in a pattern that is a plain permutation of bits, so it is a Triton linear layout. MLX relies on the same mapping on every chip generation. Knowing it lets attention's softmax work on the fragments in registers instead of going through threadgroup memory.

**Why is M5's matrix layout handled separately?**
M5's matrix units are reached through Metal 4 tensor operations, whose fragment layout Apple defines as opaque and device specific. It gets its own layout, used only through Apple's library. `simdgroup_matrix` remains the path that works on every chip.

**Why split float atomics in the lowering?**
Triton has no target information for Apple GPUs, so `coalesce` widens float atomic adds to several elements per thread. Metal atomics work on one element, so the lowering turns them back into scalar atomics.

**Why is software pipelining deferred?**
Triton's pipelining relies on asynchronous copies, which Metal does not have. Apple GPUs hide memory latency by running many SIMD-groups at once. An Apple version, prefetching the next loop iteration into registers, is added only if measurements show a loss.

**Why put hardware facts in one table keyed by GPU family?**
They differ across generations: `bfloat` support, 64-bit atomics, Dynamic Caching from M3 onward, and M5's matrix units. Best tile sizes differ too. One table keeps those differences in one place instead of scattered through passes.

## Running kernels

**Why launch on PyTorch's MPS stream without copies?**
Mac memory is shared by the CPU and GPU, so the data is already where the GPU can read it. For vLLM's many small kernels, the cost is in dispatching, copying and waiting, not in the kernels themselves.

**Why load kernels through `torch.mps` from Python instead of a libtorch C++ extension?**
Both run at the same speed. A C++ extension must be rebuilt for every torch version, and vllm-metal pins a nightly torch.

**Why test kernels through a Python harness and not only `run-grid`?**
Attention and MoE need fp16 and bf16 tensors, block tables and a reference result. A harness that loads kernels with `torch.mps.load_metallib` and compares with torch on the CPU covers all of that, and it exercises the same launch path vLLM will use.

**Why does our Triton launcher take tensors instead of `data_ptr()`?**
On MPS, `data_ptr()` loses the `MTLBuffer` a Metal kernel needs. The tensor still has it, with the byte offset.

**Why ship one `applegpu-compile` binary instead of `applegpu-opt` and `applegpu-translate`?**
Each tool links LLVM statically, so shipping two would package LLVM twice.

**Why call the compiler as a subprocess from Triton?**
It keeps Triton's LLVM and ours in separate processes. The cost is a few milliseconds per kernel compile, which Triton caches.

**Why is the GPU test runner Objective-C++ and not Swift?**
CMake builds it with the same toolchain as the rest of the project, with no Swift dependency, and the Metal API is the same from either language.

## Build and tests

**Why put LLVM and Triton in `deps/` inside the repository instead of using submodules?**
Neither is needed by every clone: tier 1 tests and CI do not need Triton, and LLVM can come from an existing build. A submodule pins the source but not the build, which is the slow part. `utils/setup.sh` and `utils/pins.txt` pin both and set them up in one command.

**Why pin Triton 3.9 rather than 3.8?**
vLLM's kernels compile with 3.9, and the only API difference that matters (`tl.advance`) is unused by vLLM. 3.9 also has the plugin interfaces for out-of-tree layouts.

**Why Python 3.13?**
vllm-metal supports only Python 3.12 and 3.13.

**Why does CI cache the LLVM and Triton builds?**
Building LLVM takes hours and Triton about half an hour. Keying the cache on `utils/pins.txt` rebuilds them only when a pin changes.

**Why three test tiers?**
So every machine runs what it can and reports the rest as unsupported. Saved IR fixtures need nothing, regenerating them needs Triton, and end-to-end runs need a Mac GPU. GitHub's hosted runners cover the first two; contributors' Macs cover the third.
