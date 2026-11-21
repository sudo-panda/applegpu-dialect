"""Print the TTGIR of one @triton.jit kernel, compiled for an Apple GPU.

Usage:
  triton_to_ttgir.py <kernel.py> <kernel-name>
      --signature x_ptr=*fp32,n=i32 [--constexpr BLOCK=128,stride=1]
      [--divisible n]
      [--num-warps 4] [--target apple:m1] [--plugin PATH]
      [--triton-opt PATH]

It runs Triton's TTIR passes and its target-independent TTGIR passes (the list
in AGENTS.md, "Design rules"), with 32 threads per warp. The output is the
generic form that applegpu-opt reads with --allow-unregistered-dialect,
printed by triton-opt with --mlir-print-op-generic --mlir-print-local-scope
--mlir-print-debuginfo.

Arguments are specialized as Triton's launcher does:
- Pointer arguments get tt.divisibility = 16 (PyTorch allocations are aligned).
- Integer arguments given a value with --constexpr become constants; pass the
  ones that are 1 at run time (unit strides), as the launcher does.
- Integer arguments named in --divisible get tt.divisibility = 16; name the
  ones that are multiples of 16 at run time, as the launcher does.
- Parameters left out take their default, when it is None or the parameter
  is a tl.constexpr (Triton specializes both to constants).

Constants are numbers, True, False or None (a pointer argument passed as None).

--plugin loads a Triton plugin (dialects and passes); its passes run right
after optimize_thread_locality.

--vllm is for kernels in vLLM's sources. vLLM's macOS build turns Triton off
(its kernels become placeholders) unless Triton has a backend named "cpu", so
an inactive one is registered in this process before the kernel is loaded.
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys

# FP fusion only matters in later stages; setting it silences a warning.
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

import triton
from triton._C.libtriton import ir, passes
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton.compiler import ASTSource

# Only for Triton's front end; no NVIDIA pass runs.
FRONTEND_TARGET = GPUTarget("cuda", 80, 32)
THREADS_PER_WARP = 32
PRINT_FLAGS = ["--mlir-print-op-generic", "--mlir-print-local-scope", "--mlir-print-debuginfo"]


def parse_pairs(text):
    pairs = {}
    if not text:
        return pairs
    for item in text.split(","):
        name, sep, value = item.partition("=")
        if not sep:
            sys.exit(f"error: expected name=value, got '{item}'")
        pairs[name.strip()] = value.strip()
    return pairs


def parse_constant(value):
    named = {"None": None, "True": True, "False": False}
    if value in named:
        return named[value]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        sys.exit(f"error: constant '{value}' is not None, True, False or a number")


class InactiveDriver:
    @staticmethod
    def is_active():
        return False


def keep_vllm_triton_on():
    from triton.backends import Backend, backends
    backends.setdefault("cpu", Backend(compiler=None, driver=InactiveDriver))


def load_kernel(path, name):
    spec = importlib.util.spec_from_file_location("ttgir_kernel_module", path)
    if spec is None or spec.loader is None:
        sys.exit(f"error: cannot load {path} as a Python module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    kernel = getattr(module, name, None)
    while kernel is not None and not isinstance(kernel, triton.runtime.JITFunction):
        kernel = getattr(kernel, "fn", None)  # @triton.autotune, @triton.heuristics
    if kernel is None:
        sys.exit(f"error: '{name}' in {path} is not a @triton.jit kernel")
    return kernel


def make_ttir(pm):
    passes.common.add_inliner(pm)
    passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_combine(pm)
    passes.ttir.add_reorder_broadcast(pm)
    passes.common.add_cse(pm)
    passes.common.add_symbol_dce(pm)
    passes.ttir.add_loop_unroll(pm)


def make_ttgir(pm, target, num_warps, plugin_passes):
    common, ttir, ttgpuir = passes.common, passes.ttir, passes.ttgpuir
    ttir.add_convert_to_ttgpuir(pm, target, num_warps, THREADS_PER_WARP, 1)
    ttgpuir.add_coalesce(pm)
    ttgpuir.add_f32_dot_tc(pm, False)
    ttgpuir.add_remove_layout_conversions(pm)
    ttgpuir.add_optimize_thread_locality(pm)
    for add_pass in plugin_passes:
        add_pass(pm)
    ttgpuir.add_remove_layout_conversions(pm)
    ttgpuir.add_optimize_dot_operands(pm, True)
    ttir.add_loop_aware_cse(pm)
    ttgpuir.add_fuse_nested_loops(pm)
    common.add_canonicalizer(pm)
    ttir.add_triton_licm(pm)
    common.add_canonicalizer(pm)
    ttgpuir.add_combine_tensor_select_and_if(pm)
    common.add_canonicalizer(pm)
    ttir.add_loop_aware_cse(pm)
    ttgpuir.add_optimize_dot_operands(pm, True)
    ttgpuir.add_remove_layout_conversions(pm)
    ttgpuir.add_reduce_data_duplication(pm)
    ttgpuir.add_reorder_instructions(pm)
    ttir.add_loop_aware_cse(pm)
    common.add_symbol_dce(pm)
    common.add_sccp(pm)
    common.add_cse(pm)
    common.add_canonicalizer(pm)


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    parser.add_argument("file")
    parser.add_argument("kernel")
    parser.add_argument("--signature", required=True,
                        help="runtime arguments, e.g. x_ptr=*fp32,n=i32")
    parser.add_argument("--constexpr", default="",
                        help="constant arguments, e.g. BLOCK=128,stride=1")
    parser.add_argument("--divisible", default="",
                        help="integer arguments that are multiples of 16, e.g. M,N,K")
    parser.add_argument("--num-warps", type=int, default=4,
                        help="SIMD-groups per program (default 4)")
    parser.add_argument("--target", default="apple:m1",
                        help="ttg.target of the module (default apple:m1)")
    parser.add_argument("--plugin", help="Triton plugin library to load")
    parser.add_argument("--vllm", action="store_true",
                        help="the kernel is in vLLM's sources (keeps vLLM's Triton on)")
    parser.add_argument("--triton-opt", default=shutil.which("triton-opt"),
                        help="triton-opt from the same Triton (default: on PATH)")
    args = parser.parse_args()
    if not args.triton_opt:
        sys.exit("error: triton-opt not found; pass --triton-opt")

    plugin_passes = []
    if args.plugin:
        before = set(dir(passes.plugin))
        ir.extend_dialects_with(args.plugin)
        passes.plugin.extend_with(args.plugin)
        plugin_passes = [getattr(passes.plugin, name) for name in sorted(set(dir(passes.plugin)) - before)
                         if name.startswith("add_")]

    if args.vllm:
        keep_vllm_triton_on()
    kernel = load_kernel(args.file, args.kernel)
    given = parse_pairs(args.signature)
    constexprs = {k: parse_constant(v) for k, v in parse_pairs(args.constexpr).items()}
    divisible = {n.strip() for n in args.divisible.split(",") if n.strip()}
    unknown = [n for n in set(given) | set(constexprs) | divisible if n not in kernel.arg_names]
    if unknown:
        sys.exit(f"error: {args.kernel} has no argument {', '.join(sorted(unknown))}")
    # ASTSource wants every parameter, in order; constants are typed
    # "constexpr" and their values passed separately.
    signature = {}
    for param in kernel.params:
        name = param.name
        if (name not in constexprs and name not in given and param.has_default
                and (param.is_constexpr or param.default is None)):
            constexprs[name] = param.default  # as when a caller leaves it out
        if name in constexprs:
            signature[name] = "constexpr"
        elif name in given:
            signature[name] = given[name]
        else:
            sys.exit(f"error: argument '{name}' is in neither --signature nor --constexpr")
    for name in divisible:
        if not signature[name].startswith("i"):
            sys.exit(f"error: --divisible '{name}' is not an integer argument")
    attrs = {(i,): [["tt.divisibility", 16]] for i, name in enumerate(kernel.arg_names)
             if signature[name].startswith("*") or name in divisible}

    backend = CUDABackend(FRONTEND_TARGET)
    options = backend.parse_options({"num_warps": args.num_warps})
    source = ASTSource(kernel, signature, constexprs, attrs)

    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = source.make_ir(FRONTEND_TARGET, options, backend.get_codegen_implementation(options),
                            backend.get_module_map(), context)

    pm = ir.pass_manager(context)
    pm.enable_debug()
    make_ttir(pm)
    make_ttgir(pm, args.target, args.num_warps, plugin_passes)
    pm.run(module, "ttgir")

    env = dict(os.environ)
    if args.plugin:  # so that triton-opt can parse the plugin's attributes
        env["TRITON_PLUGIN_PATHS"] = os.pathsep.join(
            p for p in (args.plugin, env.get("TRITON_PLUGIN_PATHS")) if p)
    result = subprocess.run([args.triton_opt, *PRINT_FLAGS, "-"], input=module.str(),
                            capture_output=True, text=True, env=env)
    if result.returncode != 0:
        sys.exit(f"error: {args.triton_opt} failed:\n{result.stderr}")
    sys.stdout.write(result.stdout)


if __name__ == "__main__":
    main()
