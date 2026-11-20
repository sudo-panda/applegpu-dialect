"""Print the TTIR of one @triton.jit kernel, as Triton's own backend sees it.

Usage:
  triton_to_ttir.py <kernel.py> <kernel-name>
      --signature x_ptr=*fp32,n=i32 [--constexpr BLOCK=128]

The output is Triton's custom assembly form with locations. Pipe it through
`triton-opt --mlir-print-op-generic` to get the generic form that applegpu-opt
reads with --allow-unregistered-dialect.

Triton has no Apple backend, so the NVIDIA backend stands in as the target.
Only its TTIR stage runs (inlining, canonicalization, combine, CSE), which is
target independent. TTIR does not record num_warps, so how many threads run
one program is decided later, by the conversion pass. Nothing here needs CUDA
or a GPU.
"""

import argparse
import importlib.util
import os
import sys

# FP fusion only matters in later stages; setting it silences a warning.
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

import triton
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton.compiler import ASTSource

# Placeholder target. Below sm_90 the TTIR stage also rewrites tensor
# descriptors (TMA) into plain pointer arithmetic, which suits Apple GPUs.
TARGET = GPUTarget("cuda", 80, 32)


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
    try:
        return int(value)
    except ValueError:
        return float(value)


def load_kernel(path, name):
    spec = importlib.util.spec_from_file_location("ttir_kernel_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    kernel = getattr(module, name, None)
    if not isinstance(kernel, triton.runtime.JITFunction):
        sys.exit(f"error: '{name}' in {path} is not a @triton.jit kernel")
    return kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file")
    parser.add_argument("kernel")
    parser.add_argument("--signature", required=True,
                        help="runtime arguments, e.g. x_ptr=*fp32,n=i32")
    parser.add_argument("--constexpr", default="",
                        help="constexpr arguments, e.g. BLOCK=128")
    args = parser.parse_args()

    kernel = load_kernel(args.file, args.kernel)
    signature = parse_pairs(args.signature)
    constexprs = {k: parse_constant(v) for k, v in parse_pairs(args.constexpr).items()}
    # ASTSource wants every parameter in the signature; constexprs are typed
    # "constexpr" there and their values passed separately.
    signature.update({name: "constexpr" for name in constexprs})

    backend = CUDABackend(TARGET)
    options = backend.parse_options({})
    source = ASTSource(kernel, signature, constexprs)

    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = source.make_ir(TARGET, options, backend.get_codegen_implementation(options),
                            backend.get_module_map(), context)

    stages = {}
    backend.add_stages(stages, options, source.language)
    module = stages["ttir"](module, {})
    print(module.str())


if __name__ == "__main__":
    main()
