# -*- Python -*-

import os
import subprocess

import lit.formats

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "APPLEGPU"

config.test_format = lit.formats.ShTest()

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.applegpu_obj_root, "test")

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

config.applegpu_tools_dir = os.path.join(config.applegpu_obj_root, "bin")

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.applegpu_tools_dir, config.llvm_tools_dir]
tools = [
    "applegpu-opt",
    "applegpu-translate",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

# GPU runner built from test/runners/run_grid.mm (macOS only).
config.substitutions.append(
    ("%run_grid", '"%s"' % os.path.join(config.test_exec_root, "runners", "run-grid")))

# The Metal toolchain, for tests that compile and run kernels on the GPU.
try:
    subprocess.run(["xcrun", "-sdk", "macosx", "metal", "--version"], check=True,
                   capture_output=True, timeout=60)
    config.available_features.add("metal")
except Exception:
    pass


# Triton is never linked into this project. Its tests run Triton's own Python
# and triton-opt as outside programs, and are unsupported without them.
def triton_available(python, triton_opt):
    if not (os.path.isfile(python) and os.path.isfile(triton_opt)):
        return False
    try:
        subprocess.run([python, "-c", "import triton"], check=True,
                       capture_output=True, timeout=120)
        return True
    except Exception:
        return False


if triton_available(config.triton_python, config.triton_opt):
    config.available_features.add("triton")
    config.substitutions.append(("%triton_python", '"%s"' % config.triton_python))
    config.substitutions.append(("%triton_opt", '"%s"' % config.triton_opt))
    config.substitutions.append(("%triton_tensor_layout", '"%s"' % config.triton_tensor_layout))
    config.substitutions.append(
        ("%triton_to_ttgir", '"%s" "%s" --triton-opt "%s"' % (
            config.triton_python,
            os.path.join(config.test_source_root, "runners", "triton_to_ttgir.py"),
            config.triton_opt)))

    # The Apple Triton plugin (triton-plugin/), built with the tests.
    if config.apple_triton_plugin:
        config.available_features.add("apple-triton-plugin")
        config.substitutions.append(("%apple_triton_plugin", '"%s"' % config.apple_triton_plugin))

    # vLLM's sources, for its Triton kernels (utils/setup.sh --with-vllm).
    result = subprocess.run(
        [config.triton_python, "-c",
         "import importlib.util, os; s = importlib.util.find_spec('vllm'); "
         "print(os.path.dirname(s.origin) if s else '')"],
        capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        config.available_features.add("vllm")
        config.substitutions.append(("%vllm_dir", '"%s"' % result.stdout.strip()))
