#!/usr/bin/env bash
# Set up everything needed to build and test this repository, then run the
# tests. Dependencies go into deps/ and the Python venv into .venv (both
# ignored by git), at the versions in utils/pins.txt:
#
#   deps/llvm-project, deps/llvm-build   LLVM and MLIR (about 12 GB, an hour or more)
#   deps/triton                          Triton (about 7 GB, ~30 min)
#   .venv                                Python venv with Triton installed (editable)
#   deps/vllm-metal                      with --with-vllm: vllm-metal, installed
#                                        (editable) into .venv with vLLM and torch
#   build/                               this project
#
# Every step is skipped or incremental when it is already done, so rerunning
# the script is fast. After changing utils/pins.txt, rerun it to update.
#
# Usage: utils/setup.sh [--no-triton] [--with-vllm] [--llvm-build DIR] [--jobs N]
#   --no-triton        skip Triton; its tests are then reported as unsupported
#   --with-vllm        also install PyTorch, vLLM and vllm-metal into .venv
#                      (macOS on Apple silicon, needs Triton and the Metal toolchain)
#   --llvm-build DIR   use an existing LLVM/MLIR build instead of deps/llvm-build
#   --jobs N           parallel jobs for the LLVM and Triton builds (default:
#                      each build's own). On a 16 GB Mac, use --jobs 4: Triton's
#                      default, twice the number of cores, can hang it.
#
# Python: set PYTHON to the interpreter for the venv; the default is
# python<PYTHON_VERSION> from utils/pins.txt, found on PATH.

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEPS=$ROOT/deps

die() { echo "error: $*" >&2; exit 1; }
step() { printf '\n==> %s\n' "$*"; }

pin() {
  local value
  value=$(sed -n "s/^$1=//p" "$ROOT/utils/pins.txt")
  [[ -n $value ]] || die "$1 is missing from utils/pins.txt"
  echo "$value"
}

with_triton=1
with_vllm=0
llvm_build=""
jobs=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-triton) with_triton=0 ;;
    --with-vllm) with_vllm=1 ;;
    --llvm-build) [[ $# -ge 2 ]] || die "--llvm-build needs a directory"; llvm_build=$2; shift ;;
    --jobs) [[ ${2:-} =~ ^[1-9][0-9]*$ ]] || die "--jobs needs a positive number"; jobs=$2; shift ;;
    -h|--help) sed -n '2,/^$/s/^# \{0,1\}//p' "$0"; exit 0 ;;
    *) die "unknown option '$1' (see --help)" ;;
  esac
  shift
done

LLVM_COMMIT=$(pin LLVM_COMMIT)
TRITON_TAG=$(pin TRITON_TAG)
PYTHON_VERSION=$(pin PYTHON_VERSION)
if [[ $with_vllm == 1 ]]; then
  [[ $with_triton == 1 ]] || die "--with-vllm needs Triton; drop --no-triton"
  [[ $(uname -s)/$(uname -m) == Darwin/arm64 ]] || die "--with-vllm needs macOS on Apple silicon"
  VLLM_METAL_TAG=$(pin VLLM_METAL_TAG)
fi

for tool in git cmake ninja; do
  command -v "$tool" >/dev/null || die "'$tool' not found on PATH"
done

# ---------------------------------------------------------------- LLVM
if [[ -n $llvm_build ]]; then
  [[ -d $llvm_build ]] || die "no such directory: $llvm_build"
  llvm_build=$(cd "$llvm_build" && pwd)
  step "Using the existing LLVM build in $llvm_build"
else
  llvm_src=$DEPS/llvm-project
  llvm_build=$DEPS/llvm-build
  if [[ ! -d $llvm_src/.git ]]; then
    step "Fetching LLVM $LLVM_COMMIT (only that commit)"
    git init -q "$llvm_src"
    git -C "$llvm_src" remote add origin https://github.com/llvm/llvm-project.git
  fi
  if [[ $(git -C "$llvm_src" rev-parse -q --verify HEAD || true) != "$LLVM_COMMIT" ]]; then
    git -C "$llvm_src" fetch --depth 1 origin "$LLVM_COMMIT"
    git -C "$llvm_src" -c advice.detachedHead=false checkout -q FETCH_HEAD
  fi
  step "Building LLVM and MLIR in $llvm_build (an hour or more the first time)"
  # LLVM_PARALLEL_LINK_JOBS=2: linking many tools at once can stall a 16 GB Mac.
  cmake -G Ninja -S "$llvm_src/llvm" -B "$llvm_build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_PROJECTS=mlir \
      -DLLVM_TARGETS_TO_BUILD=host \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DLLVM_PARALLEL_LINK_JOBS=2
  ninja -C "$llvm_build" ${jobs:+-j "$jobs"}
fi
[[ -f $llvm_build/lib/cmake/mlir/MLIRConfig.cmake ]] || die "$llvm_build has no MLIR build (lib/cmake/mlir)"
[[ -x $llvm_build/bin/llvm-lit ]] || die "$llvm_build has no bin/llvm-lit"

# ---------------------------------------------------------------- Triton
if [[ $with_triton == 1 ]]; then
  triton=$DEPS/triton
  python=${PYTHON:-python$PYTHON_VERSION}
  command -v "$python" >/dev/null || die "'$python' not found; install Python $PYTHON_VERSION or set PYTHON (or use --no-triton)"
  found=$("$python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  [[ $found == "$PYTHON_VERSION" ]] || die "'$python' is Python $found; Triton needs $PYTHON_VERSION (set PYTHON)"

  if [[ ! -d $triton/.git ]]; then
    step "Fetching Triton $TRITON_TAG"
    git clone -q --depth 1 --branch "$TRITON_TAG" https://github.com/triton-lang/triton.git "$triton"
  fi
  want=$(git -C "$triton" rev-parse -q --verify "refs/tags/$TRITON_TAG^{commit}" || true)
  if [[ -z $want || $(git -C "$triton" rev-parse HEAD) != "$want" ]]; then
    step "Moving Triton to $TRITON_TAG"
    git -C "$triton" fetch -q --depth 1 origin tag "$TRITON_TAG"
    git -C "$triton" -c advice.detachedHead=false checkout -q "refs/tags/$TRITON_TAG"
  fi

  venv=$ROOT/.venv
  if [[ ! -x $venv/bin/python ]]; then
    step "Creating the venv in $venv with $python"
    "$python" -m venv "$venv"
  fi
  found=$("$venv/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  [[ $found == "$PYTHON_VERSION" ]] || die "$venv uses Python $found, not $PYTHON_VERSION; delete .venv and deps/triton/build, then rerun"

  step "Building Triton (about 30 minutes the first time)"
  "$venv/bin/pip" install -q -r "$triton/python/requirements.txt"
  # Without build isolation the build folder stays usable for incremental
  # rebuilds. MAX_JOBS is set only with --jobs. TRITON_EXT_ENABLED=1 exports
  # libtriton's symbols so that plugins (triton-plugin/) can load
  (cd "$triton" && env TRITON_EXT_ENABLED=1 ${jobs:+MAX_JOBS=$jobs} \
      "$venv/bin/pip" install -q -e . --no-build-isolation)
else
  step "Skipping Triton (--no-triton): the Triton tests will be unsupported"
fi

# ---------------------------------------------------------------- vLLM
# vllm-metal publishes ready-built packages for only some Python versions, so
# we install it from its sources in deps/vllm-metal, which works with any
# version its pyproject.toml allows, the way its own install.sh does for a
# source checkout:
#   1. vLLM, from the package vLLM builds for macOS;
#   2. vllm-metal, editable, so Python uses deps/vllm-metal directly;
#   3. the nightly torch vllm-metal needs, which replaces vLLM's torch;
#   4. vllm-metal's Metal kernels and C++ extensions, built on this machine.
# Only VLLM_METAL_TAG is pinned here. vllm-metal names everything else: the
# vLLM release in .github/vllm-release-tag.commit, torch in
# vllm_metal/requirements-torch.txt, the rest in pyproject.toml.
if [[ $with_vllm == 1 ]]; then
  vllm_metal=$DEPS/vllm-metal
  py=$venv/bin/python
  if [[ ! -d $vllm_metal/.git ]]; then
    step "Fetching vllm-metal $VLLM_METAL_TAG"
    git clone -q --depth 1 --branch "$VLLM_METAL_TAG" https://github.com/vllm-project/vllm-metal.git "$vllm_metal"
  fi
  want=$(git -C "$vllm_metal" rev-parse -q --verify "refs/tags/$VLLM_METAL_TAG^{commit}" || true)
  if [[ -z $want || $(git -C "$vllm_metal" rev-parse HEAD) != "$want" ]]; then
    step "Moving vllm-metal to $VLLM_METAL_TAG"
    git -C "$vllm_metal" fetch -q --depth 1 origin tag "$VLLM_METAL_TAG"
    git -C "$vllm_metal" -c advice.detachedHead=false checkout -q "refs/tags/$VLLM_METAL_TAG"
  fi
  vllm_tag=$(tr -d '[:space:]' < "$vllm_metal/.github/vllm-release-tag.commit")
  have=$("$py" -c 'from importlib.metadata import version; print(version("vllm").split("+")[0])' 2>/dev/null || true)
  if [[ v$have != "$vllm_tag" ]]; then
    step "Installing vLLM $vllm_tag (its macOS package)"
    # The macOS arm64 wheel attached to that vLLM release on GitHub.
    wheel=$(curl -fsSL "https://api.github.com/repos/vllm-project/vllm/releases/tags/$vllm_tag" | "$py" -c '
import json, sys
urls = [a["browser_download_url"] for a in json.load(sys.stdin)["assets"]
        if a["name"].endswith(".whl") and "macosx" in a["name"] and "arm64" in a["name"]]
print(urls[0] if urls else "")') || die "could not read vLLM release $vllm_tag from GitHub"
    [[ -n $wheel ]] || die "vLLM release $vllm_tag has no macOS arm64 wheel"
    "$venv/bin/pip" install -q "$wheel"
  fi

  step "Installing vllm-metal $VLLM_METAL_TAG and its torch"
  # vllm-metal replaces vLLM's torch with a nightly, so pip warns that vLLM's
  # torch requirement is not met. That is expected.
  "$venv/bin/pip" install -q -e "$vllm_metal[build]"
  "$venv/bin/pip" install -q -r "$vllm_metal/vllm_metal/requirements-torch.txt"

  step "Building vllm-metal's Metal kernels"
  # Apple's clang, not whichever clang is first on PATH: torch builds the C++
  # extensions with $CC/$CXX or `clang`, and another clang's own libc++ headers
  # may not match the macOS SDK (with a Homebrew LLVM, <complex> can fail with
  # "undeclared identifier 'INFINITY'"). /usr/bin/clang finds the
  # SDK itself; the toolchain binary from `xcrun -f clang` does not.
  (cd "$vllm_metal" && export CC=/usr/bin/clang CXX=/usr/bin/clang++ \
      && "$py" -c 'from vllm_metal.metal.build import require_nax_sdk; require_nax_sdk()' \
      && "$py" -m vllm_metal.metal.build)

  "$py" -c 'import triton, sys; sys.exit(0 if triton.__path__[0].startswith(sys.argv[1]) else "error: installing vLLM replaced deps/triton in .venv")' "$DEPS/triton/" \
      || die "rerun utils/setup.sh without --with-vllm to reinstall Triton from deps/triton"
  "$py" -c 'import torch, vllm, vllm_metal; assert torch.backends.mps.is_available(), "MPS not available"; print("torch", torch.__version__, "vllm", vllm.__version__, "MPS available")'
fi

# ---------------------------------------------------------------- this project
step "Building this project in $ROOT/build and running the tests"
cmake -G Ninja -S "$ROOT" -B "$ROOT/build" \
    -DMLIR_DIR="$llvm_build/lib/cmake/mlir" \
    -DLLVM_EXTERNAL_LIT="$llvm_build/bin/llvm-lit" \
    -DMLIR_INCLUDE_TESTS=ON
ninja -C "$ROOT/build" check-applegpu
