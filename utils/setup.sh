#!/usr/bin/env bash
# Set up everything needed to build and test this repository, then run the
# tests. Dependencies go into deps/ and the Python venv into .venv (both
# ignored by git), at the versions in utils/pins.txt:
#
#   .venv                     Python venv: lit (requirements.txt), and Triton
#                             (editable) unless --no-triton
#   deps/triton               Triton (about 7 GB, ~30 min)
#   LLVM and MLIR             the prebuilt LLVM that Triton's TRITON_TAG uses
#                             (its cmake/llvm-info.json): Triton's own copy in
#                             ~/.triton/llvm when it is there, otherwise
#                             downloaded into deps/llvm-<hash>-<platform>-<build>
#   deps/vllm-metal           with --with-vllm: vllm-metal, installed
#                             (editable) into .venv with vLLM and torch
#   build/                    this project
#
# Every step is skipped or incremental when it is already done, so rerunning
# the script is fast. After changing utils/pins.txt, rerun it to update.
#
# Usage: utils/setup.sh [--no-triton] [--with-vllm] [--llvm-build DIR] [--jobs N]
#   --no-triton        skip Triton and build without Triton support: no Triton
#                      code is built and no Triton test is run (only Triton's
#                      llvm-info.json is fetched, to find the LLVM)
#   --with-vllm        also install PyTorch, vLLM and vllm-metal into .venv
#                      (macOS on Apple silicon, needs Triton and the Metal toolchain)
#   --llvm-build DIR   use an LLVM/MLIR build of your own instead; it must be
#                      built from the LLVM commit Triton uses, with MLIR
#   --jobs N           parallel jobs for the Triton build (default: its own).
#                      On a 16 GB Mac, use --jobs 4: Triton's default, twice
#                      the number of cores, can hang it.
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

TRITON_TAG=$(pin TRITON_TAG)
PYTHON_VERSION=$(pin PYTHON_VERSION)
if [[ $with_vllm == 1 ]]; then
  [[ $with_triton == 1 ]] || die "--with-vllm needs Triton; drop --no-triton"
  [[ $(uname -s)/$(uname -m) == Darwin/arm64 ]] || die "--with-vllm needs macOS on Apple silicon"
  VLLM_METAL_TAG=$(pin VLLM_METAL_TAG)
fi

for tool in git cmake ninja curl tar; do
  command -v "$tool" >/dev/null || die "'$tool' not found on PATH"
done

# ---------------------------------------------------------------- Python
python=${PYTHON:-python$PYTHON_VERSION}
command -v "$python" >/dev/null || die "'$python' not found; install Python $PYTHON_VERSION or set PYTHON"
found=$("$python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
[[ $found == "$PYTHON_VERSION" ]] || die "'$python' is Python $found, not $PYTHON_VERSION (set PYTHON)"
venv=$ROOT/.venv
py=$venv/bin/python
if [[ ! -x $py ]]; then
  step "Creating the venv in $venv with $python"
  "$python" -m venv "$venv"
fi
found=$("$py" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
[[ $found == "$PYTHON_VERSION" ]] || die "$venv uses Python $found, not $PYTHON_VERSION; delete .venv and deps/triton/build, then rerun"
step "Installing this project's Python packages (requirements.txt)"
"$venv/bin/pip" install -q -r "$ROOT/requirements.txt"

# ---------------------------------------------------------------- Triton
if [[ $with_triton == 1 ]]; then
  triton=$DEPS/triton
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

  step "Building Triton (about 30 minutes the first time)"
  "$venv/bin/pip" install -q -r "$triton/python/requirements.txt"
  # Without build isolation the build folder stays usable for incremental
  # rebuilds. MAX_JOBS is set only with --jobs. TRITON_EXT_ENABLED=1 exports
  # libtriton's symbols so that plugins (triton-plugin/) can load
  (cd "$triton" && env TRITON_EXT_ENABLED=1 ${jobs:+MAX_JOBS=$jobs} \
      "$venv/bin/pip" install -q -e . --no-build-isolation)
  llvm_info=$(cat "$triton/cmake/llvm-info.json")
else
  step "Skipping Triton (--no-triton): building without Triton support"
  llvm_info=$(curl -fsSL "https://raw.githubusercontent.com/triton-lang/triton/$TRITON_TAG/cmake/llvm-info.json") \
      || die "could not fetch Triton $TRITON_TAG's cmake/llvm-info.json"
fi

# ---------------------------------------------------------------- LLVM
# One LLVM: the one Triton's TRITON_TAG uses, named in its cmake/llvm-info.json
# (commit, build number and a checksum per platform), prebuilt by Triton's
# project, as Triton's own build downloads it.
llvm_hash=$("$py" -c 'import json, sys; print(json.loads(sys.argv[1])["llvm_hash"])' "$llvm_info")

# The commit an LLVM build was made from.
llvm_revision() {
  sed -n 's/^#define LLVM_REVISION R"(\(.*\))"$/\1/p' "$1/include/llvm/Support/VCSRevision.h" 2>/dev/null || true
}

if [[ -n $llvm_build ]]; then
  [[ -d $llvm_build ]] || die "no such directory: $llvm_build"
  llvm_build=$(cd "$llvm_build" && pwd)
  step "Using the LLVM build in $llvm_build"
else
  # Triton's names for the platforms it builds LLVM for (get_llvm_system_suffix
  # in its python/build_helpers.py).
  case $(uname -s)/$(uname -m) in
    Darwin/arm64) platform=macos-arm64 ;;
    Darwin/x86_64) platform=macos-x64 ;;
    Linux/aarch64|Linux/arm64)
      if grep -qs '^ID="almalinux"' /etc/os-release; then platform=almalinux-arm64; else platform=ubuntu-arm64; fi ;;
    Linux/x86_64)
      glibc=$(getconf GNU_LIBC_VERSION | sed 's/^glibc //')
      if [[ $(printf '%s\n' "$glibc" 2.29 | sort -V | head -1) == 2.29 ]]; then platform=ubuntu-x64; else platform=almalinux-x64; fi ;;
    *) die "Triton has no prebuilt LLVM for $(uname -s)/$(uname -m); build LLVM $llvm_hash with MLIR and pass --llvm-build" ;;
  esac
  read -r name sha256 < <("$py" -c '
import json, sys
info = json.loads(sys.argv[1])
print("llvm-%s-%s-%d" % (info["llvm_hash"][:8], sys.argv[2], info["build_number"]), info["sha256sum"][sys.argv[2]])' "$llvm_info" "$platform")
  cached=${TRITON_HOME:-$HOME}/.triton/llvm/$name
  if [[ -f $cached/lib/cmake/mlir/MLIRConfig.cmake ]]; then
    llvm_build=$cached
    step "Using Triton's copy of LLVM ${llvm_hash:0:8} in $llvm_build"
  else
    llvm_build=$DEPS/$name
    if [[ ! -f $llvm_build/lib/cmake/mlir/MLIRConfig.cmake ]]; then
      url=https://oaitriton.blob.core.windows.net/public/llvm-builds/$name.tar.gz
      step "Downloading LLVM ${llvm_hash:0:8} ($url)"
      mkdir -p "$DEPS"
      curl -fL --retry 3 -o "$DEPS/$name.tar.gz.part" "$url" || die "could not download $url"
      got=$("$py" -c 'import hashlib, sys; print(hashlib.file_digest(open(sys.argv[1], "rb"), "sha256").hexdigest())' "$DEPS/$name.tar.gz.part")
      [[ $got == "$sha256" ]] || die "$url: checksum $got, expected $sha256 (from Triton's llvm-info.json)"
      rm -rf "$llvm_build"
      tar -xzf "$DEPS/$name.tar.gz.part" -C "$DEPS"
      rm "$DEPS/$name.tar.gz.part"
      [[ -d $llvm_build ]] || die "$name.tar.gz did not unpack into $llvm_build"
    fi
    step "Using LLVM ${llvm_hash:0:8} in $llvm_build"
  fi
fi
[[ -f $llvm_build/lib/cmake/mlir/MLIRConfig.cmake ]] || die "$llvm_build has no MLIR (lib/cmake/mlir)"
for tool in FileCheck not; do
  [[ -x $llvm_build/bin/$tool ]] || die "$llvm_build has no bin/$tool, which the tests use"
done
revision=$(llvm_revision "$llvm_build")
[[ $revision == "$llvm_hash" ]] || die "$llvm_build is LLVM ${revision:-of unknown commit}, but Triton $TRITON_TAG uses $llvm_hash"

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
    -DLLVM_EXTERNAL_LIT="$venv/bin/lit" \
    -DMLIR_INCLUDE_TESTS=ON \
    -DAPPLEGPU_ENABLE_TRITON=$([[ $with_triton == 1 ]] && echo ON || echo OFF)
ninja -C "$ROOT/build" check-applegpu
