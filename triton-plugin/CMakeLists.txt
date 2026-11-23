# The Apple Triton plugin: a shared library that Triton loads into its own
# process. It is built against Triton's headers and Triton's LLVM, never ours,
# and never linked into applegpu-opt. Every setting that must match Triton
# (source folder, LLVM, compiler, version) is read from Triton's CMake cache.
#
#   cmake -G Ninja -S triton-plugin -B build/triton-plugin \
#       -DTRITON_BUILD_DIR=deps/triton/build/cmake.<platform>-cpython-<version>
#
# The top-level CMakeLists.txt does this when it finds Triton.

cmake_minimum_required(VERSION 3.20)

set(TRITON_BUILD_DIR "" CACHE PATH "Triton's CMake build folder, holding CMakeCache.txt")
if(NOT EXISTS "${TRITON_BUILD_DIR}/CMakeCache.txt")
  message(FATAL_ERROR "TRITON_BUILD_DIR='${TRITON_BUILD_DIR}' has no CMakeCache.txt")
endif()
load_cache("${TRITON_BUILD_DIR}" READ_WITH_PREFIX TRITON_
  triton_SOURCE_DIR LLVM_DIR CMAKE_C_COMPILER CMAKE_CXX_COMPILER TRITON_VERSION)

# The same compiler as Triton, so that the C++ ABI matches.
if(NOT CMAKE_C_COMPILER)
  set(CMAKE_C_COMPILER "${TRITON_CMAKE_C_COMPILER}")
endif()
if(NOT CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER "${TRITON_CMAKE_CXX_COMPILER}")
endif()

# C is enabled only because LLVM's CMake helpers check linker flags with it.
project(AppleTritonPlugin LANGUAGES C CXX)

find_package(MLIR REQUIRED CONFIG
  PATHS "${TRITON_LLVM_DIR}/../mlir" NO_DEFAULT_PATH)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(TableGen)
include(AddMLIR)

set(TRITON_SRC "${TRITON_triton_SOURCE_DIR}")
message(STATUS "Apple Triton plugin for Triton ${TRITON_TRITON_VERSION} (${TRITON_SRC})")

# Triton's settings: C++20, no RTTI or exceptions, assertions on (Triton and
# its LLVM are built with them).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${TRITON_SRC}/include
  ${TRITON_BUILD_DIR}/include
  ${TRITON_SRC})  # triton/Tools/PluginUtils.h includes python/src/ir.h
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})

set(LLVM_TARGET_DEFINITIONS AppleTT.td)
mlir_tablegen(AppleTTDialect.h.inc -gen-dialect-decls -dialect=apple_tt)
mlir_tablegen(AppleTTDialect.cpp.inc -gen-dialect-defs -dialect=apple_tt)
add_public_tablegen_target(AppleTTIncGen)

add_library(AppleTritonPlugin SHARED
  AppleTTDialect.cpp
  Plugin.cpp)
add_dependencies(AppleTritonPlugin AppleTTIncGen)
target_compile_options(AppleTritonPlugin PRIVATE -O2 -fno-exceptions -fno-rtti)
target_compile_definitions(AppleTritonPlugin PRIVATE
  TRITON_EXT_ENABLED=1
  TRITON_VERSION="${TRITON_TRITON_VERSION}")
# Every Triton, MLIR and LLVM symbol comes from the process that loads the
# plugin (libtriton or triton-opt).
if(APPLE)
  target_link_options(AppleTritonPlugin PRIVATE -undefined dynamic_lookup)
endif()
