# Triton for the Triton tests, used as outside programs, never linked. When
# unset, they are looked for where utils/setup.sh puts them: the repository's
# venv (.venv) and the Triton build (APPLEGPU_TRITON_BUILD_DIR, from the
# top-level CMakeLists.txt). If not found, the Triton tests are unsupported.
set(APPLEGPU_TRITON_PYTHON "" CACHE FILEPATH "Python interpreter that can import triton")
set(APPLEGPU_TRITON_OPT "" CACHE FILEPATH "triton-opt from the same Triton build")

if(NOT APPLEGPU_TRITON_PYTHON AND EXISTS "${APPLEGPU_SOURCE_DIR}/.venv/bin/python")
  set(APPLEGPU_TRITON_PYTHON "${APPLEGPU_SOURCE_DIR}/.venv/bin/python")
endif()
if(NOT APPLEGPU_TRITON_OPT AND EXISTS "${APPLEGPU_TRITON_BUILD_DIR}/bin/triton-opt")
  set(APPLEGPU_TRITON_OPT "${APPLEGPU_TRITON_BUILD_DIR}/bin/triton-opt")
endif()
get_filename_component(_applegpu_triton_bin "${APPLEGPU_TRITON_OPT}" DIRECTORY)
set(APPLEGPU_TRITON_TENSOR_LAYOUT "${_applegpu_triton_bin}/triton-tensor-layout")
message(STATUS "Triton for tests: python='${APPLEGPU_TRITON_PYTHON}' triton-opt='${APPLEGPU_TRITON_OPT}' plugin='${APPLEGPU_TRITON_PLUGIN}'")

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(APPLEGPU_TEST_DEPENDS
        FileCheck
        applegpu-opt
        applegpu-translate
        )

# GPU runner for test/Integration, used as %run_grid. Test helper only: it is
# built into the test tree, not installed, and only on macOS.
if(APPLE)
  enable_language(OBJCXX)
  add_executable(run-grid runners/run_grid.mm)
  set_target_properties(run-grid PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/runners
    OBJCXX_STANDARD 17
    FOLDER "Tests")
  target_compile_options(run-grid PRIVATE -fobjc-arc)
  target_link_libraries(run-grid PRIVATE "-framework Foundation" "-framework Metal")
  list(APPEND APPLEGPU_TEST_DEPENDS run-grid)
endif()

if(TARGET apple-triton-plugin)
  list(APPEND APPLEGPU_TEST_DEPENDS apple-triton-plugin)
endif()

add_lit_testsuite(check-applegpu "Running the applegpu regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${APPLEGPU_TEST_DEPENDS}
        )
set_target_properties(check-applegpu PROPERTIES FOLDER "Tests")

add_lit_testsuites(APPLEGPU ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${APPLEGPU_TEST_DEPENDS})
