# Triton for test/Integration/Triton. Both are used as outside programs, never
# linked. When unset, they are looked for where utils/setup.sh puts them: the
# repository's venv (.venv) and the Triton build in deps/triton. If not found,
# the Triton tests are reported as unsupported.
set(APPLEGPU_TRITON_PYTHON "" CACHE FILEPATH "Python interpreter that can import triton")
set(APPLEGPU_TRITON_OPT "" CACHE FILEPATH "triton-opt from the same Triton build")

set(_applegpu_triton_dir "${APPLEGPU_SOURCE_DIR}/deps/triton")
if(NOT APPLEGPU_TRITON_PYTHON AND EXISTS "${APPLEGPU_SOURCE_DIR}/.venv/bin/python")
  set(APPLEGPU_TRITON_PYTHON "${APPLEGPU_SOURCE_DIR}/.venv/bin/python")
endif()
if(NOT APPLEGPU_TRITON_OPT)
  file(GLOB _applegpu_triton_opt "${_applegpu_triton_dir}/build/*/bin/triton-opt")
  if(_applegpu_triton_opt)
    list(GET _applegpu_triton_opt 0 APPLEGPU_TRITON_OPT)
  endif()
endif()
message(STATUS "Triton for tests: python='${APPLEGPU_TRITON_PYTHON}' triton-opt='${APPLEGPU_TRITON_OPT}'")

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

add_lit_testsuite(check-applegpu "Running the applegpu regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${APPLEGPU_TEST_DEPENDS}
        )
set_target_properties(check-applegpu PROPERTIES FOLDER "Tests")

add_lit_testsuites(APPLEGPU ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${APPLEGPU_TEST_DEPENDS})
