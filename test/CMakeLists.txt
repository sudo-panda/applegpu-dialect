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
