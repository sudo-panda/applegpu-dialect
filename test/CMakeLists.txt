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

add_lit_testsuite(check-applegpu "Running the applegpu regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${APPLEGPU_TEST_DEPENDS}
        )
set_target_properties(check-applegpu PROPERTIES FOLDER "Tests")

add_lit_testsuites(APPLEGPU ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${APPLEGPU_TEST_DEPENDS})
