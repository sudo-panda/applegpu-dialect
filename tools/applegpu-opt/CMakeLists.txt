set(LIBS
    MLIRArithDialect
    MLIRFuncDialect
    MLIRGPUDialect
    MLIRMemRefDialect
    MLIROptLib
    MLIRAppleGPU
  )
add_llvm_executable(applegpu-opt applegpu-opt.cpp)

llvm_update_compile_flags(applegpu-opt)
target_link_libraries(applegpu-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(applegpu-opt)
