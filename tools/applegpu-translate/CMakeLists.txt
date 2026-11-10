set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_executable(applegpu-translate
  applegpu-translate.cpp
  )
llvm_update_compile_flags(applegpu-translate)
target_link_libraries(applegpu-translate
  PRIVATE
  MLIRAppleGPUTargetMSL
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRTranslateLib
  )

mlir_check_link_libraries(applegpu-translate)
