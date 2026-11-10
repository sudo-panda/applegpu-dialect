add_mlir_translation_library(MLIRAppleGPUTargetMSL
  TranslateToMSL.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/applegpu/Target/MSL

  LINK_LIBS PUBLIC
  MLIRAppleGPU
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRSupport
  MLIRTranslateLib
  )
