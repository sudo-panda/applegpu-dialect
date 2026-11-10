add_mlir_dialect_library(MLIRAppleGPU
        AppleGPUDialect.cpp
        AppleGPUOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/applegpu/Dialect/AppleGPU/IR

        DEPENDS
        MLIRAppleGPUOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRGPUDialect
        )
