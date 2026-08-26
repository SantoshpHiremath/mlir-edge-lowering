cmake_minimum_required(VERSION 3.20)
project(edge-opt LANGUAGES C CXX)

find_package(MLIR REQUIRED CONFIG PATHS /usr/lib/llvm-18/lib/cmake/mlir NO_DEFAULT_PATH)
find_package(LLVM REQUIRED CONFIG PATHS /usr/lib/llvm-18/lib/cmake/llvm NO_DEFAULT_PATH)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_executable(edge-opt
  lib/WorkloadAnalysisPass.cpp
  lib/edge-opt-main.cpp
)

llvm_update_compile_flags(edge-opt)

target_link_libraries(edge-opt PRIVATE
  MLIROptLib
  MLIRLinalgDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRTensorDialect
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
