//===- edge-opt-main.cpp - Custom MLIR opt tool with our pass --------===//
//
// A small, real out-of-tree "opt" tool (mirrors the standard MLIR
// pattern of building a custom mlir-opt with your own passes
// registered alongside the standard dialects/passes) that registers
// WorkloadAnalysisPass so it's runnable as
// `edge-opt --edge-workload-analysis input.mlir`.
//
//===--------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

namespace mlir {
void registerWorkloadAnalysisPass();
}

// Deliberately registers only the dialects this tool's input actually
// uses (func/arith/linalg/tensor), rather than mlir::registerAllDialects.
// Registering everything requires linking every dialect library in the
// distro's split-lib packaging, which pulls in unrelated targets
// (TOSA pipelines, GPU/SPIR-V, etc.) this tool has no use for — scoping
// down to what's needed keeps the build honest about what edge-opt
// actually supports.
int main(int argc, char **argv) {
  mlir::registerWorkloadAnalysisPass();

  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                   mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "edge-opt: custom MLIR tool with edge AI workload analysis pass\n", registry));
}
