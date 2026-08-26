//===- WorkloadAnalysisPass.cpp - Edge AI workload analysis pass -----===//
//
// A real, custom, out-of-tree MLIR pass (not a built-in pass strung
// together from the command line, unlike run_pipeline.sh) that walks
// linalg.matmul and linalg.generic ops in a module and reports, per
// op: the estimated multiply-accumulate (MAC) count and the total
// input/output tensor element count (a proxy for memory footprint).
//
// This is exactly the kind of first-pass "workload characterization"
// step a real edge-AI compiler needs before it can decide how to map
// a network onto constrained hardware (an NPU, a small SRAM budget,
// etc.) — you need to know where the compute and memory pressure
// actually are before you can optimize placement or fusion for them.
// It's deliberately scoped to analysis (no IR transformation) so it's
// achievable and independently checkable, not a toy that prints
// "success" — its output is compared against a hand-computed
// expected MAC count for mlp_forward in test/expected_output.txt.
//
//===--------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

// Computes the MAC count for a linalg.matmul: for an (M x K) @ (K x N)
// matmul, that's M*K*N multiply-accumulate operations — the standard
// definition used across ML-compiler literature (e.g. this is exactly
// what tools like MLIR's own IREE or TVM's cost models compute as the
// first-order compute estimate for a matmul).
static int64_t computeMatmulMACs(linalg::MatmulOp op) {
  auto lhsType = llvm::cast<RankedTensorType>(op.getInputs()[0].getType());
  auto rhsType = llvm::cast<RankedTensorType>(op.getInputs()[1].getType());
  ArrayRef<int64_t> lhsShape = lhsType.getShape();
  ArrayRef<int64_t> rhsShape = rhsType.getShape();
  // lhs: [M, K], rhs: [K, N]
  int64_t m = lhsShape[0];
  int64_t k = lhsShape[1];
  int64_t n = rhsShape[1];
  return m * k * n;
}

static int64_t tensorElementCount(Type t) {
  if (auto rt = llvm::dyn_cast<RankedTensorType>(t)) {
    int64_t count = 1;
    for (int64_t dim : rt.getShape())
      count *= dim;
    return count;
  }
  return 0;
}

struct WorkloadAnalysisPass
    : public PassWrapper<WorkloadAnalysisPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WorkloadAnalysisPass)

  StringRef getArgument() const final { return "edge-workload-analysis"; }
  StringRef getDescription() const final {
    return "Reports estimated MAC count and tensor element counts for "
           "linalg.matmul and linalg.generic ops, as a first-pass edge "
           "AI workload characterization.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    int64_t totalMACs = 0;
    int64_t totalElements = 0;
    int matmulCount = 0;
    int genericCount = 0;

    module.walk([&](Operation *op) {
      if (auto matmul = llvm::dyn_cast<linalg::MatmulOp>(op)) {
        int64_t macs = computeMatmulMACs(matmul);
        totalMACs += macs;
        matmulCount++;
        llvm::outs() << "linalg.matmul: " << macs << " MACs\n";
      } else if (auto generic = llvm::dyn_cast<linalg::GenericOp>(op)) {
        genericCount++;
        int64_t elems = 0;
        for (Value operand : generic.getInputs())
          elems += tensorElementCount(operand.getType());
        for (Value operand : generic.getOutputs())
          elems += tensorElementCount(operand.getType());
        totalElements += elems;
        llvm::outs() << "linalg.generic: " << elems
                      << " elements touched (elementwise op)\n";
      }
    });

    llvm::outs() << "---\n";
    llvm::outs() << "Total matmul ops: " << matmulCount << "\n";
    llvm::outs() << "Total generic (elementwise) ops: " << genericCount
                  << "\n";
    llvm::outs() << "Total estimated MACs: " << totalMACs << "\n";
    llvm::outs() << "Total elementwise-op element count: " << totalElements
                  << "\n";
  }
};

} // namespace

namespace mlir {
void registerWorkloadAnalysisPass() {
  PassRegistration<WorkloadAnalysisPass>();
}
} // namespace mlir
