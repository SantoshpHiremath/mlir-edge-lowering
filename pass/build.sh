#!/bin/bash
# Builds edge-opt (the custom MLIR tool with WorkloadAnalysisPass
# registered). Requires the LLVM/MLIR 18 dev packages:
#   sudo apt-get install -y llvm-18-dev libmlir-18-dev mlir-18-tools cmake ninja-build
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/build"
cd "$SCRIPT_DIR/build"
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja
echo "Built pass/build/edge-opt"
echo "Try: ./build/edge-opt --edge-workload-analysis ../mlir/mlp_forward.mlir -o /dev/null"
