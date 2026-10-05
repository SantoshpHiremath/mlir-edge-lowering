# mlir-edge-lowering

A small MLIR compiler project: hand-authored neural-network operations lowered toward executable code, plus a custom pass that does basic workload analysis relevant to mapping AI workloads onto edge hardware. Every part of it is real LLVM/MLIR 18 infrastructure, built and run, and scoped as an entry-level exploration of the MLIR toolchain.

## What it does

- `mlir/mlp_forward.mlir`: a small 2-layer MLP forward pass (Linear, ReLU, Linear), hand-authored directly in MLIR's `linalg`/`arith`/`tensor` dialects. This is the same IR level (linalg-on-tensor) a ML-compiler frontend such as torch-mlir lowers a PyTorch model down to before target-specific optimization.
- `mlir/mlp_main.mlir`: the same network wrapped with a `main` that supplies concrete constant weights and prints the output.
- `run_pipeline.sh`: lowers `mlp_main.mlir` through a multi-stage MLIR pass pipeline (bufferization, linalg-to-loops, control-flow lowering, LLVM dialect) and executes the result via `mlir-cpu-runner`, MLIR's JIT execution tool.
- `pass/`: a custom, out-of-tree MLIR pass (`WorkloadAnalysisPass.cpp`) compiled into a small custom `edge-opt` tool (the standard MLIR pattern for building your own `mlir-opt` with extra passes registered). It walks a module's `linalg.matmul`/`linalg.generic` ops and reports estimated multiply-accumulate (MAC) counts and tensor element counts per op, a first-pass workload characterization step an edge-AI compiler needs before deciding how to map a network onto constrained hardware.
- `tests/`: 8 tests total, all passing, verifying numeric output against an independent NumPy reference and pass output against hand-computed MAC and element counts, not just "did it run without crashing."

## Scope

The project is a 2-layer MLP and a workload-analysis pass (not an optimization or NPU-lowering pass), and everything runs and is verified on the host CPU via `mlir-cpu-runner`.

LLVM/MLIR 18.1.3 was installed from the distribution packages (`llvm-18-dev`, `libmlir-18-dev`, `mlir-18-tools`) and used directly: `mlir-opt-18` for lowering, `mlir-cpu-runner-18` for JIT execution, and the MLIR C++ headers and static libraries for the custom pass.

The custom pass (`pass/lib/WorkloadAnalysisPass.cpp`, `pass/lib/edge-opt-main.cpp`) is C++17, compiled against the MLIR C++ APIs (`PassWrapper`, `OperationPass`, `DialectRegistry`, walking the IR via `module.walk()`).

## Results

The custom pass's reported numbers were verified by hand for both `mlir/mlp_forward.mlir` (24 + 18 = 42 total MACs, 18 + 9 = 27 total elementwise elements; see `pass/test/expected_output.txt` for the derivation) and, separately, a differently-shaped standalone matmul (10×20 @ 20×30 → 6000 MACs) to confirm the MAC-counting formula (`M*K*N`) generalizes rather than being fit to one example. `tests/test_workload_analysis_pass.py` also verifies the tool fails cleanly with a parser error on malformed MLIR input, confirming it is a parser-backed compiler tool rather than a script pattern-matching text.

The compiled and executed MLIR program outputs `-0.229, 0.1635, 0.3785`, matching NumPy's `-0.22900003, 0.16350001, 0.37850004` to float32 print precision.

## Tests

- `mlir-opt-18 mlir/mlp_forward.mlir` parses and round-trips cleanly.
- `bash run_pipeline.sh` lowers `mlp_main.mlir` through the multi-stage pipeline and executes it via `mlir-cpu-runner-18`, printing 3 floating-point numbers.
- `python3 -m pytest tests/test_mlp_reference.py -v` compares the compiled and executed MLIR program's output against an independent NumPy computation of the identical network (`abs tolerance 1e-3`).
- The reference test is sensitive to real changes: I temporarily changed one bias constant in `mlp_main.mlir` only (leaving the Python reference untouched), and the test correctly failed (`0.571 != -0.229`) before I reverted the change and confirmed the suite green again.
- `bash pass/build.sh` configures and builds the custom `edge-opt` tool from a clean directory with CMake + Ninja against the installed MLIR/LLVM 18 CMake package config.
- `pass/build/edge-opt --edge-workload-analysis mlir/mlp_forward.mlir` produces output hand-verified against `pass/test/expected_output.txt`.
- `python3 -m pytest tests/test_workload_analysis_pass.py -v` checks the pass's compiled-binary output against those hand-computed numbers, and confirms it rejects malformed MLIR with a parser error rather than silently misreporting.
- Full suite: `python3 -m pytest tests/ -v` gives 8/8 passing.

## Project structure

```
mlir/
  mlp_forward.mlir
  mlp_main.mlir
pass/
  CMakeLists.txt
  build.sh
  lib/WorkloadAnalysisPass.cpp
  lib/edge-opt-main.cpp
  test/expected_output.txt
tests/
  test_mlp_reference.py
  test_workload_analysis_pass.py
run_pipeline.sh
```

## Running it

```bash
# Lower and JIT-execute the MLP
bash run_pipeline.sh

# Build and run the custom workload-analysis pass
bash pass/build.sh
./pass/build/edge-opt --edge-workload-analysis mlir/mlp_forward.mlir -o /dev/null

# Run all tests
python3 -m pytest tests/ -v
```

## Notes

Nothing in the project targets an embedded device or NPU backend; execution is on the host CPU. The workload-analysis pass is analysis-only (no IR transformation), which keeps it independently checkable against hand-computed counts.

## Possible extensions

- Lower from a real frontend (torch-mlir) instead of hand-authored IR.
- Extend the pass into a fusion or tiling transformation guided by the MAC and footprint estimates.
- Cross-compile to an embedded target and add an NPU-oriented lowering.
