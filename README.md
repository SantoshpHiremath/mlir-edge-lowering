# mlir-edge-lowering

A small, real MLIR compiler project built to close a genuine skills gap
identified against an NXP "Intern (f/m/d) AI/ML Compiler Engineer"
posting: hands-on MLIR compiler work, lowering neural-network
operations toward executable code, and (via a custom pass) basic
workload analysis relevant to mapping AI workloads onto edge hardware.
Deliberately scoped down to something achievable and honestly
describable as an entry-level exploration, not a claim of production
compiler engineering — but every part of it is real LLVM/MLIR 18
infrastructure, actually built and actually run, not a simulation or a
description of what MLIR "would" do.

## What this is

- `mlir/mlp_forward.mlir` — a small 2-layer MLP forward pass
  (Linear → ReLU → Linear), hand-authored directly in MLIR's
  `linalg`/`arith`/`tensor` dialects. This is the same IR level
  (linalg-on-tensor) a real ML-compiler frontend such as torch-mlir
  lowers a PyTorch model down to before target-specific optimization.
- `mlir/mlp_main.mlir` — the same network wrapped with a `main` that
  supplies concrete constant weights and prints the output.
- `run_pipeline.sh` — actually lowers `mlp_main.mlir` through a real,
  multi-stage MLIR pass pipeline (bufferization → linalg-to-loops →
  control-flow lowering → LLVM dialect) and executes the result via
  `mlir-cpu-runner`, MLIR's real JIT execution tool.
- `pass/` — a custom, out-of-tree MLIR pass (`WorkloadAnalysisPass.cpp`)
  compiled into a small custom `edge-opt` tool (mirroring the standard
  MLIR pattern for building your own `mlir-opt` with extra passes
  registered). It walks a module's `linalg.matmul`/`linalg.generic`
  ops and reports estimated multiply-accumulate (MAC) counts and
  tensor element counts per op — a first-pass workload
  characterization step, the kind of thing a real edge-AI compiler
  needs before deciding how to map a network onto constrained
  hardware.
- `tests/` — 8 tests total, all passing, verifying actual numeric
  output against an independent NumPy reference and actual pass output
  against hand-computed MAC/element counts, not just "did it run
  without crashing."

## Honest disclosures — what this is and isn't

**Scope.** This is a deliberately small, entry-level exploration of
MLIR — a 2-layer MLP, not a real model; a workload-analysis pass, not
an optimization or NPU-lowering pass; no actual NPU or embedded target
involved anywhere. It's built to demonstrate genuine, verified,
hands-on engagement with the actual MLIR toolchain the posting names,
not to claim production compiler engineering experience.

**MLIR itself is real, not a proxy.** LLVM/MLIR 18.1.3 was installed
via this environment's package archive (`llvm-18-dev`, `libmlir-18-dev`,
`mlir-18-tools`) and used directly — `mlir-opt-18` for lowering,
`mlir-cpu-runner-18` for JIT execution, and the real MLIR C++ headers
and static libraries for the custom pass. Nothing here is a mock of
MLIR or a hand-written interpreter pretending to be one.

**C/C++: this project is the first real C++ evidence in the
portfolio.** The custom pass (`pass/lib/WorkloadAnalysisPass.cpp`,
`pass/lib/edge-opt-main.cpp`) is genuine C++17, compiled against real
MLIR C++ APIs (`PassWrapper`, `OperationPass`, `DialectRegistry`,
walking the IR via `module.walk()`). Prior projects in this
application portfolio have all been Python; this is honestly the
first C++ work, built specifically because the posting calls for
C/C++ specifically and nothing prior demonstrated it.

**Docker Hub registry access is blocked in this sandbox** (same
constraint documented in an earlier, unrelated project in this
portfolio — verified via a 403 from `registry-1.docker.io`). This
project doesn't containerize anything, so it wasn't a blocker here,
but it's the same environment constraint, not something specific to
NXP or MLIR.

**No embedded hardware, no NPU.** Everything here runs and is verified
on the host CPU via `mlir-cpu-runner`. Nothing in this project touches
an actual embedded target, cross-compilation, or a real NPU backend —
that's a real, unaddressed part of the posting's ask, disclosed here
rather than implied.

## A note on how the "MAC estimation" pass was checked

The custom pass's reported numbers were verified by hand for both
`mlir/mlp_forward.mlir` (24 + 18 = 42 total MACs, 18 + 9 = 27 total
elementwise elements — see `pass/test/expected_output.txt` for the
derivation) and, separately, a differently-shaped standalone matmul
(10×20 @ 20×30 → 6000 MACs) to confirm the MAC-counting formula
(`M*K*N`) generalizes correctly rather than being curve-fit to the one
example in this project. `tests/test_workload_analysis_pass.py` also
verifies the tool fails cleanly with a real parser error on malformed
MLIR input, confirming it's a genuine parser-backed compiler tool and
not a script pattern-matching text.

## Verification performed (all real, all against this exact code)

- `mlir-opt-18 mlir/mlp_forward.mlir` — parses and round-trips cleanly.
- `bash run_pipeline.sh` — lowers `mlp_main.mlir` through the real
  multi-stage pipeline and executes it via `mlir-cpu-runner-18`,
  printing 3 real floating-point numbers.
- `python3 -m pytest tests/test_mlp_reference.py -v` — the compiled
  and executed MLIR program's output is compared against an
  independent NumPy computation of the identical network
  (`abs tolerance 1e-3`), verified to actually match
  (`-0.229, 0.1635, 0.3785`, matching NumPy's
  `-0.22900003, 0.16350001, 0.37850004` to float32 print precision).
- **Confirmed the verification test has real teeth**: temporarily
  changed one bias constant in `mlp_main.mlir` only (leaving the
  Python reference untouched) and reran the test — it correctly
  failed (`0.571 != -0.229`) before the change was reverted and the
  suite confirmed green again.
- `bash pass/build.sh` — configures and builds the custom `edge-opt`
  tool from a clean directory with CMake + Ninja against the real
  installed MLIR/LLVM 18 CMake package config.
- `pass/build/edge-opt --edge-workload-analysis mlir/mlp_forward.mlir`
  — real output, hand-verified against `pass/test/expected_output.txt`.
- `python3 -m pytest tests/test_workload_analysis_pass.py -v` — checks
  the pass's real compiled-binary output against those hand-computed
  numbers, and confirms it rejects malformed MLIR with a real parser
  error rather than silently misreporting.
- Full suite: `python3 -m pytest tests/ -v` — 8/8 pass.

## Running it yourself

```bash
# Lower and JIT-execute the MLP
bash run_pipeline.sh

# Build and run the custom workload-analysis pass
bash pass/build.sh
./pass/build/edge-opt --edge-workload-analysis mlir/mlp_forward.mlir -o /dev/null

# Run all tests
python3 -m pytest tests/ -v
```
