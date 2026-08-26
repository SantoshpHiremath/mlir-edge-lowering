"""NumPy reference implementation of the same 2-layer MLP forward pass
hand-written in mlir/mlp_main.mlir, using the exact same constant
weights/biases. This is the single source of truth for what the MLIR
program's numeric output SHOULD be — the real verification step is
run_pipeline.sh actually executing the compiled MLIR via
mlir-cpu-runner and this test comparing the printed output against
compute_reference() below, not just checking that the pipeline runs
without crashing.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent


def compute_reference():
    x = np.array([[1.0, -2.0, 0.5, 3.0]], dtype=np.float32)

    w1 = np.array([
        [0.1, 0.2, -0.1, 0.05, 0.3, -0.2],
        [0.4, -0.3, 0.2, 0.1, -0.1, 0.05],
        [-0.2, 0.1, 0.3, -0.4, 0.2, 0.1],
        [0.05, 0.15, -0.25, 0.2, -0.1, 0.3],
    ], dtype=np.float32)
    b1 = np.array([[0.01, -0.02, 0.03, 0.0, -0.01, 0.02]], dtype=np.float32)

    w2 = np.array([
        [0.2, -0.1, 0.3],
        [-0.3, 0.2, 0.1],
        [0.1, 0.4, -0.2],
        [0.05, -0.05, 0.15],
        [-0.2, 0.3, 0.1],
        [0.15, -0.1, 0.2],
    ], dtype=np.float32)
    b2 = np.array([[0.1, -0.1, 0.05]], dtype=np.float32)

    layer1 = np.maximum(x @ w1 + b1, 0.0)  # Linear + ReLU
    output = layer1 @ w2 + b2              # Linear (no activation)
    return output[0]


def test_reference_shape_and_finiteness():
    """Sanity check on the reference itself before it's used to verify
    anything else."""
    out = compute_reference()
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))


def test_reference_matches_manual_hand_calculation():
    """Independent hand/manual-style check of at least one output
    element, computed a different way (explicit loops, not matrix
    ops) — guards against a bug in compute_reference() itself being
    silently 'confirmed' by the MLIR run just reproducing the same
    bug."""
    x = [1.0, -2.0, 0.5, 3.0]
    w1_col0 = [0.1, 0.4, -0.2, 0.05]
    b1_0 = 0.01
    layer1_0 = max(sum(x[i] * w1_col0[i] for i in range(4)) + b1_0, 0.0)

    # Full layer 1 computed via reference for the remaining columns,
    # then layer 2 element 0 computed manually from that.
    ref = compute_reference()
    x_arr = np.array([x], dtype=np.float32)
    w1 = np.array([
        [0.1, 0.2, -0.1, 0.05, 0.3, -0.2],
        [0.4, -0.3, 0.2, 0.1, -0.1, 0.05],
        [-0.2, 0.1, 0.3, -0.4, 0.2, 0.1],
        [0.05, 0.15, -0.25, 0.2, -0.1, 0.3],
    ], dtype=np.float32)
    b1 = np.array([[0.01, -0.02, 0.03, 0.0, -0.01, 0.02]], dtype=np.float32)
    layer1_full = np.maximum(x_arr @ w1 + b1, 0.0)[0]

    assert layer1_full[0] == pytest.approx(layer1_0, abs=1e-5)

    w2_col0 = [0.2, -0.3, 0.1, 0.05, -0.2, 0.15]
    b2_0 = 0.1
    out_0_manual = sum(layer1_full[i] * w2_col0[i] for i in range(6)) + b2_0
    assert ref[0] == pytest.approx(out_0_manual, abs=1e-5)


def _run_pipeline_and_get_output():
    """Actually runs the shell script that lowers and executes the
    real MLIR program via mlir-cpu-runner, and parses its stdout.
    Skips (not fails) if the MLIR toolchain isn't available in the
    current environment, so this test suite can still run elsewhere,
    but in this project's actual sandbox it runs for real."""
    script = PROJECT_ROOT / "run_pipeline.sh"
    if not script.exists():
        pytest.skip("run_pipeline.sh not found")
    result = subprocess.run(
        ["bash", str(script)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        pytest.fail(
            f"run_pipeline.sh failed (exit {result.returncode}).\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
    # The script's final section prints exactly 3 float lines from
    # mlir-cpu-runner (see run_pipeline.sh) — take the last 3 non-empty
    # lines to be robust to any earlier informational echo lines.
    numeric_lines = [l for l in lines if _looks_numeric(l)]
    if len(numeric_lines) < 3:
        pytest.fail(f"expected at least 3 numeric output lines, got: {lines}")
    return [float(v) for v in numeric_lines[-3:]]


def _looks_numeric(line):
    try:
        float(line.strip())
        return True
    except ValueError:
        return False


def test_compiled_mlir_output_matches_numpy_reference():
    """THE core verification of this project: the real, hand-written
    MLIR program is actually lowered through a genuine multi-stage
    MLIR pass pipeline (linalg -> loops -> LLVM dialect -> LLVM IR)
    and executed with mlir-cpu-runner, and its printed numeric output
    is compared against the independent NumPy reference above. This is
    what proves the lowering pipeline is semantics-preserving for this
    program, not just that it doesn't crash."""
    mlir_output = _run_pipeline_and_get_output()
    reference = compute_reference()

    assert len(mlir_output) == 3
    for i, (got, expected) in enumerate(zip(mlir_output, reference)):
        assert got == pytest.approx(float(expected), abs=1e-3), (
            f"output[{i}]: MLIR-compiled result {got} != NumPy reference {expected}"
        )
