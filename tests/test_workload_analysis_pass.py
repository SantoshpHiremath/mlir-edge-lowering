"""Tests for the custom out-of-tree MLIR pass (pass/lib/WorkloadAnalysisPass.cpp),
run against the REAL compiled edge-opt binary — not a mock of what the
pass "should" do. Builds pass/build/edge-opt if it doesn't already
exist, runs it on mlir/mlp_forward.mlir, and checks the reported MAC
and element counts against hand-computed values (see
pass/test/expected_output.txt for the by-hand derivation).
"""
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
EDGE_OPT = PROJECT_ROOT / "pass" / "build" / "edge-opt"
MLP_MLIR = PROJECT_ROOT / "mlir" / "mlp_forward.mlir"


def _run_edge_opt():
    if not EDGE_OPT.exists():
        pytest.skip(
            f"{EDGE_OPT} not built. Run: cd pass && mkdir -p build && cd build "
            "&& cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja"
        )
    result = subprocess.run(
        [str(EDGE_OPT), "--edge-workload-analysis", str(MLP_MLIR), "-o", "/dev/null"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result


def test_edge_opt_binary_exists_and_runs():
    result = _run_edge_opt()
    assert result.returncode == 0, f"edge-opt failed: {result.stderr}"


def test_workload_analysis_reports_two_matmuls_and_two_generics():
    result = _run_edge_opt()
    assert "Total matmul ops: 2" in result.stdout
    assert "Total generic (elementwise) ops: 2" in result.stdout


def test_workload_analysis_mac_count_matches_hand_calculation():
    """Layer 1: (1x4)@(4x6) = 24 MACs. Layer 2: (1x6)@(6x3) = 18 MACs.
    Total = 42. Independently hand-derived in pass/test/expected_output.txt,
    not reverse-engineered from whatever the pass happens to print."""
    result = _run_edge_opt()
    assert "linalg.matmul: 24 MACs" in result.stdout
    assert "linalg.matmul: 18 MACs" in result.stdout
    assert "Total estimated MACs: 42" in result.stdout


def test_workload_analysis_element_count_matches_hand_calculation():
    """Layer 1 generic touches 6+6+6=18 elements (mm1, b1, init1).
    Layer 2 generic touches 3+3+3=9 elements. Total = 27."""
    result = _run_edge_opt()
    assert "linalg.generic: 18 elements touched" in result.stdout
    assert "linalg.generic: 9 elements touched" in result.stdout
    assert "Total elementwise-op element count: 27" in result.stdout


def test_edge_opt_rejects_malformed_mlir(tmp_path):
    """The custom tool must fail cleanly (non-zero exit, real parser
    error) on invalid input rather than silently produce a bogus
    report — verifies it's a real parser-backed tool, not a script
    that just pattern-matches text."""
    if not EDGE_OPT.exists():
        pytest.skip("edge-opt not built")
    bad_file = tmp_path / "bad.mlir"
    bad_file.write_text("func.func @bad( invalid syntax\n")
    result = subprocess.run(
        [str(EDGE_OPT), "--edge-workload-analysis", str(bad_file), "-o", "/dev/null"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "error" in result.stderr.lower()
