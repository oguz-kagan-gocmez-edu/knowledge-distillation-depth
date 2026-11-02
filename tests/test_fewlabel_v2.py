"""Checks for scripts/fewlabel_v2.py on tiny synthetic data (CPU, no downloads)."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fewlabel_v2.py"
BASE = ["--data", "dummy", "--device", "cpu", "--student", "tiny", "--pool", "32", "--n-val", "16",
        "--labeled", "25%", "--recipes", "gt,semi_edge", "--steps", "8", "--eval-every", "4"]


def run(tmp, *extra):
    subprocess.run([sys.executable, str(SCRIPT), *BASE, "--cache-dir", str(tmp / "cache"), *extra],
                   check=True, capture_output=True, text=True)


def results(path):
    return json.loads((path / "report.json").read_text())["results"]


@pytest.fixture(scope="module")
def fewlabel():
    spec = importlib.util.spec_from_file_location("fewlabel_v2", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_budgets(fewlabel):
    assert fewlabel.parse_budgets("250", 5000) == [250]
    assert fewlabel.parse_budgets("1%,5%,250", 5000) == [50, 250]
    assert fewlabel.parse_budgets("100%", 300) == [300]
    assert fewlabel.parse_budgets("0.1%", 300) == [1]


def test_resume_gives_same_result_as_uninterrupted_run(tmp_path):
    run(tmp_path, "--out", str(tmp_path / "a"), "--ckpt-dir", str(tmp_path / "ck_a"))
    # stop in the middle of the first run, then start again with the same arguments
    run(tmp_path, "--out", str(tmp_path / "b"), "--ckpt-dir", str(tmp_path / "ck_b"), "--stop-after", "5")
    assert (tmp_path / "ck_b" / "gt_lab8_seed0.pt").exists()
    run(tmp_path, "--out", str(tmp_path / "b"), "--ckpt-dir", str(tmp_path / "ck_b"))
    a, b = results(tmp_path / "a"), results(tmp_path / "b")
    assert a.keys() == b.keys()
    for k in a:
        assert a[k]["runs"][0]["history"] == b[k]["runs"][0]["history"], k


def test_finished_runs_are_skipped(tmp_path):
    run(tmp_path, "--out", str(tmp_path / "a"), "--ckpt-dir", str(tmp_path / "ck"))
    run(tmp_path, "--out", str(tmp_path / "c"), "--ckpt-dir", str(tmp_path / "ck"))
    a, c = results(tmp_path / "a"), results(tmp_path / "c")
    assert all(a[k]["mean"] == c[k]["mean"] for k in a)


def test_teacher_file_is_reused(tmp_path):
    run(tmp_path, "--out", str(tmp_path / "a"))
    pool = next((tmp_path / "cache").glob("dummy_pool32_*")) / "pool"
    rgb = np.load(pool / "rgb.npy")
    disp_mtime = (pool / "disp_small.npy").stat().st_mtime_ns
    (pool / "rgb.npy").unlink()
    (pool / "depth.npy").unlink()
    run(tmp_path, "--out", str(tmp_path / "b"))
    assert np.array_equal(np.load(pool / "rgb.npy"), rgb)
    assert (pool / "disp_small.npy").stat().st_mtime_ns == disp_mtime   # teacher not run again
    a, b = results(tmp_path / "a"), results(tmp_path / "b")
    assert all(a[k]["runs"][0]["history"] == b[k]["runs"][0]["history"] for k in a)
