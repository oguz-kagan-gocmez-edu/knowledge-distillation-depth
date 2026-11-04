"""Checks for scripts/fewlabel_v2.py on tiny synthetic data (CPU, no downloads)."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

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
    done = torch.load(tmp_path / "ck" / "gt_lab8_seed0.done.pt", weights_only=False)
    assert done["best_state"].keys() == done["final_state"].keys()      # weights kept for the test set


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


def test_changing_val_size_keeps_pool_teacher(tmp_path):
    run(tmp_path, "--out", str(tmp_path / "a"))
    pool = tmp_path / "cache" / "dummy_pool32_seed0" / "pool"
    disp_mtime = (pool / "disp_small.npy").stat().st_mtime_ns
    run(tmp_path, "--out", str(tmp_path / "b"), "--n-val", "24")
    assert (pool / "disp_small.npy").stat().st_mtime_ns == disp_mtime
    assert (tmp_path / "cache" / "dummy_pool32_seed0" / "val24" / "disp_small.npy").exists()


def test_old_cache_layout_is_still_used(tmp_path):
    run(tmp_path, "--out", str(tmp_path / "a"))
    new = tmp_path / "cache" / "dummy_pool32_seed0"
    old = tmp_path / "cache" / "dummy_pool32_val16_seed0"
    old.mkdir()
    (new / "pool").rename(old / "pool")
    (new / "val16").rename(old / "val")
    new.rmdir()
    run(tmp_path, "--out", str(tmp_path / "b"))
    assert not new.exists()                      # nothing rebuilt in the new layout
    a, b = results(tmp_path / "a"), results(tmp_path / "b")
    assert all(a[k]["runs"][0]["history"] == b[k]["runs"][0]["history"] for k in a)


def test_label_subsets(fewlabel):
    assert fewlabel.labeled_indices(5, 100, 0).tolist() == [0, 1, 2, 3, 4]
    a1, a5 = fewlabel.labeled_indices(10, 1000, 1), fewlabel.labeled_indices(50, 1000, 1)
    assert set(a1.tolist()) <= set(a5.tolist())                       # 1% inside 5%
    assert set(a1.tolist()) != set(fewlabel.labeled_indices(10, 1000, 2).tolist())


def test_teacher_weight(fewlabel):
    w = lambda step, decay: fewlabel.teacher_weight(step, 100, 0.3, decay)
    assert w(15, 0) == 0.5 and w(30, 0) == 1.0 and w(100, 0) == 1.0
    assert w(50, 0.5) == 1.0 and w(75, 0.5) == 0.5 and w(100, 0.5) == 0.0


def test_new_options_keep_old_signatures(fewlabel):
    import argparse
    keys = ("student", "teacher", "steps", "eval_every", "batch_size", "lr", "lambda_t", "lambda_e",
            "ramp", "clip", "no_pretrained", "data", "n_val", "val_scene_frac", "seed")
    args = argparse.Namespace(**{k: 0 for k in keys}, teacher_decay=0.0, label_subset=0)
    assert "teacher_decay" not in fewlabel.run_signature("gt", 4, 0, 32, args)   # old checkpoints resume
    args.label_subset = 2
    assert fewlabel.run_signature("gt", 4, 0, 32, args)["label_subset"] == 2


def test_evaluate_test_set(tmp_path):
    from PIL import Image
    test = tmp_path / "nyu" / "data" / "nyu2_test"
    test.mkdir(parents=True)
    rows, rng = [], np.random.default_rng(0)
    for i in range(3):
        Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)).save(test / f"{i}_colors.png")
        Image.fromarray(rng.integers(500, 9000, (480, 640)).astype(np.uint16)).save(test / f"{i}_depth.png")
        rows.append(f"data/nyu2_test/{i}_colors.png,data/nyu2_test/{i}_depth.png")
    (tmp_path / "nyu" / "data" / "nyu2_test.csv").write_text("\n".join(rows) + "\n")
    run(tmp_path, "--out", str(tmp_path / "a"), "--ckpt-dir", str(tmp_path / "ck"))
    cmd = [sys.executable, str(ROOT / "scripts" / "evaluate_test.py"), "--data", str(tmp_path / "nyu"),
           "--ckpt-dir", str(tmp_path / "ck"), "--device", "cpu", "--out", str(tmp_path / "t")]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    rep = json.loads((tmp_path / "t" / "report.json").read_text())
    assert rep["protocol"]["test_images"] == 3
    assert {(s["recipe"], s["model"]) for s in rep["summary"]} == {
        (r, m) for r in ("gt", "semi_edge") for m in ("best", "final")}
    assert {"raw", "aligned"} <= set(rep["summary"][0]["metrics"])
