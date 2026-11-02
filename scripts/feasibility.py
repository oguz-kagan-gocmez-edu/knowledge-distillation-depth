#!/usr/bin/env python3
"""
CS466 term project - feasibility test (full labels, short run).

What this script does, in one run:
  1. Loads a small slice of NYU Depth V2 (default: 2000 train, 200 val images).
  2. Runs the teacher (Depth Anything V2) once and aligns its output to metric
     depth with a per-image scale and shift.
  3. Builds the small student (MobileNetV3 encoder + light decoder) and measures
     its parameters, FLOPs and latency.
  4. Trains the student briefly with two recipes:
        gt   = real depth only            (E1, baseline)
        full = real depth + teacher + edge (E4, full recipe)
  5. Evaluates both (AbsRel, RMSE, delta<1.25), saves a figure and a JSON report,
     and estimates time and disk for the full project (50,688 images).

Run from the repository root:
  python scripts/feasibility.py --source dummy --quick        # offline pipeline check
  python scripts/feasibility.py --quick                       # 2-3 min, streams from Hugging Face
  python scripts/feasibility.py --source densedepth --data-zip data/nyu_data

Results go to results/feasibility/ (report.json, examples.png). Data + teacher
outputs are cached in cache/, so a second run skips the slow part.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kd_depth.core import (  # noqa: E402
    FULL_TRAIN,
    MAX_D,
    MIN_D,
    RECIPES,
    DepthAnythingTeacher,
    DummyTeacher,
    H,
    Student,
    W,
    build_split,
    count_flops,
    depth_metrics,
    latency_ms,
    pick_device,
    predict,
    save_figure,
    train_recipe,
)
from kd_depth.runinfo import run_info  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["hf", "densedepth", "dummy"], default="hf")
    ap.add_argument("--data-zip", help="DenseDepth-style NYU zip or extracted folder "
                                       "(for --source densedepth; the Kaggle copy works too)")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--teacher", choices=["small", "base", "large"], default="small")
    ap.add_argument("--teacher-batch", type=int, default=8)
    ap.add_argument("--student", choices=list(Student.SIZES), default="small")
    ap.add_argument("--recipes", default="gt,full", help=f"comma list from {list(RECIPES)}")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lambda-edge", type=float, default=1.0)
    ap.add_argument("--plan-epochs", type=int, default=20, help="epochs per run in the full plan")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/feasibility")
    ap.add_argument("--cache-dir", default="cache", help="data + teacher output cache")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny run to check that everything works")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.quick:
        args.n_train, args.n_val, args.steps = 200, 40, 100
    if args.source == "dummy":
        args.no_pretrained = True
    recipes = [r.strip() for r in args.recipes.split(",") if r.strip()]
    for r in recipes:
        if r not in RECIPES:
            raise SystemExit(f"unknown recipe {r!r}; choose from {list(RECIPES)}")

    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    info = run_info()
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    print(f"Device: {device} | torch {torch.__version__} | source: {args.source}")

    # 1) teacher + data
    print("\n[1/4] Loading data and running the teacher once")
    teacher = DummyTeacher() if args.source == "dummy" else DepthAnythingTeacher(args.teacher, device)
    print(f"  teacher: {teacher.name} ({teacher.params / 1e6:.1f}M params)")
    train, train_meta = build_split(args, "train", args.n_train, teacher, device, cache_dir)
    val, val_meta = build_split(args, "val", args.n_val, teacher, device, cache_dir)
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    # 2) student size and speed
    print("\n[2/4] Measuring the student")
    probe = Student(args.student, pretrained=False)
    eff = {
        "student": args.student,
        "params_M": sum(p.numel() for p in probe.parameters()) / 1e6,
        "GFLOPs": count_flops(probe) / 1e9,
        "latency_ms_device": latency_ms(probe, device),
        "latency_ms_cpu": latency_ms(probe, torch.device("cpu"), warmup=3, runs=20),
    }
    print(f"  {eff['params_M']:.2f}M params, {eff['GFLOPs']:.2f} GFLOPs at {W}x{H}, "
          f"{eff['latency_ms_device']:.1f} ms on {device}, {eff['latency_ms_cpu']:.1f} ms on CPU")

    # 3) short training runs
    print(f"\n[3/4] Training {recipes} for {args.steps} steps each")
    results, preds = {}, {}
    for name in recipes:
        model, info = train_recipe(name, train, args, device)
        preds[name] = predict(model, val["rgb"], device)
        info["val"] = depth_metrics(preds[name], val["depth"])
        results[name] = info
        del model

    # 4) evaluation, estimates, verdict
    print("\n[4/4] Evaluating")
    train_valid = train["depth"][(train["depth"] > MIN_D) & (train["depth"] < MAX_D)]
    const = np.full_like(val["depth"], float(np.median(train_valid)))
    ref = {
        "constant_median_depth": depth_metrics(const, val["depth"]),
        "teacher_aligned_per_image_optimistic": depth_metrics(val["teacher"], val["depth"]),
    }

    t_img = train_meta["teacher_s_per_img"]
    step_s = float(np.mean([r["s_per_step"] for r in results.values()]))
    epoch_h = step_s * math.ceil(FULL_TRAIN / args.batch_size) / 3600
    run_h = epoch_h * args.plan_epochs
    plan_runs = 6.5   # E1-E4 (4 runs) + E5 (2 more sizes) + E6 (5k + 20k ~ 0.5 run)
    est = {
        "teacher_precompute_h_50k": t_img * FULL_TRAIN / 3600,
        "teacher_cache_gb_50k_fp16": FULL_TRAIN * H * W * 2 / 1e9,
        "lowres_dataset_cache_gb_50k": FULL_TRAIN * H * W * (3 + 2 + 2) / 1e9,
        "train_h_per_epoch_50k": epoch_h,
        "train_h_per_run": run_h,
        "plan_runs": plan_runs,
        "plan_total_h": run_h * plan_runs + t_img * FULL_TRAIN / 3600,
    }

    def beats_constant(m):
        c = ref["constant_median_depth"]
        wins = [m["AbsRel"] < c["AbsRel"], m["RMSE"] < c["RMSE"], m["delta1"] > c["delta1"]]
        return sum(wins) >= 2

    learns = all(beats_constant(results[r]["val"]) for r in results)
    total = est["plan_total_h"]
    if total <= 60:
        fit = "COMFORTABLE: the full plan fits easily on this machine."
    elif total <= 150:
        fit = ("TIGHT: doable, but use fewer epochs, a smaller student, or run the "
               "longest jobs on a faster Colab GPU.")
    else:
        fit = ("TOO HEAVY on this machine as planned: cut epochs/resolution or move "
               "training to a faster Colab GPU (L4/A100).")
    teacher_effect = None
    if {"gt", "full"} <= results.keys():
        g, f_ = results["gt"]["val"], results["full"]["val"]
        rel = (g["AbsRel"] - f_["AbsRel"]) / g["AbsRel"]      # > 0 means 'full' is better
        if rel > 0.03 and f_["delta1"] > g["delta1"]:
            teacher_effect = f"yes ({rel:.0%} lower AbsRel than 'gt')"
        elif rel < -0.03 and f_["delta1"] < g["delta1"]:
            teacher_effect = f"no, 'full' is worse ({-rel:.0%} higher AbsRel)"
        else:
            teacher_effect = "no clear difference (within noise, < 3%)"
    verdict = {
        "student_learns": bool(learns),
        "compute_fit": fit,
        "teacher_effect_in_short_run": teacher_effect,
    }

    report = {
        "run": info,
        "config": vars(args) | {"device": str(device), "torch": torch.__version__,
                                "resolution": [W, H]},
        "data": {"train": train_meta, "val": val_meta},
        "efficiency": eff, "training": results, "reference": ref,
        "estimates_full_project": est, "verdict": verdict,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    save_figure(out / "examples.png", val, val["teacher"], preds)

    # ---- printed summary --------------------------------------------------
    line = "-" * 66
    print(f"\n{line}\nFEASIBILITY SUMMARY  ({device}, {train_meta['n']} train / "
          f"{val_meta['n']} val images, {args.steps} steps)\n{line}")
    print(f"{'Model':<34}{'AbsRel':>9}{'RMSE':>9}{'d<1.25':>9}")
    rows = [("Constant depth (sanity floor)", ref["constant_median_depth"])]
    rows += [(f"Student, recipe '{n}'", r["val"]) for n, r in results.items()]
    rows += [("Teacher, GT-aligned (optimistic)", ref["teacher_aligned_per_image_optimistic"])]
    for label, m in rows:
        print(f"{label:<34}{m['AbsRel']:>9.3f}{m['RMSE']:>9.3f}{m['delta1']:>9.3f}")
    print(line)
    print(f"Student: {eff['params_M']:.2f}M params | {eff['GFLOPs']:.2f} GFLOPs | "
          f"{eff['latency_ms_device']:.1f} ms ({device}) | {eff['latency_ms_cpu']:.1f} ms (CPU)")
    print(f"Teacher speed: {t_img * 1000:.0f} ms/image | training: {step_s:.3f} s/step "
          f"(batch {args.batch_size}) | peak memory "
          f"{max(r['peak_mem_gb'] for r in results.values()):.1f} GB")
    print(line)
    print("Estimates for the full project (50,688 training images):")
    print(f"  teacher pre-compute : {est['teacher_precompute_h_50k']:.1f} h, "
          f"cache {est['teacher_cache_gb_50k_fp16']:.1f} GB")
    print(f"  one training run    : {est['train_h_per_run']:.1f} h "
          f"({args.plan_epochs} epochs x {est['train_h_per_epoch_50k']:.2f} h)")
    print(f"  whole plan E1-E6    : ~{est['plan_total_h']:.0f} h of compute")
    print(line)
    print(f"Student learns (beats constant depth): "
          f"{'YES' if learns else 'NO - train longer or check the data'}")
    if teacher_effect is not None:
        print(f"Teacher helps in this short run      : {teacher_effect}")
    print(f"Compute: {fit}")
    print(f"{line}\nSaved: {out / 'report.json'} and {out / 'examples.png'}")


if __name__ == "__main__":
    main()
