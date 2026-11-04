#!/usr/bin/env python3
"""
Evaluate finished runs on the 654-image NYU Depth V2 test set (Eigen split).

Reads the .done.pt files that scripts/fewlabel_v2.py writes into --ckpt-dir. Each holds the
best-on-validation model (best_state) and the final-step model (final_state).

Protocol (the usual one for NYU): the student predicts at 256x192, the prediction is upsampled
to 640x480, and metrics are computed inside the Eigen crop on pixels with 0.001 m < depth < 10 m.
Three views of the same predictions:
  raw      metric depth as predicted (what the report uses)
  aligned  each prediction scaled by median(real) / median(predicted): structure only, no scale
  ranges   AbsRel and RMSE for near (< 3 m), middle (3-6 m) and far (> 6 m) pixels

Run from the repository root, e.g.
  python scripts/evaluate_test.py --ckpt-dir checkpoints/main_1_5 --out results/test_main_1_5
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import kd_depth.core as fz  # noqa: E402
from kd_depth.core import MAX_D, MIN_D  # noqa: E402
from kd_depth.runinfo import run_info  # noqa: E402

_spec = importlib.util.spec_from_file_location("fewlabel_v2", Path(__file__).with_name("fewlabel_v2.py"))
fl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fl)

CROP = (slice(45, 471), slice(41, 601))        # Eigen crop on 480x640
RANGES = {"near": (0.0, 3.0), "mid": (3.0, 6.0), "far": (6.0, MAX_D)}
METRICS = ("AbsRel", "RMSE", "delta1", "delta2", "delta3", "log10")


def load_test(path):
    """RGB at the student's input size (uint8) and real depth at full size inside the crop (meters)."""
    rows, read, resolve = fl.open_densedepth(path, "nyu2_test.csv")
    rgb = np.zeros((len(rows), fz.H, fz.W, 3), np.uint8)
    depth = []
    for i, (rp, dp) in enumerate(rows):
        im = Image.open(io.BytesIO(read(resolve(rp)))).convert("RGB")
        rgb[i] = np.asarray(im.resize((fz.W, fz.H), Image.BILINEAR))
        d = fz.to_meters(np.array(Image.open(io.BytesIO(read(resolve(dp))))))
        if d.shape != (480, 640):
            raise SystemExit(f"{dp}: expected 480x640 depth, got {d.shape}")
        depth.append(d[CROP].astype(np.float32))
    return rgb, np.stack(depth)


def image_metrics(p, g):
    """Metrics of one image; p and g are depth maps of the same size, already cropped."""
    v = (g > MIN_D) & (g < MAX_D)
    p, g = np.clip(p[v], MIN_D, MAX_D), g[v]
    out = {}
    for name, q in (("raw", p), ("aligned", np.clip(p * np.median(g) / np.median(p), MIN_D, MAX_D))):
        ratio = np.maximum(q / g, g / q)
        out[name] = {"AbsRel": np.mean(np.abs(q - g) / g), "RMSE": np.sqrt(np.mean((q - g) ** 2)),
                     "delta1": np.mean(ratio < 1.25), "delta2": np.mean(ratio < 1.25 ** 2),
                     "delta3": np.mean(ratio < 1.25 ** 3),
                     "log10": np.mean(np.abs(np.log10(q) - np.log10(g)))}
    for name, (lo, hi) in RANGES.items():
        m = (g >= lo) & (g < hi)
        if m.any():
            out[name] = {"AbsRel": np.mean(np.abs(p[m] - g[m]) / g[m]),
                         "RMSE": np.sqrt(np.mean((p[m] - g[m]) ** 2)), "pixels": int(m.sum())}
    return out


@torch.no_grad()
def evaluate(model, rgb, depth, device, bs=16):
    """Per-image metrics averaged over the test set (ranges: over images that have such pixels)."""
    model.eval()
    acc = {}
    for i in range(0, len(rgb), bs):
        x = torch.from_numpy(np.array(rgb[i:i + bs])).permute(0, 3, 1, 2).float().div(255).to(device)
        pred = model(x).float().unsqueeze(1)
        pred = F.interpolate(pred, size=(480, 640), mode="bilinear", align_corners=False)
        pred = pred[:, 0, CROP[0], CROP[1]].cpu().numpy()
        for p, g in zip(pred, depth[i:i + bs]):
            for view, ms in image_metrics(p, g).items():
                for k, v in ms.items():
                    acc.setdefault(view, {}).setdefault(k, []).append(float(v))
    return {view: {k: (float(np.sum(v)) if k == "pixels" else float(np.mean(v))) for k, v in ms.items()}
            for view, ms in acc.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/nyu_data", help="DenseDepth zip or folder (with nyu2_test.csv)")
    ap.add_argument("--ckpt-dir", nargs="+", required=True, help="folder(s) with .done.pt files")
    ap.add_argument("--which", default="best,final", help="best and/or final model of each run")
    ap.add_argument("--out", default="results/test")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = fz.pick_device(args.device)
    which = [w.strip() for w in args.which.split(",")]

    print(f"Device: {device}\nLoading the test set from {args.data} ...")
    rgb, depth = load_test(args.data)
    print(f"  {len(rgb)} test images")

    runs, skipped = [], []
    for cdir in args.ckpt_dir:
        for f in sorted(Path(cdir).glob("*.done.pt")):
            r = torch.load(f, weights_only=False, map_location="cpu")
            sig = r["signature"]
            for w in which:
                state = r.get(f"{w}_state")
                if state is None:
                    skipped.append(f"{f.name} ({w})")
                    continue
                model = fz.Student(sig["student"], pretrained=False).to(device)
                model.load_state_dict(state)
                m = evaluate(model, rgb, depth, device)
                runs.append({"file": str(f), "model": w, "recipe": sig["recipe"], "n_lab": sig["n_lab"],
                             "seed": sig["train_seed"], "student": sig["student"],
                             "label_subset": sig.get("label_subset", 0),
                             "teacher_decay": sig.get("teacher_decay", 0.0), "steps": sig["steps"],
                             "metrics": m})
                print(f"  {f.parent.name}/{f.name} [{w}]  AbsRel {m['raw']['AbsRel']:.3f}  "
                      f"RMSE {m['raw']['RMSE']:.3f}  d1 {m['raw']['delta1']:.3f}  "
                      f"(aligned AbsRel {m['aligned']['AbsRel']:.3f})")
    if skipped:
        print(f"  no weights saved (finished before weights were kept), skipped: {', '.join(skipped)}")
    if not runs:
        raise SystemExit("nothing to evaluate")

    # mean +- std over seeds for each setting
    groups = {}
    for r in runs:
        key = (r["student"], r["steps"], r["teacher_decay"], r["n_lab"], r["recipe"], r["model"])
        groups.setdefault(key, []).append(r)
    summary = []
    line = "-" * 104
    print(f"\n{line}\nTEST SET (654 images, Eigen crop, cap 10 m); mean ± std over seeds\n{line}")
    print(f"{'student':<8}{'steps':>7}{'decay':>6}{'labeled':>9}  {'recipe':<12}{'model':<7}{'n':>3}"
          f"{'AbsRel':>15}{'RMSE':>8}{'d1':>7}{'aligned':>9}{'far RMSE':>10}")
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2], k[3], k[5], k[4])):
        rs = groups[key]
        agg = {view: {k: [float(np.mean([r["metrics"][view][k] for r in rs])),
                          float(np.std([r["metrics"][view][k] for r in rs]))]
                      for k in rs[0]["metrics"][view] if k != "pixels"}
               for view in rs[0]["metrics"] if all(view in r["metrics"] for r in rs)}
        summary.append({"student": key[0], "steps": key[1], "teacher_decay": key[2], "n_lab": key[3],
                        "recipe": key[4], "model": key[5], "seeds": sorted(r["seed"] for r in rs),
                        "metrics": agg})
        raw = agg["raw"]
        far = f"{agg['far']['RMSE'][0]:.3f}" if "far" in agg else "-"
        exp = fl.RECIPES[key[4]]["exp"] + " " + key[4]
        print(f"{key[0]:<8}{key[1]:>7}{key[2]:>6.2g}{key[3]:>9}  {exp:<12}{key[5]:<7}{len(rs):>3}"
              f"{raw['AbsRel'][0]:>9.3f}±{raw['AbsRel'][1]:.3f}{raw['RMSE'][0]:>8.3f}"
              f"{raw['delta1'][0]:>7.3f}{agg['aligned']['AbsRel'][0]:>9.3f}{far:>10}")
    print(line)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"run": run_info(), "config": vars(args), "protocol": {
        "test_images": len(rgb), "crop": "Eigen [45:471, 41:601] on 480x640", "max_depth": MAX_D,
        "upsampling": "bilinear from 256x192", "ranges_m": RANGES}, "summary": summary, "runs": runs,
        "skipped": skipped}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"Saved: {out / 'report.json'}")


if __name__ == "__main__":
    main()
