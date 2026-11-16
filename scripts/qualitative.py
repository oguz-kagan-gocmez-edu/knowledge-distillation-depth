#!/usr/bin/env python3
"""
Qualitative figure: test images with the real depth, the teacher and the students, plus error maps.

Each row is one test image; the images are evenly spaced over the 654 test images (not picked by
result). Columns: RGB, real depth, teacher (aligned per image to the real depth, as in the test
reference), each student's prediction (metric, no alignment) and, for the first two students,
their AbsRel error per pixel. Everything is shown inside the Eigen crop.

  python scripts/qualitative.py --data data/nyu_data \\
      --ckpt "E1 1%=checkpoints/fix_1_short/gt_lab462_seed0.done.pt" \\
             "E4 1%=checkpoints/fix_1_short/semi_edge_lab462_seed0.done.pt" --out figures
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import importlib.util
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import kd_depth.core as fz  # noqa: E402

_spec = importlib.util.spec_from_file_location("evaluate_test", Path(__file__).with_name("evaluate_test.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)
CROP, MIN_D, MAX_D = ev.CROP, fz.MIN_D, fz.MAX_D


def load_images(path, idx):
    rows, read, resolve = ev.fl.open_densedepth(path, "nyu2_test.csv")
    out = []
    for i in idx:
        rp, dp = rows[i]
        rgb = Image.open(io.BytesIO(read(resolve(rp)))).convert("RGB")
        depth = fz.to_meters(np.array(Image.open(io.BytesIO(read(resolve(dp))))))
        out.append((rgb, depth[CROP].astype(np.float32)))
    return out


@torch.no_grad()
def student_pred(ckpt, rgbs, device):
    r = torch.load(ckpt, weights_only=False, map_location="cpu")
    model = fz.Student(r["signature"]["student"], pretrained=False).to(device).eval()
    model.load_state_dict(r["best_state"])
    x = np.stack([np.asarray(im.resize((fz.W, fz.H), Image.BILINEAR)) for im in rgbs])
    x = torch.from_numpy(x).permute(0, 3, 1, 2).float().div(255).to(device)
    p = F.interpolate(model(x).float().unsqueeze(1), size=(480, 640), mode="bilinear", align_corners=False)
    return p[:, 0, CROP[0], CROP[1]].cpu().numpy()


@torch.no_grad()
def teacher_pred(size, rgbs, depths, device):
    t = fz.DepthAnythingTeacher(size, device)
    inputs = t.proc(images=rgbs, return_tensors="pt").to(device)
    d = F.interpolate(t.model(**inputs).predicted_depth.float().unsqueeze(1), size=(480, 640),
                      mode="bilinear", align_corners=False)[:, 0, CROP[0], CROP[1]].cpu().numpy()
    return np.stack([fz.align_teacher(di, g)[0] for di, g in zip(d, depths)])


def absrel(p, g):
    v = (g > MIN_D) & (g < MAX_D)
    p = np.clip(p, MIN_D, MAX_D)
    err = np.where(v, np.abs(p - g) / np.maximum(g, MIN_D), np.nan)
    return float(np.mean(err[v])), err


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/nyu_data")
    ap.add_argument("--ckpt", nargs="+", required=True, help='"label=path/to/run.done.pt", in column order')
    ap.add_argument("--teacher", default="small", help="teacher size, or 'none'")
    ap.add_argument("--rows", type=int, default=4, help="number of test images (evenly spaced)")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--name", default="qualitative")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = fz.pick_device(args.device)
    ckpts = [c.split("=", 1) for c in args.ckpt]
    if any(len(c) != 2 for c in ckpts):
        raise SystemExit('--ckpt takes "label=path"')

    n_test = len(ev.fl.open_densedepth(args.data, "nyu2_test.csv")[0])
    idx = np.linspace(0, n_test - 1, args.rows + 2).round().astype(int)[1:-1]
    imgs = load_images(args.data, idx)
    rgbs, depths = [im for im, _ in imgs], [d for _, d in imgs]
    cols = [("Real depth", np.stack(depths))]
    if args.teacher != "none":
        cols.append(("Teacher (aligned)", teacher_pred(args.teacher, rgbs, depths, device)))
    cols += [(label, student_pred(path, rgbs, device)) for label, path in ckpts]
    errs = cols[-len(ckpts):][:2]

    n_cols = 1 + len(cols) + len(errs)
    fig, axes = plt.subplots(len(idx), n_cols, figsize=(1.9 * n_cols, 1.55 * len(idx)), squeeze=False)
    for r in range(len(idx)):
        g = depths[r]
        vmin, vmax = np.percentile(g[g > MIN_D], [2, 98])
        ax = axes[r, 0]
        ax.imshow(np.asarray(rgbs[r])[CROP])
        ax.set_ylabel(f"test #{idx[r]}", fontsize=7)
        for c, (_, pred) in enumerate(cols, start=1):
            axes[r, c].imshow(pred[r], cmap="viridis", vmin=vmin, vmax=vmax)
            if c > 1:
                axes[r, c].text(4, 18, f"AbsRel {absrel(pred[r], g)[0]:.3f}", fontsize=6.5, color="white",
                                bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.2, "lw": 0})
        for c, (_, pred) in enumerate(errs, start=1 + len(cols)):
            im = axes[r, c].imshow(absrel(pred[r], g)[1], cmap="magma", vmin=0, vmax=0.5)
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
    titles = ["RGB"] + [label for label, _ in cols] + [f"Error {label}" for label, _ in errs]
    for ax, t in zip(axes[0], titles):
        ax.set_title(t, fontsize=8)
    fig.tight_layout(pad=0.3, w_pad=0.2, h_pad=0.3)
    if errs:
        cb = fig.colorbar(im, ax=axes[:, -len(errs):].ravel().tolist(), fraction=0.04, pad=0.01)
        cb.set_label("|pred − real| / real", fontsize=7)
        cb.ax.tick_params(labelsize=6)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        meta = {"CreationDate": None} if ext == "pdf" else {}
        fig.savefig(out / f"{args.name}.{ext}", dpi=200, bbox_inches="tight", metadata=meta)
    print(f"Saved: {out / args.name}.png/.pdf (test images {', '.join(map(str, idx))})")


if __name__ == "__main__":
    main()
