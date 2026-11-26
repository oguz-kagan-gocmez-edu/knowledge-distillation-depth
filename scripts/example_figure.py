"""Figure for the README's usage example: one NYU test image and the depth predicted by the shared models.

Runs the released models from Hugging Face through `hf/modeling.py`, exactly as the example in the
README and the model card, on one test image (the Eigen crop, as in the qualitative figure). The
teacher (Depth Anything V2 Small) predicts relative depth only, so it is drawn after a per-image
scale and shift to the real depth, as in the qualitative figure; the students need no alignment.

    python scripts/example_figure.py --data data/nyu_data --index 432 --out figures/example.png
"""

import argparse
import importlib.util
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="nyu_data.zip or its extracted folder")
    ap.add_argument("--index", type=int, default=432, help="test image index (row of nyu2_test.csv)")
    ap.add_argument("--repo", default="oguzkagangocmez/kd-depth-students", help="Hugging Face repo")
    ap.add_argument("--variants", default="small-1pct,small-5pct,small-25pct", help="comma-separated")
    ap.add_argument("--teacher", default="small", help="teacher size, or '' to leave it out")
    ap.add_argument("--out", default="figures/example.png")
    args = ap.parse_args(argv)

    fl = _load("fewlabel_v2", ROOT / "scripts" / "fewlabel_v2.py")
    ev = _load("evaluate_test", ROOT / "scripts" / "evaluate_test.py")
    qual = _load("qualitative", ROOT / "scripts" / "qualitative.py")
    modeling = _load("modeling", ROOT / "hf" / "modeling.py")

    rows, read, resolve = fl.open_densedepth(args.data, "nyu2_test.csv")
    rgb_path, depth_path = rows[args.index]
    full = Image.open(io.BytesIO(read(resolve(rgb_path)))).convert("RGB")
    rgb = np.array(full)[ev.CROP]
    variants = args.variants.split(",")
    depths = [modeling.predict(modeling.load_student(v, repo_id=args.repo).eval(), rgb) for v in variants]
    titles = list(variants)
    if args.teacher:
        real = qual.fz.to_meters(np.array(Image.open(io.BytesIO(read(resolve(depth_path))))))[ev.CROP]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        teacher = qual.teacher_pred(args.teacher, [full], [real.astype(np.float32)], device)[0]
        depths.insert(0, teacher)
        titles.insert(0, f"teacher ({args.teacher}, aligned)")
    lo, hi = min(d.min() for d in depths), max(d.max() for d in depths)   # one color scale for all

    fig, axes = plt.subplots(1, 1 + len(depths), figsize=(3.6 * (1 + len(depths)), 3.0),
                             constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("input")
    for ax, title, d in zip(axes[1:], titles, depths):
        im = ax.imshow(d, cmap="inferno_r", vmin=lo, vmax=hi)
        ax.set_title(title)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.02, label="depth (m)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"Saved: {args.out} (test image {args.index}, depth {lo:.2f}-{hi:.2f} m)")


if __name__ == "__main__":
    main()
