"""Figure for the README's usage example: one NYU test image and the depth predicted by a shared model.

Runs the released model from Hugging Face through `hf/modeling.py`, exactly as the example in the
README and the model card, on one test image (the Eigen crop, as in the qualitative figure).

    python scripts/example_figure.py --data data/nyu_data --index 131 --out figures/example.png
"""

import argparse
import importlib.util
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
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
    ap.add_argument("--index", type=int, default=131, help="test image index (row of nyu2_test.csv)")
    ap.add_argument("--repo", default="oguzkagangocmez/kd-depth-students", help="Hugging Face repo")
    ap.add_argument("--variant", default="small-1pct")
    ap.add_argument("--out", default="figures/example.png")
    args = ap.parse_args(argv)

    fl = _load("fewlabel_v2", ROOT / "scripts" / "fewlabel_v2.py")
    ev = _load("evaluate_test", ROOT / "scripts" / "evaluate_test.py")
    modeling = _load("modeling", ROOT / "hf" / "modeling.py")

    rows, read, resolve = fl.open_densedepth(args.data, "nyu2_test.csv")
    rgb = np.array(Image.open(io.BytesIO(read(resolve(rows[args.index][0])))).convert("RGB"))[ev.CROP]
    model = modeling.load_student(args.variant, repo_id=args.repo).eval()
    depth = modeling.predict(model, rgb)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("input")
    im = axes[1].imshow(depth, cmap="inferno_r")
    axes[1].set_title(f"{args.variant} prediction")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02, label="depth (m)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"Saved: {args.out} (test image {args.index}, depth {depth.min():.2f}-{depth.max():.2f} m)")


if __name__ == "__main__":
    main()
