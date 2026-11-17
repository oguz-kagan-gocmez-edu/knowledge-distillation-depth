"""Export trained students to a folder ready to upload to Hugging Face.

Reads the .done.pt files that scripts/fewlabel_v2.py writes into --ckpt-dir, takes the
best-on-validation weights (best_state), and writes

    <out>/<variant>.safetensors   one file per model
    <out>/config.json             student size and metadata per variant
    <out>/modeling.py, README.md  copied from hf/

Each export is checked: the saved weights are loaded into hf/modeling.py's Student and must give
the same output as the project's own Student.

    python scripts/export_hf.py --out hf_export \\
        --model small-1pct=checkpoints/fix_1_short/semi_edge_lab462_seed0.done.pt \\
        --model small-5pct=checkpoints/gtmask_decay/semi_edge_lab2312_seed0.done.pt \\
        --model small-25pct=checkpoints/gtmask_decay/semi_edge_lab11562_seed0.done.pt

Upload (from your own account):  huggingface-cli upload <user>/<repo> hf_export .
"""
import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kd_depth.core import H, Student, W  # noqa: E402

HF_DIR = ROOT / "hf"


def load_modeling():
    spec = importlib.util.spec_from_file_location("kd_hf_modeling", HF_DIR / "modeling.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def infer_size(state):
    """The student size from the weights: the encoder's first conv width and depth tell them apart."""
    for size in Student.SIZES:
        ref = Student(size, pretrained=False).state_dict()
        if ref.keys() == state.keys() and all(ref[k].shape == state[k].shape for k in ref):
            return size
    raise SystemExit("weights do not match any student size")


def export_one(name, done_path, out, modeling):
    from safetensors.torch import load_file, save_file

    d = torch.load(done_path, weights_only=False, map_location="cpu")
    state = {k: v.contiguous() for k, v in d["best_state"].items()}
    size = infer_size(state)
    save_file(state, str(out / f"{name}.safetensors"))

    # the project's model and the standalone copy must agree on the saved weights
    ref = Student(size, pretrained=False).eval()
    ref.load_state_dict(state)
    hub = modeling.Student(size).eval()
    hub.load_state_dict(load_file(str(out / f"{name}.safetensors")))
    x = torch.rand(2, 3, H, W)
    with torch.no_grad():
        diff = (ref(x) - hub(x)).abs().max().item()
    if diff > 1e-5:
        raise SystemExit(f"{name}: standalone model differs from the project's ({diff:.2e})")

    params = sum(p.numel() for p in ref.parameters())
    best = d.get("best") or {}
    return {"size": size, "parameters": params, "source": Path(done_path).name,
            "best_step": best.get("step"),
            "val": {k: round(float(best[k]), 4) for k in ("AbsRel", "RMSE", "delta1") if k in best}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=PATH",
                    help="variant name and its .done.pt file; repeat for several models")
    ap.add_argument("--out", required=True, help="output folder")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    modeling = load_modeling()
    cfg_path = out / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    cfg.setdefault("input", {"height": H, "width": W, "range": "RGB in [0, 1]"})
    cfg.setdefault("output", {"unit": "m", "min": 1e-3, "max": 10.0})
    cfg.setdefault("variants", {})

    for spec in args.model:
        name, _, path = spec.partition("=")
        if not name or not path:
            ap.error(f"--model needs NAME=PATH, got {spec!r}")
        info = export_one(name, path, out, modeling)
        cfg["variants"][name] = info
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"{name}: {info['size']} student, {info['parameters'] / 1e6:.2f}M parameters, "
              f"validation {info['val']} -> {out / (name + '.safetensors')}")

    for f in ("modeling.py", "README.md"):
        shutil.copy(HF_DIR / f, out / f)
    print(f"done: {out}")


if __name__ == "__main__":
    main()
