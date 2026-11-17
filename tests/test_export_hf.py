"""Checks for scripts/export_hf.py and hf/modeling.py (CPU, no downloads)."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("size", ["tiny", "small"])
def test_export_roundtrip(tmp_path, size):
    export = load("export_hf", ROOT / "scripts" / "export_hf.py")   # also puts src/ on the path
    modeling = load("hf_modeling", ROOT / "hf" / "modeling.py")
    H, W, Student = export.H, export.W, export.Student

    torch.manual_seed(0)
    model = Student(size, pretrained=False)
    for m in model.modules():                 # non-trivial BatchNorm statistics
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 1.5)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    done = tmp_path / "semi_edge_lab462_seed0.done.pt"
    torch.save({"best": {"AbsRel": 0.176, "RMSE": 0.7, "delta1": 0.72, "step": 21000},
                "best_state": state, "final_state": state}, done)

    out = tmp_path / "export"
    export.main(["--out", str(out), "--model", f"{size}-1pct={done}"])

    cfg = json.loads((out / "config.json").read_text())
    info = cfg["variants"][f"{size}-1pct"]
    assert info["size"] == size and info["best_step"] == 21000 and info["val"]["AbsRel"] == 0.176
    assert info["parameters"] == sum(p.numel() for p in model.parameters())
    assert (out / "modeling.py").exists() and (out / "README.md").exists()

    hub = modeling.load_student(f"{size}-1pct", local_dir=out)
    x = torch.rand(1, 3, H, W)
    with torch.no_grad():
        assert torch.allclose(model.eval()(x), hub(x), atol=1e-5)
    depth = modeling.predict(hub, (torch.rand(240, 320, 3) * 255).byte().numpy())
    assert depth.shape == (240, 320) and depth.min() >= 1e-3 and depth.max() <= 10.0
