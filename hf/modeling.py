"""Standalone loader for the kd-depth students (needs only torch, torchvision and safetensors).

    from modeling import load_student
    model = load_student("small-1pct")              # from the Hugging Face repo next to this file
    depth = model(rgb)                               # rgb: (B, 3, 192, 256) in [0, 1] -> (B, 192, 256) in m

The architecture is a copy of `Student` in src/kd_depth/core.py of the project repository;
tests/test_export_hf.py checks that both give the same output.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

H, W = 192, 256            # training resolution (height, width)
MIN_D, MAX_D = 1e-3, 10.0  # output range in meters


def dwsep(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False),
        nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
        nn.Conv2d(cin, cout, 1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class Student(nn.Module):
    SIZES = {"tiny": ("mnv3_small", 16), "small": ("mnv3_large", 32), "base": ("effnet_b0", 32)}

    def __init__(self, size="small"):
        super().__init__()
        from torchvision import models

        backbone, w = self.SIZES[size]
        build = {"mnv3_small": models.mobilenet_v3_small, "mnv3_large": models.mobilenet_v3_large,
                 "effnet_b0": models.efficientnet_b0}[backbone]
        self.enc = build(weights=None).features      # the trained weights are loaded afterwards

        taps = {}
        with torch.no_grad():
            x = torch.zeros(1, 3, H, W)
            for i, layer in enumerate(self.enc):
                x = layer(x)
                taps[H // x.shape[2]] = (i, x.shape[1])
        self.tap_idx = {taps[s][0]: s for s in (2, 4, 8, 16)}

        widths = {32: 4 * w, 16: 4 * w, 8: 2 * w, 4: 2 * w, 2: w, 1: w}
        self.bottleneck = nn.Sequential(
            nn.Conv2d(taps[32][1], widths[32], 1, bias=False),
            nn.BatchNorm2d(widths[32]), nn.ReLU(inplace=True))
        self.ups = nn.ModuleList()
        cin = widths[32]
        for s in (16, 8, 4, 2):
            self.ups.append(dwsep(cin + taps[s][1], widths[s]))
            cin = widths[s]
        self.head = nn.Sequential(dwsep(cin, widths[1]), nn.Conv2d(widths[1], 1, 3, padding=1))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):                     # x: (B, 3, H, W) RGB in [0, 1]
        size = x.shape[-2:]
        x = (x - self.mean) / self.std
        skips = {}
        for i, layer in enumerate(self.enc):
            x = layer(x)
            if i in self.tap_idx:
                skips[self.tap_idx[i]] = x
        y = self.bottleneck(x)
        for s, up in zip((16, 8, 4, 2), self.ups):
            y = F.interpolate(y, size=skips[s].shape[-2:], mode="bilinear", align_corners=False)
            y = up(torch.cat([y, skips[s]], 1))
        y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
        y = self.head(y)[:, 0]
        return MIN_D + (MAX_D - MIN_D) * torch.sigmoid(y)   # metric depth in meters


def load_student(variant, repo_id=None, local_dir=None, device="cpu"):
    """Load `<variant>.safetensors` from `local_dir`, or from the Hub repo `repo_id`."""
    from safetensors.torch import load_file

    if local_dir is None:
        from huggingface_hub import hf_hub_download

        if repo_id is None:
            raise ValueError("give repo_id (Hugging Face repo) or local_dir")
        cfg_path = hf_hub_download(repo_id, "config.json")
        weights = hf_hub_download(repo_id, f"{variant}.safetensors")
    else:
        cfg_path = Path(local_dir) / "config.json"
        weights = Path(local_dir) / f"{variant}.safetensors"
    cfg = json.loads(Path(cfg_path).read_text())["variants"][variant]
    model = Student(cfg["size"])
    model.load_state_dict(load_file(str(weights)))
    return model.to(device).eval()


@torch.no_grad()
def predict(model, image):
    """PIL image or HxWx3 uint8 array -> depth in meters at the image's own size (numpy, HxW)."""
    import numpy as np

    rgb = torch.from_numpy(np.array(image.convert("RGB") if hasattr(image, "convert") else image))  # a copy
    h, w = rgb.shape[:2]
    x = rgb.permute(2, 0, 1)[None].float().div(255)
    x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
    dev = next(model.parameters()).device
    d = model(x.to(dev))[:, None]
    return F.interpolate(d, size=(h, w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
