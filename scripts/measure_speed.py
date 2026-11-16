#!/usr/bin/env python3
"""
Size and speed of the teacher and the three students: parameters, GFLOPs and latency at batch 1.

  students   256x192 input (the size they are trained and evaluated at)
  teacher    518x686, the input Depth Anything V2 gets for a 640x480 NYU image (short side 518,
             multiple of 14), and 196x252, about the students' input, for a like-for-like number

Latency is the median of repeated forward passes (model only, no pre- or post-processing) on the
chosen device and on the CPU. Run from the repository root:
  python scripts/measure_speed.py --out results/speed
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import copy
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import kd_depth.core as fz  # noqa: E402
from kd_depth.runinfo import run_info  # noqa: E402

TEACHER_ID = "depth-anything/Depth-Anything-V2-Small-hf"


class TeacherNet(torch.nn.Module):
    """The teacher as a plain module: pixel values in, relative inverse depth out."""

    def __init__(self, hf_model):
        super().__init__()
        self.m = hf_model

    def forward(self, x):
        return self.m(pixel_values=x).predicted_depth


def flops(model, shape):
    from torch.utils.flop_counter import FlopCounterMode
    m = copy.deepcopy(model).cpu().eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(torch.zeros(1, 3, *shape))
    return fc.get_total_flops()


@torch.no_grad()
def latency(model, shape, device, warmup, runs):
    m = copy.deepcopy(model).to(device).eval()
    x = torch.rand(1, 3, *shape, device=device)
    for _ in range(warmup):
        m(x)
    fz.sync(device)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        m(x)
        fz.sync(device)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/speed")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--runs", type=int, default=50, help="timed passes on the device (CPU: a fifth)")
    ap.add_argument("--no-teacher", action="store_true", help="students only (no model download)")
    args = ap.parse_args()
    device = fz.pick_device(args.device)
    cpu = torch.device("cpu")

    models = [(f"Student {s}", fz.Student(s, pretrained=False), (fz.H, fz.W))
              for s in ("tiny", "small", "base")]
    if not args.no_teacher:
        from transformers import AutoModelForDepthEstimation
        t = TeacherNet(AutoModelForDepthEstimation.from_pretrained(TEACHER_ID))
        models += [("Teacher (Depth Anything V2 Small)", t, (518, 686)),
                   ("Teacher at the students' size", t, (196, 252))]

    rows = []
    cpu_name = platform.processor() or platform.machine()
    print(f"Device: {device} | CPU threads: {torch.get_num_threads()} | {cpu_name}")
    for name, model, shape in models:
        r = {"model": name, "input_hw": list(shape),
             "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
             "GFLOPs": flops(model, shape) / 1e9,
             "latency_ms_device": latency(model, shape, device, 10, args.runs),
             "latency_ms_cpu": latency(model, shape, cpu, 3, max(5, args.runs // 5))}
        rows.append(r)
        print(f"  {name:<36} {shape[1]}x{shape[0]:<5} {r['params_M']:6.2f}M  {r['GFLOPs']:7.2f} GFLOPs  "
              f"{r['latency_ms_device']:7.1f} ms ({device.type})  {r['latency_ms_cpu']:7.1f} ms (cpu)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"run": run_info(), "config": vars(args), "device": str(device),
              "cpu_threads": torch.get_num_threads(), "rows": rows}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"Saved: {out / 'report.json'}")


if __name__ == "__main__":
    main()
