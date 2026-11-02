"""
Shared code for the CS466 depth-distillation project.

Data loading (Hugging Face stream, DenseDepth NYU zip/folder, synthetic dummy data),
the Depth Anything V2 teacher, the small Student network, losses, metrics,
efficiency measurements and the example figure. The experiment scripts in
scripts/ import from here.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import copy
import csv
import io
import json
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

H, W = 192, 256            # training / evaluation resolution
MIN_D, MAX_D = 1e-3, 10.0  # valid depth range in meters (NYU is indoor, < 10 m)
FULL_TRAIN = 50688         # size of the common NYU training subset (DenseDepth)
RECIPES = {                # weights for (real-depth loss, teacher loss, edge loss)
    "gt": (1.0, 0.0, 0.0),
    "teacher": (0.0, 1.0, 0.0),
    "both": (1.0, 1.0, 0.0),
    "full": (1.0, 1.0, 1.0),
}


# ----------------------------------------------------------------------------
# Device helpers
# ----------------------------------------------------------------------------
def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def mem_gb(device):
    """Memory currently held by the process on this device, in GB."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak / 1e6


# ----------------------------------------------------------------------------
# Data sources: each yields (PIL RGB image, depth in meters as float32 array)
# ----------------------------------------------------------------------------
def as_pil(x):
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, dict) and x.get("bytes") is not None:
        return Image.open(io.BytesIO(x["bytes"]))
    if isinstance(x, (bytes, bytearray)):
        return Image.open(io.BytesIO(x))
    if isinstance(x, np.ndarray):
        return Image.fromarray(x)
    raise TypeError(f"Unknown image type: {type(x)}")


def to_meters(arr):
    """Convert a stored depth map to meters, whatever format it came in."""
    a = np.asarray(arr)
    if a.ndim == 3:
        a = a[..., 0]
    if a.dtype == np.uint8:            # 8-bit PNG: 0..255 maps to 0..10 m
        return a.astype(np.float32) / 255.0 * 10.0
    a = a.astype(np.float32)
    if np.nanmax(a) > 50.0:            # looks like millimeters
        a = a / 1000.0
    return np.nan_to_num(a, nan=0.0)


HF_PATTERNS = [
    "hf://datasets/sayakpaul/nyu_depth_v2@refs%2Fconvert%2Fparquet/default/partial-{split}/*.parquet",
    "hf://datasets/sayakpaul/nyu_depth_v2@~parquet/default/partial-{split}/*.parquet",
]


def iter_hf(split, n, seed):
    """Stream NYU Depth V2 from Hugging Face (parquet copy, no 35 GB download)."""
    from datasets import load_dataset

    errors = []
    for pattern in HF_PATTERNS:
        try:
            ds = load_dataset("parquet", data_files={split: pattern.format(split=split)},
                              split=split, streaming=True)
            if split == "train":
                ds = ds.shuffle(seed=seed, buffer_size=1000)
            it = iter(ds)
            first = next(it)
            break
        except Exception as e:  # try the next way of addressing the files
            errors.append(f"{pattern}: {type(e).__name__}: {e}")
    else:
        raise RuntimeError(
            "Could not stream NYU Depth V2 from Hugging Face. Check your internet "
            "connection, or download the DenseDepth zip and use "
            "--source densedepth --data-zip nyu_data.zip\n" + "\n".join(errors))

    def gen():
        yield first
        yield from it

    for count, ex in enumerate(gen()):
        if count >= n:
            break
        rgb = as_pil(ex["image"]).convert("RGB")
        depth = to_meters(np.array(as_pil(ex["depth_map"])))
        yield rgb, depth


def iter_densedepth(path, split, n, seed):
    """Read the DenseDepth-style NYU data (~4.1 GB): a zip (read without extracting)
    or an extracted folder. Works with the original nyu_data.zip and with the
    Kaggle copy, where everything sits inside an extra nyu_data/ folder."""
    path = Path(path).expanduser()
    if path.is_dir():
        names = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()}
        read = lambda nm: (path / nm).read_bytes()
    elif path.is_file():
        z = zipfile.ZipFile(path)
        names = set(z.namelist())
        read = z.read
    else:
        raise SystemExit(f"--data-zip: {path} does not exist")

    want = "nyu2_train.csv" if split == "train" else "nyu2_test.csv"
    csvs = sorted((nm for nm in names if nm.endswith(want) and "__MACOSX" not in nm), key=len)
    if not csvs:
        raise RuntimeError(f"{want} not found inside {path}")
    csv_name = csvs[0]
    rows = [r for r in csv.reader(io.StringIO(read(csv_name).decode("utf-8"))) if len(r) >= 2]
    random.Random(seed).shuffle(rows)

    # paths in the csv look like "data/nyu2_train/...": find the folder they are relative to
    cut = csv_name.rfind("data/")
    root = csv_name[:cut] if cut >= 0 else csv_name[: csv_name.rfind("/") + 1]

    def resolve(p):
        p = p.strip()
        bare = p.removeprefix("data/")
        for cand in (p, root + p, root + "data/" + bare, "data/" + bare, bare):
            if cand in names:
                return cand
        raise KeyError(f"{p} not found (looked under '{root}')")

    for rgb_path, depth_path in ((r[0], r[1]) for r in rows[:n]):
        rgb = Image.open(io.BytesIO(read(resolve(rgb_path)))).convert("RGB")
        depth = to_meters(np.array(Image.open(io.BytesIO(read(resolve(depth_path))))))
        yield rgb, depth


def iter_dummy(split, n, seed):
    """Synthetic scenes (a sloped room + one object) to test the pipeline offline."""
    rng = np.random.default_rng(seed + (0 if split == "train" else 1000))
    yy, xx = np.mgrid[0:480, 0:640].astype(np.float32)
    yy, xx = yy / 480.0, xx / 640.0
    for _ in range(n):
        a, b = rng.uniform(-1, 1, 2)
        depth = 1.0 + 7.0 * (0.5 + 0.5 * np.tanh(2 * (a * yy + b * xx)))
        cx, cy, r = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8), rng.uniform(0.05, 0.2)
        depth[(xx - cx) ** 2 + (yy - cy) ** 2 < r * r] = rng.uniform(0.8, 2.0)
        shade = np.clip(1.0 - depth / 9.0, 0, 1)
        rgb = np.stack([shade, shade * rng.uniform(0.5, 1.0), 1.0 - shade], -1)
        rgb = (np.clip(rgb + rng.normal(0, 0.03, rgb.shape), 0, 1) * 255).astype(np.uint8)
        depth = depth.astype(np.float32)
        depth[rng.random(depth.shape) < 0.02] = 0.0   # fake sensor holes
        yield Image.fromarray(rgb), depth


def sample_iterator(args, split, n):
    if args.source == "hf":
        return iter_hf("train" if split == "train" else "validation", n, args.seed)
    if args.source == "densedepth":
        if not args.data_zip:
            raise SystemExit("--source densedepth needs --data-zip path/to/nyu_data.zip")
        return iter_densedepth(args.data_zip, split, n, args.seed)
    return iter_dummy(split, n, args.seed)


# ----------------------------------------------------------------------------
# Teacher
# ----------------------------------------------------------------------------
class DepthAnythingTeacher:
    IDS = {
        "small": "depth-anything/Depth-Anything-V2-Small-hf",
        "base": "depth-anything/Depth-Anything-V2-Base-hf",
        "large": "depth-anything/Depth-Anything-V2-Large-hf",
    }

    def __init__(self, size, device):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.name = self.IDS[size]
        self.device = device
        self.proc = AutoImageProcessor.from_pretrained(self.name)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.name).to(device).eval()
        self.params = sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def __call__(self, pils):
        inputs = self.proc(images=pils, return_tensors="pt").to(self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
            out = self.model(**inputs).predicted_depth          # relative inverse depth
        out = F.interpolate(out.unsqueeze(1).float(), size=(H, W),
                            mode="bilinear", align_corners=False)
        return out[:, 0].cpu().numpy()


class DummyTeacher:
    """Fake teacher for --source dummy: a noisy inverse depth with unknown scale/shift."""
    name = "dummy-teacher"
    params = 0

    def __call__(self, pils):
        outs = []
        for p in pils:
            red = np.asarray(p.resize((W, H), Image.BILINEAR), np.float32)[..., 0] / 255.0
            depth_guess = 9.0 * (1.0 - red) + 0.5
            outs.append(3.0 / depth_guess + 0.2)
        return np.stack(outs)


def align_teacher(disp, depth):
    """Fit scale s and shift t so that s*disp + t matches 1/depth (least squares)."""
    valid = (depth > MIN_D) & (depth < MAX_D)
    if valid.sum() < 100:
        return np.clip(1.0 / np.clip(disp, 1e-3, None), 0.1, MAX_D).astype(np.float32), False
    x = disp[valid].astype(np.float64)
    y = 1.0 / depth[valid].astype(np.float64)
    s, t = np.linalg.lstsq(np.stack([x, np.ones_like(x)], 1), y, rcond=None)[0]
    inv = s * disp + t
    return (1.0 / np.clip(inv, 1.0 / MAX_D, 10.0)).astype(np.float32), bool(s > 0)


# ----------------------------------------------------------------------------
# Loading pass: fetch images, run teacher, store everything at H x W
# ----------------------------------------------------------------------------
def build_split(args, split, n, teacher, device, cache_dir):
    cache = cache_dir / f"{args.source}_{split}_{n}_{args.teacher}.npz"
    if cache.exists() and not args.no_cache:
        d = np.load(cache, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        print(f"  [{split}] loaded {len(d['rgb'])} samples from cache {cache.name}")
        return {"rgb": d["rgb"], "depth": d["depth"], "teacher": d["teacher"]}, meta

    rgbs = np.zeros((n, H, W, 3), np.uint8)
    depths = np.zeros((n, H, W), np.float32)
    teach = np.zeros((n, H, W), np.float32)
    batch, k, bad = [], 0, 0
    fetch_s, batch_times = 0.0, []

    def flush():
        nonlocal k, bad
        sync(device)
        t0 = time.perf_counter()
        disp = teacher([b[0] for b in batch])
        sync(device)
        batch_times.append((time.perf_counter() - t0, len(batch)))
        for (rgb, depth), d in zip(batch, disp):
            rgbs[k] = np.asarray(rgb.resize((W, H), Image.BILINEAR))
            depths[k] = np.asarray(Image.fromarray(depth).resize((W, H), Image.NEAREST))
            teach[k], ok = align_teacher(d, depths[k])
            bad += not ok
            k += 1
        batch.clear()

    t_start = time.perf_counter()
    t0 = time.perf_counter()
    for rgb, depth in sample_iterator(args, split, n):
        fetch_s += time.perf_counter() - t0
        batch.append((rgb, depth))
        if len(batch) == args.teacher_batch:
            flush()
            if k % 200 < args.teacher_batch:
                print(f"  [{split}] {k}/{n} images ({time.perf_counter() - t_start:.0f}s)")
        t0 = time.perf_counter()
    if batch:
        flush()
    if k < n:
        print(f"  [{split}] only {k} samples were available (asked for {n})")
    rgbs, depths, teach = rgbs[:k], depths[:k], teach[:k]

    # skip the first teacher batch when timing (it includes warm-up)
    timed = batch_times[1:] if len(batch_times) > 1 else batch_times
    teacher_s_per_img = sum(t for t, _ in timed) / max(1, sum(c for _, c in timed))
    meta = {
        "n": int(k),
        "fetch_s_per_img": fetch_s / max(1, k),
        "teacher_s_per_img": teacher_s_per_img,
        "teacher_align_failures": int(bad),
    }
    np.savez(cache, rgb=rgbs, depth=depths, teacher=teach.astype(np.float16).astype(np.float32),
             meta=json.dumps(meta))
    return {"rgb": rgbs, "depth": depths, "teacher": teach}, meta


# ----------------------------------------------------------------------------
# Student: MobileNetV3 encoder + light FastDepth-style decoder
# ----------------------------------------------------------------------------
def dwsep(cin, cout):
    """Depthwise-separable conv block (cheap 3x3 conv)."""
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False),
        nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
        nn.Conv2d(cin, cout, 1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class Student(nn.Module):
    # name: (encoder, decoder width)  ->  about 1.0M / 3.2M / 4.2M parameters
    SIZES = {"tiny": ("mnv3_small", 16), "small": ("mnv3_large", 32), "base": ("effnet_b0", 32)}

    def __init__(self, size="small", pretrained=True):
        super().__init__()
        from torchvision import models

        backbone, w = self.SIZES[size]
        build, weights_enum = {
            "mnv3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights),
            "mnv3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights),
            "effnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights),
        }[backbone]
        weights = weights_enum.DEFAULT if pretrained else None
        try:
            self.enc = build(weights=weights).features
        except Exception as e:
            print(f"  ! could not load ImageNet weights ({e}); using random init")
            self.enc = build(weights=None).features

        # find the last encoder layer at each stride (2, 4, 8, 16, 32)
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

    def forward(self, x):                     # x: (B, 3, H, W) in [0, 1]
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
        return MIN_D + (MAX_D - MIN_D) * torch.sigmoid(y)   # depth in meters


# ----------------------------------------------------------------------------
# Losses and metrics
# ----------------------------------------------------------------------------
def silog_loss(pred, gt, mask, lam=0.85):
    """Scale-invariant log loss (Eigen et al.), on valid sensor pixels only."""
    d = torch.log(pred[mask]) - torch.log(gt[mask])
    if d.numel() == 0:
        return pred.sum() * 0.0
    return torch.sqrt((d ** 2).mean() - lam * d.mean() ** 2 + 1e-8)


def edge_loss(pred, target):
    """Match horizontal and vertical depth gradients (sharp object borders)."""
    dx = (pred[:, :, 1:] - pred[:, :, :-1]) - (target[:, :, 1:] - target[:, :, :-1])
    dy = (pred[:, 1:, :] - pred[:, :-1, :]) - (target[:, 1:, :] - target[:, :-1, :])
    return dx.abs().mean() + dy.abs().mean()


def depth_metrics(preds, gts):
    """Standard metrics, computed per image and then averaged."""
    absrel, rmse, d1 = [], [], []
    for p, g in zip(preds, gts):
        v = (g > MIN_D) & (g < MAX_D)
        if v.sum() == 0:
            continue
        p, g = np.clip(p[v], MIN_D, MAX_D), g[v]
        absrel.append(np.mean(np.abs(p - g) / g))
        rmse.append(np.sqrt(np.mean((p - g) ** 2)))
        d1.append(np.mean(np.maximum(p / g, g / p) < 1.25))
    return {"AbsRel": float(np.mean(absrel)), "RMSE": float(np.mean(rmse)),
            "delta1": float(np.mean(d1))}


@torch.no_grad()
def predict(model, rgb, device, bs=16):
    model.eval()
    out = []
    for i in range(0, len(rgb), bs):
        x = torch.from_numpy(rgb[i:i + bs]).permute(0, 3, 1, 2).float().div(255).to(device)
        out.append(model(x).float().cpu().numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------------------
# Efficiency measurements
# ----------------------------------------------------------------------------
def count_flops(model):
    from torch.utils.flop_counter import FlopCounterMode
    m = copy.deepcopy(model).cpu().eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(torch.zeros(1, 3, H, W))
    return fc.get_total_flops()


@torch.no_grad()
def latency_ms(model, device, warmup=10, runs=50):
    m = copy.deepcopy(model).to(device).eval()
    x = torch.rand(1, 3, H, W, device=device)
    for _ in range(warmup):
        m(x)
    sync(device)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        m(x)
        sync(device)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


# ----------------------------------------------------------------------------
# Training one recipe
# ----------------------------------------------------------------------------
def train_recipe(name, train, args, device):
    w_gt, w_t, w_e = RECIPES[name]
    w_e *= args.lambda_edge
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    model = Student(args.student, pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=0.1)
    rgb = torch.from_numpy(train["rgb"])
    depth = torch.from_numpy(train["depth"])
    teach = torch.from_numpy(train["teacher"])
    n, bs = len(rgb), args.batch_size

    losses, step_times, peak = [], [], 0.0
    model.train()
    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        idx = torch.randint(0, n, (bs,), generator=gen)
        x = rgb[idx].permute(0, 3, 1, 2).float().div(255)
        d, t = depth[idx].clone(), teach[idx].clone()
        flip = torch.rand(bs, generator=gen) < 0.5          # random horizontal flip
        x[flip], d[flip], t[flip] = x[flip].flip(-1), d[flip].flip(-1), t[flip].flip(-1)
        x, d, t = x.to(device), d.to(device), t.to(device)

        pred = model(x)
        mask = (d > MIN_D) & (d < MAX_D)
        log_pred, log_t = torch.log(pred), torch.log(t)   # log space keeps losses on one scale
        loss = pred.sum() * 0.0
        if w_gt:
            loss = loss + w_gt * silog_loss(pred, d, mask)
        if w_t:
            loss = loss + w_t * (log_pred - log_t).abs().mean()
        if w_e:
            loss = loss + w_e * edge_loss(log_pred, log_t)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())                          # .item() also syncs
        step_times.append(time.perf_counter() - t0)
        if step % 50 == 0 or step == args.steps:
            peak = max(peak, mem_gb(device))
        if step % max(1, args.steps // 10) == 0:
            print(f"  [{name}] step {step}/{args.steps}  loss {np.mean(losses[-50:]):.4f}")

    timed = step_times[10:] if len(step_times) > 20 else step_times
    return model, {
        "final_loss": float(np.mean(losses[-50:])),
        "s_per_step": float(np.median(timed)),
        "peak_mem_gb": float(peak),
    }


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def save_figure(path, val, teacher_val, preds, k=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = np.linspace(0, len(val["rgb"]) - 1, min(k, len(val["rgb"]))).astype(int)
    cols = [("Image", None), ("Real depth (sensor)", None), ("Teacher (aligned)", teacher_val)]
    cols += [(f"Student: {name}", p) for name, p in preds.items()]
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#9e9e9e")                                  # missing sensor pixels

    fig, axes = plt.subplots(len(idx), len(cols), figsize=(2.6 * len(cols), 2.1 * len(idx)),
                             squeeze=False, constrained_layout=True)
    for r, i in enumerate(idx):
        gt = val["depth"][i]
        valid = (gt > MIN_D) & (gt < MAX_D)
        vmin, vmax = np.percentile(gt[valid], [2, 98]) if valid.any() else (0, MAX_D)
        im = None
        for c, (title, arr) in enumerate(cols):
            ax = axes[r, c]
            if c == 0:
                ax.imshow(val["rgb"][i])
            elif c == 1:
                im = ax.imshow(np.ma.masked_where(~valid, gt), cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                ax.imshow(arr[i], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=9)
        cb = fig.colorbar(im, ax=axes[r, -1], fraction=0.046, pad=0.02)
        cb.set_label("depth (m)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.savefig(path, dpi=120)
    plt.close(fig)
