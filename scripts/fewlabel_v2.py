#!/usr/bin/env python3
"""
CS466 term project - few-label experiment, v2 (E1-E4 at one label budget, optional E5 sweep)

v2 changes: teacher weight ramp-up, gradient clipping, scale column, "mixed" verdict.

Question: when only a few training images have real depth, does learning from the
teacher on the other images help the small student?

Recipes (same student, same number of steps, best checkpoint on validation):
  gt         E1  real depth only, on the labeled images
  gt_t       E2  real depth + teacher, on the labeled images only
  semi       E3  real depth on the labeled images + teacher on ALL pool images
  semi_edge  E4  E3 + edge (gradient-matching) loss          <- full recipe

The teacher term never uses real depth: it is scale-and-shift invariant
(Ranftl et al., MiDaS), so it also works on unlabeled images.
Validation uses held-out training SCENES, never the 654 test images.

Run from the repository root:
  python scripts/fewlabel_v2.py --quick         # ~3 min smoke test
  python scripts/fewlabel_v2.py                 # E1-E4, 250 of 5000 images labeled (5%), ~40 min
  python scripts/fewlabel_v2.py --labeled 1%,5%,25%,100% --recipes gt,semi_edge
                                      # mini E5 sweep (1/5/25/100%), ~75 min
Data defaults to data/nyu_data (zip or folder). Results go to results/fewlabel_v2/,
the data + teacher cache to cache/.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import csv
import io
import json
import random
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import kd_depth.core as fz  # noqa: E402
from kd_depth import poolcache  # noqa: E402
from kd_depth.core import MAX_D, MIN_D  # noqa: E402
from kd_depth.runinfo import run_info  # noqa: E402

RECIPES = {
    "gt":        dict(teacher=False, pool="labeled", edge=False, exp="E1"),
    "gt_t":      dict(teacher=True,  pool="labeled", edge=False, exp="E2"),
    "semi":      dict(teacher=True,  pool="all",     edge=False, exp="E3"),
    "semi_edge": dict(teacher=True,  pool="all",     edge=True,  exp="E4"),
}


# ----------------------------------------------------------------------------
# Data: scene-based split of the DenseDepth training set
# ----------------------------------------------------------------------------
def open_densedepth(path, csv_file="nyu2_train.csv"):
    path = Path(path).expanduser()
    if path.is_dir():
        names = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()}
        read = lambda nm: (path / nm).read_bytes()
    elif path.is_file():
        z = zipfile.ZipFile(path)
        names = set(z.namelist())
        read = z.read
    else:
        raise SystemExit(f"--data: {path} does not exist")
    csvs = sorted((nm for nm in names if nm.endswith(csv_file) and "__MACOSX" not in nm), key=len)
    if not csvs:
        raise SystemExit(f"{csv_file} not found inside {path}")
    csv_name = csvs[0]
    rows = [r[:2] for r in csv.reader(io.StringIO(read(csv_name).decode("utf-8"))) if len(r) >= 2]
    cut = csv_name.rfind("data/")
    root = csv_name[:cut] if cut >= 0 else csv_name[: csv_name.rfind("/") + 1]

    def resolve(p):
        p = p.strip()
        bare = p.removeprefix("data/")
        for cand in (p, root + p, root + "data/" + bare, "data/" + bare, bare):
            if cand in names:
                return cand
        raise KeyError(f"{p} not found (looked under '{root}')")

    return rows, read, resolve


def scene_of(rgb_path):
    """'data/nyu2_train/kitchen_0028b_out/12.jpg' -> 'kitchen_0028' (a/b parts = same room)."""
    folder = rgb_path.strip().split("/")[-2]
    return re.sub(r"[a-z]?_out$", "", folder)


def scene_split(rows, val_frac, seed):
    by_scene = {}
    for r in rows:
        by_scene.setdefault(scene_of(r[0]), []).append(r)
    scenes = sorted(by_scene)
    random.Random(seed).shuffle(scenes)
    n_val = max(1, round(len(scenes) * val_frac))
    train = [r for s in scenes[n_val:] for r in by_scene[s]]
    val = [r for s in scenes[:n_val] for r in by_scene[s]]
    return train, val, len(scenes), n_val


def iter_rows(rows, read, resolve):
    for rgb_path, depth_path in rows:
        rgb = Image.open(io.BytesIO(read(resolve(rgb_path)))).convert("RGB")
        depth = fz.to_meters(np.array(Image.open(io.BytesIO(read(resolve(depth_path))))))
        yield rgb, depth


def cache_dirs(root, tag, n_pool, n_val, seed):
    """Folders for the pool and validation caches.

    The pool does not depend on the validation size, so changing --n-val keeps the pool's
    teacher outputs. Folders from the older layout (<tag>_pool<N>_val<M>_seed<S>/{pool,val})
    are still used when they exist; meta.json checks that they hold the same images.
    """
    base = root / f"{tag}_pool{n_pool}_seed{seed}"
    pool, val = base / "pool", base / f"val{n_val}"
    if not pool.exists():
        legacy = sorted(root.glob(f"{tag}_pool{n_pool}_val*_seed{seed}/pool"))
        if legacy:
            pool = legacy[0]
    legacy_val = root / f"{tag}_pool{n_pool}_val{n_val}_seed{seed}" / "val"
    if not val.exists() and legacy_val.exists():
        val = legacy_val
    return pool, val


def build_data(args, device):
    """Pool (labeled + unlabeled training images) and validation set, with teacher outputs.

    Old single-file caches (.npz, all in RAM) are still read, so past runs stay reproducible.
    New caches are folders of .npy files read from disk on demand (kd_depth.poolcache),
    so the full ~45K-image pool works with little RAM.
    """
    dummy = args.data == "dummy"
    tag = "dummy" if dummy else "densedepth"
    old = Path(args.cache_dir) / f"{tag}_pool{args.pool}_val{args.n_val}_seed{args.seed}_{args.teacher}.npz"
    if old.exists() and not args.no_cache:
        d = np.load(old)
        print(f"  loaded data + teacher outputs from cache {old.name}")
        meta = json.loads(str(d["meta"]))
        pool = {k: d[f"pool_{k}"] for k in ("rgb", "depth", "disp")}
        val = {k: d[f"val_{k}"] for k in ("rgb", "depth", "disp")}
        return pool, val, meta

    loaded = []                                 # load the teacher once, and only if needed

    def make_teacher():
        if not loaded:
            loaded.append(fz.DummyTeacher() if dummy else fz.DepthAnythingTeacher(args.teacher, device))
            print(f"  teacher: {loaded[0].name}")
        return loaded[0]

    if dummy:
        if args.pool < 0:
            raise SystemExit("--pool -1 (all images) needs real data")
        n_pool, n_val = args.pool, args.n_val
        meta = {"scenes": None}
        pool_src = lambda: fz.iter_dummy("train", n_pool, args.seed)
        val_src = lambda: fz.iter_dummy("val", n_val, args.seed)
        pool_key = poolcache.list_hash(["dummy-train", n_pool, args.seed])
        val_key = poolcache.list_hash(["dummy-val", n_val, args.seed])
    else:
        rows, read, resolve = open_densedepth(args.data)
        train_rows, val_rows, n_sc, n_val_sc = scene_split(rows, args.val_scene_frac, args.seed)
        rng = random.Random(args.seed)
        rng.shuffle(train_rows)
        rng.shuffle(val_rows)
        pool_rows = train_rows if args.pool < 0 else train_rows[:args.pool]
        val_rows = val_rows[:args.n_val]
        n_pool, n_val = len(pool_rows), len(val_rows)
        print(f"  {len(rows)} images in {n_sc} rooms -> {n_val_sc} rooms held out for validation; "
              f"pool {n_pool}, val {n_val}")
        meta = {"rooms": n_sc, "val_rooms": n_val_sc, "pool": n_pool, "val": n_val}
        pool_src = lambda: iter_rows(pool_rows, read, resolve)
        val_src = lambda: iter_rows(val_rows, read, resolve)
        pool_key, val_key = poolcache.list_hash(pool_rows), poolcache.list_hash(val_rows)

    pool_dir, val_dir = cache_dirs(Path(args.cache_dir), tag, n_pool, n_val, args.seed)
    if args.no_cache:
        for d in (pool_dir, val_dir):
            if d.exists():
                shutil.rmtree(d)
    pool = poolcache.build(pool_dir, n_pool, pool_src, pool_key, make_teacher, args.teacher,
                           args.teacher_batch, "pool")
    val = poolcache.build(val_dir, n_val, val_src, val_key, make_teacher, args.teacher,
                          args.teacher_batch, "val")
    val = {k: np.asarray(v) for k, v in val.items()}           # small: keep in RAM
    loaded.clear()                                             # free the teacher's memory
    return pool, val, meta


# ----------------------------------------------------------------------------
# Teacher loss (scale-and-shift invariant, no real depth needed) + edge loss
# ----------------------------------------------------------------------------
def ssi_normalize(x, eps=1e-6):
    """Per image: subtract the median, divide by the mean absolute deviation."""
    b = x.shape[0]
    f = x.reshape(b, -1)
    med = f.median(dim=1, keepdim=True).values
    s = (f - med).abs().mean(dim=1, keepdim=True).clamp_min(eps)
    return ((f - med) / s).reshape_as(x)


def teacher_terms(pred, tdisp, with_edge, scales=4, w=None):
    """w: optional weight per image (the batch mean is taken over w * loss of each image)."""
    pdisp = 1.0 / pred.clamp(min=0.1)                 # student depth -> inverse depth
    a, b = ssi_normalize(pdisp), ssi_normalize(tdisp)
    avg = (lambda v: v.mean()) if w is None else (lambda v: (v.flatten(1).mean(1) * w).mean())
    l_t = avg((a - b).abs())
    if not with_edge:
        return l_t, None
    r = (a - b).unsqueeze(1)                          # gradient matching (MiDaS style)
    l_e = 0.0
    for s in range(scales):
        if s:
            r = F.avg_pool2d(r, 2)
        l_e = l_e + avg((r[..., :, 1:] - r[..., :, :-1]).abs()) \
                  + avg((r[..., 1:, :] - r[..., :-1, :]).abs())
    return l_t, l_e / scales


def scale_ratio(preds, gts):
    """Median of (median predicted depth / median real depth) over images.
    1.0 = right scale; 1.8 = the student thinks everything is 1.8x farther."""
    r = []
    for p, g in zip(preds, gts):
        v = (g > MIN_D) & (g < MAX_D)
        if v.any():
            r.append(np.median(p[v]) / np.median(g[v]))
    return float(np.median(r))


# ----------------------------------------------------------------------------
# Training one (recipe, label budget, seed)
# ----------------------------------------------------------------------------
METRICS = ("AbsRel", "RMSE", "delta1", "delta2", "delta3", "log10")


def aggregate(f, metrics):
    """f (mean or std) over runs, per metric. Runs finished with older code have fewer metrics."""
    keys = [k for k in METRICS if all(k in m for m in metrics)]
    return {k: float(f([m[k] for m in metrics])) for k in keys}


def run_name(recipe, n_lab, seed):
    return f"{recipe}_lab{n_lab}_seed{seed}"


def run_signature(recipe, n_lab, seed, n_pool, args):
    """Everything that changes a run's result: a checkpoint is only resumed if this matches."""
    keys = ("student", "teacher", "steps", "eval_every", "batch_size", "lr", "lambda_t",
            "lambda_e", "ramp", "clip", "no_pretrained", "data", "n_val", "val_scene_frac", "seed")
    sig = {"recipe": recipe, "n_lab": n_lab, "train_seed": seed, "n_pool": n_pool,
           **{k: getattr(args, k) for k in keys}}
    # added later: only stored when they differ from the default, so older checkpoints still resume
    for k, default in OPTIONAL_DEFAULTS.items():
        if getattr(args, k, default) != default:
            sig[k] = getattr(args, k)
    if getattr(args, "aug", False):
        sig["aug_scale"] = args.aug_scale
    return sig


OPTIONAL_DEFAULTS = {"teacher_decay": 0.0, "label_subset": 0, "aug": False,
                     "weight_decay": 1e-4, "dropout": 0.0, "pool_gt": False, "lambda_t_lab": 1.0}


def augment(x, d, t, gen, max_scale=1.25):
    """Random zoom-in crop and color jitter on a batch (already on its device).

    x: (B, 3, H, W) RGB in [0, 1]; d: (B, H, W) real depth in meters, 0 = no label;
    t: (B, H, W) teacher disparity. Zooming in by s makes the scene look s times closer,
    so the real depth is divided by s (Eigen et al. 2014). The teacher target is scale-and-shift
    invariant, so it is only cropped. Color changes only affect the student's input, as in
    Depth Anything's perturbed student. The crop never leaves the image, so there are no borders.
    """
    b, dev = x.shape[0], x.device
    u = lambda lo, hi, *shape: (lo + (hi - lo) * torch.rand(b, *shape, generator=gen)).to(dev)
    s = u(1.0, max_scale)
    room = 1 - 1 / s                                       # how far the crop can move
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0] = theta[:, 1, 1] = 1 / s
    theta[:, 0, 2], theta[:, 1, 2] = u(-1, 1) * room, u(-1, 1) * room
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    d = F.grid_sample(d[:, None].float(), grid, mode="nearest", padding_mode="zeros",
                      align_corners=False)[:, 0] / s.view(b, 1, 1)
    t = F.grid_sample(t[:, None], grid, mode="bilinear", padding_mode="border",
                      align_corners=False)[:, 0]
    gamma, bright, color = u(0.9, 1.1, 1, 1, 1), u(0.75, 1.25, 1, 1, 1), u(0.9, 1.1, 3, 1, 1)
    x = (x.clamp(0, 1) ** gamma * bright * color).clamp(0, 1)
    return x, d, t


def labeled_indices(n_lab, n_pool, subset):
    """Pool positions of the labeled images. Subsets are nested: 1% is inside 5% for the same k."""
    if subset == 0:
        return torch.arange(n_lab)
    order = np.random.RandomState(subset).permutation(n_pool)
    return torch.from_numpy(order[:n_lab].copy())


def teacher_weight(step, steps, ramp, decay):
    """Ramps up over the first `ramp` of training, stays at 1, then falls linearly to 0
    over the last `decay` of training."""
    w = min(1.0, step / (ramp * steps)) if ramp > 0 else 1.0
    if decay > 0:
        w *= min(1.0, max(0.0, (steps - step) / (decay * steps)))
    return w


def _rng_states():
    st = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        st["mps"] = torch.mps.get_rng_state()
    return st


def _set_rng_states(st):
    torch.set_rng_state(st["cpu"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])
    if "mps" in st and torch.backends.mps.is_available():
        torch.mps.set_rng_state(st["mps"])


def _save(obj, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)                      # never leaves a half-written checkpoint


def train_run(recipe, n_lab, seed, pool, val, args, device):
    cfg = RECIPES[recipe]
    n_pool, bs = len(pool["rgb"]), args.batch_size
    sig = run_signature(recipe, n_lab, seed, n_pool, args)
    ckpt = done = None
    if args.ckpt_dir:
        cdir = Path(args.ckpt_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        ckpt = cdir / f"{run_name(recipe, n_lab, seed)}.pt"
        done = cdir / f"{run_name(recipe, n_lab, seed)}.done.pt"
        if done.exists():
            r = torch.load(done, weights_only=False)
            if r["signature"] != sig:
                raise SystemExit(f"{done} was made with other settings; use another --ckpt-dir")
            print(f"    already finished (from {done.name}), skipping")
            return r["best"], r["history"], r["preds"].astype(np.float32), r["seconds"]

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = fz.Student(args.student, pretrained=not args.no_pretrained,
                       dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=max(0.1, 3 / args.steps))
    rgb, depth, disp = pool["rgb"], pool["depth"], pool["disp"]   # numpy, maybe memory-mapped
    lab = labeled_indices(n_lab, n_pool, args.label_subset)

    best, best_state, hist, logs = None, None, [], {"gt": [], "teacher": [], "edge": []}
    start, spent = 1, 0.0
    if ckpt is not None and ckpt.exists():
        c = torch.load(ckpt, weights_only=False, map_location="cpu")
        if c["signature"] != sig:
            raise SystemExit(f"{ckpt} was made with other settings; use another --ckpt-dir")
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"])
        gen.set_state(c["gen"])
        _set_rng_states(c["rng"])
        best, best_state, hist, logs = c["best"], c["best_state"], c["history"], c["logs"]
        start, spent = c["step"] + 1, c["seconds"]
        print(f"    resuming from step {c['step']} ({ckpt.name})")

    def checkpoint(step):
        _save({"signature": sig, "step": step, "seconds": spent + time.perf_counter() - t_start,
               "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
               "gen": gen.get_state(), "rng": _rng_states(), "best": best,
               "best_state": best_state, "history": hist, "logs": logs}, ckpt)

    t_start = time.perf_counter()
    for step in range(start, args.steps + 1):
        model.train()
        if cfg["pool"] == "all":                       # half labeled, half from all images
            n_l = bs // 2
            idx = torch.cat([lab[torch.randint(0, n_lab, (n_l,), generator=gen)],
                             torch.randint(0, n_pool, (bs - n_l,), generator=gen)])
            is_lab = torch.arange(bs) < n_l
        else:
            idx = lab[torch.randint(0, n_lab, (bs,), generator=gen)]
            is_lab = torch.ones(bs, dtype=torch.bool)
        ii = idx.numpy()
        x = torch.from_numpy(rgb[ii]).permute(0, 3, 1, 2).float().div(255)
        d, t = torch.from_numpy(depth[ii]), torch.from_numpy(disp[ii]).float()
        flip = torch.rand(bs, generator=gen) < 0.5
        x[flip], d[flip], t[flip] = x[flip].flip(-1), d[flip].flip(-1), t[flip].flip(-1)
        if args.pool_gt:                               # pool images that are labeled count as labeled
            is_lab = torch.isin(idx, lab)
        x, d, t, is_lab = x.to(device), d.to(device), t.to(device), is_lab.to(device)
        if args.aug:
            x, d, t = augment(x, d, t, gen, args.aug_scale)

        pred = model(x)
        mask = (d > MIN_D) & (d < MAX_D) & is_lab.view(-1, 1, 1)
        l_gt = fz.silog_loss(pred, d, mask)
        loss = l_gt
        if cfg["teacher"]:
            # ramp-up: the student first learns real depth, then the teacher term grows
            ramp = teacher_weight(step, args.steps, args.ramp, args.teacher_decay)
            w = None
            if args.lambda_t_lab != 1.0:               # teacher weight on images with real depth
                w = torch.where(is_lab, args.lambda_t_lab, 1.0).float()
            l_t, l_e = teacher_terms(pred, t, cfg["edge"], w=w)
            loss = loss + ramp * args.lambda_t * l_t
            logs["teacher"].append(l_t.item())
            if cfg["edge"]:
                loss = loss + ramp * args.lambda_e * l_e
                logs["edge"].append(l_e.item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        sched.step()
        logs["gt"].append(l_gt.item())

        if step == start + 49:
            el = time.perf_counter() - t_start
            print(f"    ~{el / 50 * (args.steps - start + 1) / 60:.0f} min left for this run")
        if step % args.eval_every == 0 or step == args.steps:
            p_val = fz.predict(model, val["rgb"], device)
            m = fz.depth_metrics(p_val, val["depth"])
            m["scale"] = scale_ratio(p_val, val["depth"])
            m["step"] = step
            hist.append(m)
            if best is None or m["AbsRel"] < best["AbsRel"]:
                best = dict(m)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            parts = "  ".join(f"{k} {np.mean(v[-args.eval_every:]):.3f}" for k, v in logs.items() if v)
            print(f"    step {step:>5}  {parts}  | val AbsRel {m['AbsRel']:.3f}  "
                  f"d1 {m['delta1']:.3f}  scale x{m['scale']:.2f}", flush=True)
        if ckpt is not None and step < args.steps and (
                step % args.ckpt_every == 0 or step == args.stop_after):
            checkpoint(step)
        if step == args.stop_after:
            print(f"stopped after step {step} (--stop-after); run again to resume")
            raise SystemExit(0)

    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    preds = fz.predict(model, val["rgb"], device)
    seconds = spent + time.perf_counter() - t_start
    if done is not None:
        # keep both models so they can be evaluated on the test set later
        _save({"signature": sig, "best": best, "history": hist, "seconds": seconds,
               "preds": preds.astype(np.float16), "best_state": best_state,
               "final_state": final_state}, done)
        ckpt.unlink(missing_ok=True)
    return best, hist, preds, seconds


def parse_budgets(spec, n_pool):
    """"250,5%" -> sorted label counts; percentages are of the pool size."""
    out = set()
    for part in spec.split(","):
        part = part.strip()
        n = round(float(part[:-1]) / 100 * n_pool) if part.endswith("%") else int(part)
        out.add(max(1, min(n, n_pool)))
    return sorted(out)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/nyu_data", help="DenseDepth zip/folder, or 'dummy'")
    ap.add_argument("--pool", type=int, default=5000,
                    help="training images (labeled + unlabeled); -1 = all training images")
    ap.add_argument("--labeled", default="250",
                    help="labeled images: count or percent of the pool (250, 5%%); comma list = sweep")
    ap.add_argument("--n-val", type=int, default=1000)   # runs before Nov 2026 used 300
    ap.add_argument("--val-scene-frac", type=float, default=0.1)
    ap.add_argument("--recipes", default="gt,gt_t,semi,semi_edge")
    ap.add_argument("--seeds", default="0", help="comma list; >1 seed gives mean +- std")
    ap.add_argument("--teacher", choices=["small", "base", "large"], default="small")
    ap.add_argument("--teacher-batch", type=int, default=8)
    ap.add_argument("--student", choices=list(fz.Student.SIZES), default="small")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lambda-t", type=float, default=1.0)
    ap.add_argument("--lambda-e", type=float, default=0.5)
    ap.add_argument("--ramp", type=float, default=0.3,
                    help="fraction of steps over which the teacher weight grows from 0 (0 = off)")
    ap.add_argument("--clip", type=float, default=1.0, help="gradient-norm clipping (0 = off)")
    ap.add_argument("--teacher-decay", type=float, default=0.0,
                    help="fraction of steps at the end over which the teacher weight falls to 0 (0 = off)")
    ap.add_argument("--label-subset", type=int, default=0,
                    help="which images are labeled: 0 = first N of the pool (default), "
                         "k > 0 = first N of a random order seeded by k")
    ap.add_argument("--aug", action="store_true",
                    help="random zoom-in crop (real depth divided by the zoom) and color jitter")
    ap.add_argument("--aug-scale", type=float, default=1.25, help="largest zoom of --aug")
    ap.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="channel dropout before the decoder head (0 = off)")
    ap.add_argument("--pool-gt", action="store_true",
                    help="also use the real depth of labeled images drawn in the pool half (E3/E4)")
    ap.add_argument("--lambda-t-lab", type=float, default=1.0,
                    help="teacher weight on images with real depth, relative to unlabeled ones")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/fewlabel_v2")
    ap.add_argument("--cache-dir", default="cache",
                    help="data + teacher output cache, shared by all scripts")
    ap.add_argument("--ckpt-dir", default=None,
                    help="save checkpoints here and resume from them (e.g. on Google Drive)")
    ap.add_argument("--ckpt-every", type=int, default=1000, help="steps between checkpoints")
    ap.add_argument("--stop-after", type=int, default=0, help=argparse.SUPPRESS)  # for testing resume
    ap.add_argument("--prepare-only", action="store_true",
                    help="only build the data + teacher cache, then exit")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="seed for the data split")
    args = ap.parse_args()

    if args.quick:
        args.pool, args.n_val, args.steps, args.eval_every = 300, 60, 60, 30
        args.labeled = ",".join(n if n.strip().endswith("%") else str(max(4, int(int(n) * 300 / 5000)))
                                for n in args.labeled.split(","))
    if args.data == "dummy":
        args.no_pretrained = True
    recipes = [r.strip() for r in args.recipes.split(",") if r.strip()]
    for r in recipes:
        if r not in RECIPES:
            raise SystemExit(f"unknown recipe {r!r}; choose from {list(RECIPES)}")
    seeds = [int(s) for s in args.seeds.split(",")]

    device = fz.pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    print(f"Device: {device} | {len(recipes)} recipes x {len(seeds)} seeds x {args.steps} steps")

    print("\n[1/3] Data (validation = held-out rooms) and teacher outputs")
    pool, val, meta = build_data(args, device)
    if args.prepare_only:
        print(f"Cache ready in {args.cache_dir} (--prepare-only): pool {len(pool['rgb'])}, "
              f"val {len(val['rgb'])}")
        return
    budgets = parse_budgets(args.labeled, len(pool["rgb"]))
    if device.type == "mps":
        torch.mps.empty_cache()

    # references on the validation rooms
    val_teacher = np.stack([fz.align_teacher(d.astype(np.float32), g)[0]
                            for d, g in zip(val["disp"], val["depth"])])
    ref = {"teacher_aligned_optimistic": fz.depth_metrics(val_teacher, val["depth"])}
    lab0 = pool["depth"][:budgets[0]]
    const = np.full_like(val["depth"], float(np.median(lab0[(lab0 > MIN_D) & (lab0 < MAX_D)])))
    ref["constant_depth"] = fz.depth_metrics(const, val["depth"])

    print(f"\n[2/3] Training: labeled {budgets} of {len(pool['rgb'])} images, recipes {recipes}")
    results, fig_preds = {}, {}
    for n_lab in budgets:
        for recipe in recipes:
            runs = []
            for seed in seeds:
                print(f"  -- {RECIPES[recipe]['exp']} {recipe}, {n_lab} labeled, seed {seed}")
                best, hist, preds, secs = train_run(recipe, n_lab, seed, pool, val, args, device)
                runs.append({"best": best, "history": hist, "minutes": secs / 60})
                if n_lab == budgets[0] and seed == seeds[0]:
                    fig_preds[recipe] = preds
            best_m = [r["best"] for r in runs]
            last_m = [r["history"][-1] for r in runs]      # final step, no checkpoint selection
            results[f"{n_lab}/{recipe}"] = {
                "labeled": n_lab, "recipe": recipe,
                "mean": aggregate(np.mean, best_m), "std": aggregate(np.std, best_m),
                "mean_last": aggregate(np.mean, last_m), "std_last": aggregate(np.std, last_m),
                "runs": runs}

    # ---- summary ------------------------------------------------------------
    print("\n[3/3] Summary")
    line = "-" * 93
    n_pool = len(pool["rgb"])
    print(f"{line}\nFEW-LABEL TEST  (validation: {len(val['rgb'])} images from held-out rooms, "
          f"best checkpoint; last = AbsRel at the final step)\n{line}")
    print(f"{'Labeled':<14}{'Recipe':<16}{'AbsRel':>8}{'RMSE':>8}{'d<1.25':>8}"
          f"{'vs gt':>10}{'scale':>7}{'best@':>7}{'last':>15}")
    verdicts = {}
    for n_lab in budgets:
        g = results.get(f"{n_lab}/gt")
        for recipe in recipes:
            r = results[f"{n_lab}/{recipe}"]
            m = r["mean"]
            rel = ""
            if g and recipe != "gt":
                rel = f"{(m['AbsRel'] - g['mean']['AbsRel']) / g['mean']['AbsRel']:+.1%}"
            step = r["runs"][0]["best"]["step"]
            scale = f"x{r['runs'][0]['best']['scale']:.2f}"
            sd = f"  (best ±{r['std']['AbsRel']:.3f})" if len(seeds) > 1 else ""
            label = f"{n_lab} ({100 * n_lab / n_pool:.3g}%)"
            last = f"{r['mean_last']['AbsRel']:.3f}"
            if len(seeds) > 1:
                last += f" ±{r['std_last']['AbsRel']:.3f}"
            print(f"{label:<14}{RECIPES[recipe]['exp'] + ' ' + recipe:<16}{m['AbsRel']:>8.3f}"
                  f"{m['RMSE']:>8.3f}{m['delta1']:>8.3f}{rel:>10}{scale:>7}{step:>7}{last:>15}{sd}")
        if g:
            others = [results[f"{n_lab}/{r}"]["mean"] for r in recipes if r != "gt"]
            if others:
                bestm = min(others, key=lambda m: m["AbsRel"])
                rel = (g["mean"]["AbsRel"] - bestm["AbsRel"]) / g["mean"]["AbsRel"]
                dd1 = bestm["delta1"] - g["mean"]["delta1"]
                if rel > 0.03 and dd1 > 0:
                    verdicts[n_lab] = f"teacher helps ({rel:.0%} lower AbsRel, d1 {dd1:+.3f})"
                elif rel < -0.03 and dd1 < 0:
                    verdicts[n_lab] = f"teacher hurts ({-rel:.0%} higher AbsRel, d1 {dd1:+.3f})"
                elif abs(rel) > 0.03 or abs(dd1) > 0.02:
                    verdicts[n_lab] = (f"mixed: AbsRel {-rel:+.0%}, d1 {dd1:+.3f} "
                                       "(metrics disagree, check the scale column)")
                else:
                    verdicts[n_lab] = "no clear difference (< 3%)"
        print(line)
    c, t = ref["constant_depth"], ref["teacher_aligned_optimistic"]
    print(f"{'Reference':<14}{'constant depth':<16}{c['AbsRel']:>8.3f}{c['RMSE']:>8.3f}{c['delta1']:>8.3f}")
    print(f"{'Reference':<14}{'teacher (opt.)':<16}{t['AbsRel']:>8.3f}{t['RMSE']:>8.3f}{t['delta1']:>8.3f}")
    print(line)
    for n_lab, v in verdicts.items():
        print(f"{n_lab} labeled: {v}")
    if len(seeds) == 1:
        print("Note: one seed only. Differences below ~3% may be noise; use --seeds 0,1,2 to check.")

    report = {"run": run_info(), "config": vars(args) | {"device": str(device)},
              "data": meta, "reference": ref,
              "results": results, "verdicts": {str(k): v for k, v in verdicts.items()}}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    fz.save_figure(out / "examples.png", val, val_teacher, fig_preds)
    print(f"Saved: {out / 'report.json'} and {out / 'examples.png'}")


if __name__ == "__main__":
    main()
