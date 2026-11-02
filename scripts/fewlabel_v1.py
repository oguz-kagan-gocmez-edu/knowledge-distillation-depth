#!/usr/bin/env python3
"""
CS466 term project - few-label experiment (E1-E4 at one label budget, optional E5 sweep)

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

Archived first version (no ramp-up, no clipping); use fewlabel_v2.py for new runs.
Run from the repository root:
  python scripts/fewlabel_v1.py --quick         # ~3 min smoke test
  python scripts/fewlabel_v1.py                 # E1-E4, 250 of 5000 images labeled (5%), ~40 min
  python scripts/fewlabel_v1.py --labeled 50,250,1250,5000 --recipes gt,semi_edge
                                      # mini E5 sweep (1/5/25/100%), ~75 min
Data defaults to data/nyu_data (zip or folder). Results go to results/fewlabel_v1/,
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
from kd_depth.core import MAX_D, MIN_D, H, W  # noqa: E402
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
def open_densedepth(path):
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
    csvs = sorted((nm for nm in names if nm.endswith("nyu2_train.csv") and "__MACOSX" not in nm),
                  key=len)
    if not csvs:
        raise SystemExit(f"nyu2_train.csv not found inside {path}")
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


def load_set(samples, n, teacher, bs, tag):
    """Resize to H x W and store the RAW teacher output (relative inverse depth)."""
    rgb = np.zeros((n, H, W, 3), np.uint8)
    depth = np.zeros((n, H, W), np.float32)
    disp = np.zeros((n, H, W), np.float16)
    batch, k, t0 = [], 0, time.perf_counter()

    def flush():
        nonlocal k
        out = teacher([b[0] for b in batch])
        for (im, dep), d in zip(batch, out):
            rgb[k] = np.asarray(im.resize((W, H), Image.BILINEAR))
            depth[k] = np.asarray(Image.fromarray(dep).resize((W, H), Image.NEAREST))
            disp[k] = d
            k += 1
        batch.clear()

    for im, dep in samples:
        batch.append((im, dep))
        if len(batch) == bs:
            flush()
            if k % 500 < bs:
                print(f"  [{tag}] {k}/{n} images ({time.perf_counter() - t0:.0f}s)")
    if batch:
        flush()
    print(f"  [{tag}] {k}/{n} images ({time.perf_counter() - t0:.0f}s)")
    return {"rgb": rgb[:k], "depth": depth[:k], "disp": disp[:k]}


def build_data(args, device, out):
    tag = "dummy" if args.data == "dummy" else "densedepth"
    cache = Path(args.cache_dir) / f"{tag}_pool{args.pool}_val{args.n_val}_seed{args.seed}_{args.teacher}.npz"
    if cache.exists() and not args.no_cache:
        d = np.load(cache)
        print(f"  loaded data + teacher outputs from cache {cache.name}")
        meta = json.loads(str(d["meta"]))
        pool = {k: d[f"pool_{k}"] for k in ("rgb", "depth", "disp")}
        val = {k: d[f"val_{k}"] for k in ("rgb", "depth", "disp")}
        return pool, val, meta

    teacher = (fz.DummyTeacher() if args.data == "dummy"
               else fz.DepthAnythingTeacher(args.teacher, device))
    print(f"  teacher: {teacher.name}")
    if args.data == "dummy":
        meta = {"scenes": None}
        pool = load_set(fz.iter_dummy("train", args.pool, args.seed), args.pool, teacher,
                        args.teacher_batch, "pool")
        val = load_set(fz.iter_dummy("val", args.n_val, args.seed), args.n_val, teacher,
                       args.teacher_batch, "val")
    else:
        rows, read, resolve = open_densedepth(args.data)
        train_rows, val_rows, n_sc, n_val_sc = scene_split(rows, args.val_scene_frac, args.seed)
        rng = random.Random(args.seed)
        rng.shuffle(train_rows)
        rng.shuffle(val_rows)
        print(f"  {len(rows)} images in {n_sc} rooms -> {n_val_sc} rooms held out for validation")
        meta = {"rooms": n_sc, "val_rooms": n_val_sc}
        pool = load_set(iter_rows(train_rows[:args.pool], read, resolve), min(args.pool, len(train_rows)),
                        teacher, args.teacher_batch, "pool")
        val = load_set(iter_rows(val_rows[:args.n_val], read, resolve), min(args.n_val, len(val_rows)),
                       teacher, args.teacher_batch, "val")
    del teacher
    np.savez(cache, meta=json.dumps(meta),
             **{f"pool_{k}": v for k, v in pool.items()}, **{f"val_{k}": v for k, v in val.items()})
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


def teacher_terms(pred, tdisp, with_edge, scales=4):
    pdisp = 1.0 / pred.clamp(min=0.1)                 # student depth -> inverse depth
    a, b = ssi_normalize(pdisp), ssi_normalize(tdisp)
    l_t = (a - b).abs().mean()
    if not with_edge:
        return l_t, None
    r = (a - b).unsqueeze(1)                          # gradient matching (MiDaS style)
    l_e = 0.0
    for s in range(scales):
        if s:
            r = F.avg_pool2d(r, 2)
        l_e = l_e + (r[..., :, 1:] - r[..., :, :-1]).abs().mean() \
                  + (r[..., 1:, :] - r[..., :-1, :]).abs().mean()
    return l_t, l_e / scales


# ----------------------------------------------------------------------------
# Training one (recipe, label budget, seed)
# ----------------------------------------------------------------------------
def train_run(recipe, n_lab, seed, pool, val, args, device):
    cfg = RECIPES[recipe]
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = fz.Student(args.student, pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=max(0.1, 3 / args.steps))
    rgb = torch.from_numpy(pool["rgb"])
    depth = torch.from_numpy(pool["depth"])
    disp = torch.from_numpy(pool["disp"])
    n_pool, bs = len(rgb), args.batch_size

    best, hist, logs = None, [], {"gt": [], "teacher": [], "edge": []}
    t_start = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        if cfg["pool"] == "all":                       # half labeled, half from all images
            n_l = bs // 2
            idx = torch.cat([torch.randint(0, n_lab, (n_l,), generator=gen),
                             torch.randint(0, n_pool, (bs - n_l,), generator=gen)])
            is_lab = torch.arange(bs) < n_l
        else:
            idx = torch.randint(0, n_lab, (bs,), generator=gen)
            is_lab = torch.ones(bs, dtype=torch.bool)
        x = rgb[idx].permute(0, 3, 1, 2).float().div(255)
        d, t = depth[idx].clone(), disp[idx].float()
        flip = torch.rand(bs, generator=gen) < 0.5
        x[flip], d[flip], t[flip] = x[flip].flip(-1), d[flip].flip(-1), t[flip].flip(-1)
        x, d, t, is_lab = x.to(device), d.to(device), t.to(device), is_lab.to(device)

        pred = model(x)
        mask = (d > MIN_D) & (d < MAX_D) & is_lab.view(-1, 1, 1)
        l_gt = fz.silog_loss(pred, d, mask)
        loss = l_gt
        if cfg["teacher"]:
            l_t, l_e = teacher_terms(pred, t, cfg["edge"])
            loss = loss + args.lambda_t * l_t
            logs["teacher"].append(l_t.item())
            if cfg["edge"]:
                loss = loss + args.lambda_e * l_e
                logs["edge"].append(l_e.item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        logs["gt"].append(l_gt.item())

        if step == 50:
            el = time.perf_counter() - t_start
            print(f"    ~{el / 50 * args.steps / 60:.0f} min for this run")
        if step % args.eval_every == 0 or step == args.steps:
            m = fz.depth_metrics(fz.predict(model, val["rgb"], device), val["depth"])
            m["step"] = step
            hist.append(m)
            if best is None or m["AbsRel"] < best["AbsRel"]:
                best = dict(m)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            parts = "  ".join(f"{k} {np.mean(v[-args.eval_every:]):.3f}" for k, v in logs.items() if v)
            print(f"    step {step:>5}  {parts}  | val AbsRel {m['AbsRel']:.3f}  "
                  f"d1 {m['delta1']:.3f}")

    model.load_state_dict(best_state)
    preds = fz.predict(model, val["rgb"], device)
    return best, hist, preds, time.perf_counter() - t_start


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/nyu_data", help="DenseDepth zip/folder, or 'dummy'")
    ap.add_argument("--pool", type=int, default=5000, help="training images (labeled + unlabeled)")
    ap.add_argument("--labeled", default="250", help="labeled images, comma list for a sweep")
    ap.add_argument("--n-val", type=int, default=300)
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
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/fewlabel_v1")
    ap.add_argument("--cache-dir", default="cache", help="data + teacher output cache")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="seed for the data split")
    args = ap.parse_args()

    if args.quick:
        args.pool, args.n_val, args.steps, args.eval_every = 300, 60, 60, 30
        args.labeled = ",".join(str(max(4, int(int(n) * 300 / 5000))) for n in args.labeled.split(","))
    if args.data == "dummy":
        args.no_pretrained = True
    recipes = [r.strip() for r in args.recipes.split(",") if r.strip()]
    for r in recipes:
        if r not in RECIPES:
            raise SystemExit(f"unknown recipe {r!r}; choose from {list(RECIPES)}")
    budgets = sorted({int(n) for n in args.labeled.split(",")})
    seeds = [int(s) for s in args.seeds.split(",")]

    device = fz.pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    n_runs = len(budgets) * len(recipes) * len(seeds)
    print(f"Device: {device} | {n_runs} training runs x {args.steps} steps")

    print("\n[1/3] Data (validation = held-out rooms) and teacher outputs")
    pool, val, meta = build_data(args, device, out)
    budgets = [min(b, len(pool["rgb"])) for b in budgets]
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
            keys = ("AbsRel", "RMSE", "delta1")
            mean = {k: float(np.mean([r["best"][k] for r in runs])) for k in keys}
            std = {k: float(np.std([r["best"][k] for r in runs])) for k in keys}
            results[f"{n_lab}/{recipe}"] = {"labeled": n_lab, "recipe": recipe, "mean": mean,
                                            "std": std, "runs": runs}

    # ---- summary ------------------------------------------------------------
    print("\n[3/3] Summary")
    line = "-" * 78
    n_pool = len(pool["rgb"])
    print(f"{line}\nFEW-LABEL TEST  (validation: {len(val['rgb'])} images from held-out rooms, "
          f"best checkpoint)\n{line}")
    print(f"{'Labeled':<14}{'Recipe':<16}{'AbsRel':>8}{'RMSE':>8}{'d<1.25':>8}"
          f"{'vs gt':>10}{'best@':>8}")
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
            sd = f" ±{r['std']['AbsRel']:.3f}" if len(seeds) > 1 else ""
            label = f"{n_lab} ({n_lab / n_pool:.0%})"
            print(f"{label:<14}{RECIPES[recipe]['exp'] + ' ' + recipe:<16}{m['AbsRel']:>8.3f}"
                  f"{m['RMSE']:>8.3f}{m['delta1']:>8.3f}{rel:>10}{step:>8}{sd}")
        if g:
            others = [results[f"{n_lab}/{r}"]["mean"] for r in recipes if r != "gt"]
            if others:
                bestm = min(others, key=lambda m: m["AbsRel"])
                rel = (g["mean"]["AbsRel"] - bestm["AbsRel"]) / g["mean"]["AbsRel"]
                if rel > 0.03 and bestm["delta1"] > g["mean"]["delta1"]:
                    verdicts[n_lab] = f"teacher helps ({rel:.0%} lower AbsRel)"
                elif rel < -0.03:
                    verdicts[n_lab] = f"teacher hurts ({-rel:.0%} higher AbsRel)"
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
