"""Disk-backed cache of resized images, real depth and raw teacher output.

Each array is its own .npy file, opened with mmap_mode="r", so a pool of ~45K images
(~20 GB) never has to fit in RAM. Files are built only when they are missing. This way
the expensive teacher output (disp_<teacher>.npy + meta.json) can be kept elsewhere,
e.g. on Google Drive, and copied back, while rgb/depth are rebuilt from the dataset.

meta.json stores a hash of the image list, so a reused teacher file always matches
the images it was computed on.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from kd_depth.core import H, W


def list_hash(items):
    """Short, stable fingerprint of an ordered list (e.g. the image paths of a pool)."""
    h = hashlib.sha256()
    for it in items:
        h.update(repr(it).encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


def _specs(n, teacher_tag):
    return {
        "rgb": ((n, H, W, 3), np.uint8),
        "depth": ((n, H, W), np.float32),
        f"disp_{teacher_tag}": ((n, H, W), np.float16),   # raw relative inverse depth
    }


def build(cdir, n, samples, key, teacher_factory, teacher_tag, bs=8, tag="pool"):
    """Return {"rgb", "depth", "disp"} as read-only memory-mapped arrays.

    samples:          callable returning an iterator of (PIL RGB image, depth in meters)
    key:              fingerprint of the image list (see list_hash)
    teacher_factory:  callable returning the teacher; only called if the teacher file is missing
    """
    cdir = Path(cdir)
    cdir.mkdir(parents=True, exist_ok=True)
    meta_path = cdir / "meta.json"
    meta = {"key": key, "n": n, "size": [H, W]}
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old != meta:
            raise SystemExit(f"{cdir} was built for different images ({old}, now {meta}). "
                             "Delete that folder or use another --cache-dir.")
    specs = _specs(n, teacher_tag)
    missing = [k for k in specs if not (cdir / f"{k}.npy").exists()]
    if missing:
        need_teacher = any(k.startswith("disp") for k in missing)
        print(f"  [{tag}] building {', '.join(missing)} for {n} images in {cdir}")
        _fill(cdir, n, samples(), specs, missing, teacher_factory if need_teacher else None, bs, tag)
        meta_path.write_text(json.dumps(meta))
    else:
        print(f"  [{tag}] {n} images from cache {cdir}")
    return {("disp" if k.startswith("disp") else k): np.load(cdir / f"{k}.npy", mmap_mode="r")
            for k in specs}


def _fill(cdir, n, samples, specs, missing, teacher_factory, bs, tag):
    tmp = {k: cdir / f"{k}.npy.tmp" for k in missing}
    out = {k: np.lib.format.open_memmap(tmp[k], mode="w+", dtype=specs[k][1], shape=specs[k][0])
           for k in missing}
    disp_key = next((k for k in missing if k.startswith("disp")), None)
    teacher = teacher_factory() if teacher_factory else None
    batch, k, t0 = [], 0, time.perf_counter()

    def flush():
        nonlocal k
        disp = teacher([b[0] for b in batch]) if teacher is not None else [None] * len(batch)
        for (im, dep), d in zip(batch, disp):
            if "rgb" in out:
                out["rgb"][k] = np.asarray(im.resize((W, H), Image.BILINEAR))
            if "depth" in out:
                out["depth"][k] = np.asarray(Image.fromarray(dep).resize((W, H), Image.NEAREST))
            if disp_key:
                out[disp_key][k] = d
            k += 1
        batch.clear()

    for im, dep in samples:
        batch.append((im, dep))
        if len(batch) == bs:
            flush()
            if k % 2000 < bs:
                el = time.perf_counter() - t0
                print(f"  [{tag}] {k}/{n} images ({el:.0f}s, ~{el / k * (n - k) / 60:.0f} min left)")
    if batch:
        flush()
    if k != n:
        raise RuntimeError(f"expected {n} images, got {k}")
    print(f"  [{tag}] {k}/{n} images ({time.perf_counter() - t0:.0f}s)")
    for name in list(out):
        out.pop(name).flush()
        os.replace(tmp[name], cdir / f"{name}.npy")   # only complete files get the final name
