"""Pick the indoor images of Places365 and pack them for scripts/fewlabel_v2.py --extra-pool.

Places365 is not part of this repository: download it yourself from the official page
(http://places2.csail.mit.edu/download.html). By downloading it you accept its terms: non-commercial
research and education only, and the images may not be redistributed. Keep the zip this script
writes on your own disk or Drive, never in the repository.

What you need (the validation split is enough for a first test, ~16K indoor images):
    val_large.tar                      images, shorter side 512 px; unpacks to val_large/
    filelist_places365-standard.tar    places365_val.txt: image name and class index
    categories_places365.txt, IO_places365.txt   from github.com/CSAILVision/places365
                                       (class names; 1 = indoor, 2 = outdoor)

    python scripts/prepare_places.py --images val_large --list places365_val.txt \\
        --categories categories_places365.txt --io IO_places365.txt --out places_indoor.zip

Every indoor image is center-cropped to 4:3 (as NYU) and resized to 512x384, so the teacher sees
it at about NYU's resolution; training resizes it to 256x192. The zip holds images/<name>.jpg and
list.txt (image, class). It works for the training split too (--list places365_train_standard.txt).
"""
import argparse
import random
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image

OUT_W, OUT_H = 512, 384


def read_classes(categories, io_file):
    """Class index -> name, and the set of indoor class names (IO file: 1 = indoor)."""
    names = {}
    for line in Path(categories).read_text().split("\n"):
        if line.strip():
            name, idx = line.rsplit(maxsplit=1)
            names[int(idx)] = name
    indoor = set()
    for line in Path(io_file).read_text().split("\n"):
        if line.strip():
            name, flag = line.rsplit(maxsplit=1)
            if flag == "1":
                indoor.add(name)
    unknown = indoor - set(names.values())
    if unknown:
        raise SystemExit(f"{len(unknown)} indoor classes are not in {categories}, e.g. {sorted(unknown)[:3]}")
    return names, indoor


def select(list_file, names, indoor, per_class=0, seed=0):
    """(relative image path, class name) of the indoor images; at most per_class of each class."""
    by_class = {}
    for line in Path(list_file).read_text().split("\n"):
        if line.strip():
            path, idx = line.rsplit(maxsplit=1)
            name = names[int(idx)]
            if name in indoor:
                by_class.setdefault(name, []).append(path.lstrip("/"))
    rng = random.Random(seed)
    out = []
    for name in sorted(by_class):
        paths = sorted(by_class[name])
        if per_class and len(paths) > per_class:
            paths = sorted(rng.sample(paths, per_class))
        out += [(p, name) for p in paths]
    return out


def crop_4_3(img):
    """Largest centered 4:3 landscape crop, resized to OUT_W x OUT_H."""
    w, h = img.size
    if w * 3 > h * 4:                       # too wide: cut the sides
        cw, ch = round(h * 4 / 3), h
    else:                                   # too tall: cut top and bottom
        cw, ch = w, round(w * 3 / 4)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize((OUT_W, OUT_H), Image.BICUBIC)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="folder the paths in --list are relative to")
    ap.add_argument("--list", required=True, help="places365_val.txt or places365_train_standard.txt")
    ap.add_argument("--categories", required=True, help="categories_places365.txt")
    ap.add_argument("--io", required=True, help="IO_places365.txt (1 = indoor)")
    ap.add_argument("--out", required=True, help="output zip, e.g. places_indoor.zip")
    ap.add_argument("--per-class", type=int, default=0, help="at most this many images per class (0 = all)")
    ap.add_argument("--seed", type=int, default=0, help="seed for --per-class sampling")
    args = ap.parse_args(argv)

    names, indoor = read_classes(args.categories, args.io)
    items = select(args.list, names, indoor, args.per_class, args.seed)
    if not items:
        raise SystemExit("no indoor images found: check --list, --categories and --io")
    print(f"{len(indoor)} indoor classes of {len(names)}; {len(items)} images")

    root, out = Path(args.images), Path(args.out)
    tmp = out.with_suffix(out.suffix + ".tmp")
    lines, skipped = [], 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:     # JPEGs do not compress further
        for k, (rel, name) in enumerate(items, 1):
            try:
                img = Image.open(root / rel).convert("RGB")
            except (OSError, ValueError) as e:
                print(f"  skipped {rel}: {e}")
                skipped += 1
                continue
            buf = BytesIO()
            crop_4_3(img).save(buf, "JPEG", quality=95)
            arc = "images/" + rel.replace("/", "_")
            z.writestr(arc, buf.getvalue())
            lines.append(f"{arc}\t{name}")
            if k % 2000 == 0:
                print(f"  {k}/{len(items)}")
        z.writestr("list.txt", "\n".join(lines) + "\n")
    tmp.replace(out)                                               # only a complete zip gets the name
    mb = out.stat().st_size / 1e6
    print(f"done: {len(lines)} images ({skipped} skipped) -> {out} ({mb:.0f} MB)")


if __name__ == "__main__":
    main()
