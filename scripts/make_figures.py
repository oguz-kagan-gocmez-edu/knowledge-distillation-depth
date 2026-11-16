#!/usr/bin/env python3
"""
Figures for the report, built only from the report.json files in results/ (no training, no data).

  val_curves      validation AbsRel during training, one panel per label budget
  epochs          validation AbsRel against passes over the labeled images (63k-step runs)
  test_budget     test AbsRel against the label budget
  test_checks_1   test AbsRel at 1% labels, E1 and E4 under each check

Run from the repository root:
  python scripts/make_figures.py --out figures
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
BUDGETS = {462: "1%", 2312: "5%", 11562: "25%", 46248: "100%"}

# fixed series order and colors (categorical slots 1-4, validated for color-vision deficiency)
SERIES = {
    "E1": {"label": "E1 labels only", "color": "#2a78d6", "marker": "o"},
    "E4": {"label": "E4 + teacher", "color": "#eb6834", "marker": "s"},
    "decay": {"label": "E4 + teacher decay", "color": "#1baf7a", "marker": "^"},
    "gtmask": {"label": "E4 + decay + GT mask", "color": "#eda100", "marker": "D"},
}
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e6e5e0"

# which run folders hold each (series, budget); several folders = seeds of the same setting
VAL_RUNS = {
    ("E1", 462): [("fix_1_short", "462/gt")],
    ("E4", 462): [("fix_1_short", "462/semi_edge")],
    ("decay", 462): [("decay_1_5_25", "462/semi_edge")],
    ("gtmask", 462): [("gtmask_decay", "462/semi_edge")],
    ("E1", 2312): [("main_5_short", "2312/gt")],
    ("E4", 2312): [("main_5_short", "2312/semi_edge")],
    ("decay", 2312): [("decay_1_5_25", "2312/semi_edge")],
    ("gtmask", 2312): [("gtmask_decay", "2312/semi_edge"), ("gtmask_seeds", "2312/semi_edge")],
    ("E1", 11562): [("budgets_25_100", "11562/gt"), ("e1_25_seeds", "11562/gt")],
    ("E4", 11562): [("budgets_25_100", "11562/semi_edge")],
    ("decay", 11562): [("decay_1_5_25", "11562/semi_edge")],
    ("gtmask", 11562): [("gtmask_decay", "11562/semi_edge"), ("gtmask_seeds", "11562/semi_edge")],
    ("E1", 46248): [("budgets_25_100", "46248/gt")],
    ("E4", 46248): [("budgets_25_100", "46248/semi_edge")],
    ("decay", 46248): [("decay_100", "46248/semi_edge")],
    ("gtmask", 46248): [("gtmask_decay", "46248/semi_edge")],
}
# test groups: (series, budget) -> run folders, matched with recipe and label count
TEST_RUNS = {
    "E1": {462: ["fix_1_short"], 2312: ["main_5_short"], 11562: ["budgets_25_100", "e1_25_seeds"],
           46248: ["budgets_25_100"]},
    "E4": {462: ["fix_1_short"], 2312: ["main_5_short"], 11562: ["budgets_25_100"],
           46248: ["budgets_25_100"]},
    "decay": {462: ["decay_1_5_25"], 2312: ["decay_1_5_25"], 11562: ["decay_1_5_25"], 46248: ["decay_100"]},
    "gtmask": {462: ["gtmask_decay"], 2312: ["gtmask_decay", "gtmask_seeds"],
               11562: ["gtmask_decay", "gtmask_seeds"], 46248: ["gtmask_decay"]},
}
CHECKS_1 = [("Default (3 seeds)", "fix_1_short"), ("Other subset k=1", "subsets_1_k1"),
            ("Other subset k=2", "subsets_1_k2"), ("Augmentation", "reg_1_aug"),
            ("Weight decay 0.01", "reg_1_wd"), ("Dropout 0.2", "reg_1_drop"),
            ("63k steps (3 seeds)", "main_1_5"), ("Student tiny (~1M)", "students_1_tiny"),
            ("Student base (~4M)", "students_1_base")]


def style():
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 1.5, "lines.markersize": 4.5, "savefig.bbox": "tight", "figure.dpi": 100,
    })


def histories(folder, key):
    """Validation history of every seed of one setting: list of (steps, AbsRel) arrays."""
    res = json.loads((RESULTS / folder / "report.json").read_text())["results"][key]
    return [(np.array([h["step"] for h in r["history"]]), np.array([h["AbsRel"] for h in r["history"]]))
            for r in res["runs"]]


def seed_curves(parts):
    hs = [h for folder, key in parts for h in histories(folder, key)]
    steps = hs[0][0]
    if any(len(s) != len(steps) or (s != steps).any() for s, _ in hs):
        raise SystemExit(f"seeds of {parts} were evaluated at different steps")
    return steps, np.stack([a for _, a in hs])


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print(f"  {out / name}.png/.pdf")


def val_curves(out):
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9))
    for ax, (n, name) in zip(axes, BUDGETS.items()):
        for s, spec in SERIES.items():
            steps, a = seed_curves(VAL_RUNS[(s, n)])
            m = a.mean(0)
            ax.plot(steps / 1000, m, color=spec["color"], marker=spec["marker"], label=spec["label"])
            if len(a) > 1:
                ax.fill_between(steps / 1000, a.min(0), a.max(0), color=spec["color"], alpha=0.15, lw=0)
        ax.set_title(f"{name} labeled ({n:,} images)")
        ax.set_xlabel("training step (k)")
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.axvline(12, color=INK2, lw=0.8, ls=":")
    axes[0].set_ylabel("validation AbsRel ↓")
    axes[0].annotate("decay starts", xy=(12, axes[0].get_ylim()[1]), xytext=(13, axes[0].get_ylim()[1]),
                     va="top", fontsize=7, color=INK2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, out, "val_curves")


def epochs(out):
    """main_1_5: 63k steps, 3 seeds. E1 sees 8 labeled images per step, E4 4 (the rest unlabeled)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.9))
    for ax, n in zip(axes, (462, 2312)):
        for s, per_step in (("E1", 8), ("E4", 4)):
            spec = SERIES[s]
            steps, a = seed_curves([("main_1_5", f"{n}/{'gt' if s == 'E1' else 'semi_edge'}")])
            ep = steps * per_step / n
            ax.plot(ep, a.mean(0), color=spec["color"], marker=spec["marker"], label=spec["label"])
            ax.fill_between(ep, a.min(0), a.max(0), color=spec["color"], alpha=0.15, lw=0)
        ax.set_xscale("log")
        ax.set_title(f"{BUDGETS[n]} labeled ({n:,} images), 63k steps")
        ax.set_xlabel("passes over the labeled images (epochs, log scale)")
    axes[0].set_ylabel("validation AbsRel ↓")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save(fig, out, "epochs")


def test_runs():
    runs = []
    for name in ("test_final", "test_seeds"):
        runs += json.loads((RESULTS / name / "report.json").read_text())["runs"]
    for r in runs:
        r.setdefault("group", Path(r["file"]).parent.name)
    return runs


def test_values(runs, groups, n, recipe, student="small"):
    v = [r["metrics"]["raw"]["AbsRel"] for r in runs if r["group"] in groups and r["n_lab"] == n
         and r["recipe"] == recipe and r["model"] == "best" and r["student"] == student]
    if not v:
        raise SystemExit(f"no test result for {groups} {n} {recipe}")
    return np.array(v)


def test_budget(out, runs):
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    xs = np.array(list(BUDGETS)) / 46248 * 100
    offsets = {"E1": -0.045, "E4": -0.015, "decay": 0.015, "gtmask": 0.045}   # in log10 units
    for s, spec in SERIES.items():
        recipe = "gt" if s == "E1" else "semi_edge"
        vals = [test_values(runs, TEST_RUNS[s][n], n, recipe) for n in BUDGETS]
        m = np.array([v.mean() for v in vals])
        err = np.array([v.std() for v in vals])
        x = xs * 10 ** offsets[s]
        ax.errorbar(x, m, yerr=err, color=spec["color"], marker=spec["marker"], label=spec["label"],
                    capsize=2, elinewidth=1)
    ax.set_xscale("log")
    ax.set_xticks(xs, list(BUDGETS.values()))
    ax.minorticks_off()
    ax.set_xlabel("labeled images (share of the 46,248-image pool)")
    ax.set_ylabel("test AbsRel ↓")
    ax.legend(frameon=False)
    ax.set_title("Test set (654 images): error bars = std over 3 seeds where run", fontsize=9)
    fig.tight_layout()
    save(fig, out, "test_budget")


def test_checks_1(out, runs):
    rows = []
    for label, group in CHECKS_1:
        student = {"students_1_tiny": "tiny", "students_1_base": "base"}.get(group, "small")
        e1 = test_values(runs, [group], 462, "gt", student).mean()
        e4 = test_values(runs, [group], 462, "semi_edge", student).mean()
        rows.append((label, e1, e4))
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    y = np.arange(len(rows))[::-1]
    for yi, (_, e1, e4) in zip(y, rows):
        ax.plot([e4, e1], [yi, yi], color=GRID, lw=2.5, zorder=1)
        ax.text(max(e1, e4) + 0.002, yi, f"{(e4 - e1) / e1:+.0%}", va="center", fontsize=7.5, color=INK2)
    for s, col in (("E1", 1), ("E4", 2)):
        spec = SERIES[s]
        ax.scatter([r[col] for r in rows], y, color=spec["color"], marker=spec["marker"], s=30,
                   label=spec["label"], zorder=2, edgecolors="white", linewidths=0.8)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("test AbsRel ↓")
    ax.set_title("1% labels (462 images): the teacher's gain under each check", fontsize=9)
    ax.set_xlim(right=ax.get_xlim()[1] + 0.006)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.6, -0.05))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, out, "test_checks_1")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    runs = test_runs()
    print("Figures:")
    val_curves(out)
    epochs(out)
    test_budget(out, runs)
    test_checks_1(out, runs)


if __name__ == "__main__":
    main()
