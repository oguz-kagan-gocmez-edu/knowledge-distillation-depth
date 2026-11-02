# Small and Fast Depth Estimation with Few Labels

[![CI](https://github.com/oguz-kagan-gocmez-edu/knowledge-distillation-depth/actions/workflows/ci.yml/badge.svg)](https://github.com/oguz-kagan-gocmez-edu/knowledge-distillation-depth/actions/workflows/ci.yml)

CS466 Deep Learning (Fall 2026) term project, Özyeğin University. Author: Oğuz Kağan Göçmez.

A small student network (1–4M parameters) learns monocular depth on NYU Depth V2 from a
few labeled images plus a large teacher (Depth Anything V2, 24.8M) on the unlabeled images.
The goal is a model about 8× smaller than the teacher that runs in real time, and a study of
**when the teacher helps and when it does not**.

## Current status

With 5% labels (250 of 5,000 images), learning from the teacher on the unlabeled images
lowers AbsRel by about 5% (3 seeds), see [results/](results/README.md).
With full labels, the teacher gives no gain.

## Repository layout

```
src/kd_depth/        shared code: data, teacher, Student model, losses, metrics, run info
scripts/             experiment entry points
  feasibility.py       full-label feasibility test (+ speed / FLOPs / compute estimates)
  fewlabel_v2.py       few-label experiments E1–E4 (current)
  fewlabel_v1.py       first version, kept for reproducibility
notebooks/           colab_run.ipynb: long runs on a Colab GPU, with checkpoints on Drive
tests/               unit tests (resume, cache reuse, label budgets)
results/             one folder per run: report.json, examples.png, run.log
docs/proposal/       proposal (LaTeX + PDF)
docs/course/         course project brief
data/                NYU data (not in git)
cache/               cached data + teacher outputs (not in git)
```

## Setup

```bash
git clone https://github.com/oguz-kagan-gocmez-edu/knowledge-distillation-depth.git
cd knowledge-distillation-depth
pip install -e ".[dev]"          # or: pip install -r requirements.txt
```

Data: put the DenseDepth copy of NYU Depth V2 (50,688 train / 654 test images,
`nyu_data.zip` or its extracted folder) at `data/nyu_data`. It is not in the repository.

## Running

All commands run from the repository root.

```bash
make test                                            # unit tests, ~2 min
make smoke                                           # offline check of every script, ~2 min
python scripts/fewlabel_v2.py --quick                # ~3 min on real data
python scripts/fewlabel_v2.py                        # E1–E4 at 5% labels, ~40 min (M4 Pro)
python scripts/fewlabel_v2.py --labeled 1%,5%,25%,100% --recipes gt,semi_edge --out results/sweep
python scripts/fewlabel_v2.py --pool -1 --labeled 5% --ckpt-dir checkpoints/full   # all ~45K images
python scripts/feasibility.py --source densedepth --data-zip data/nyu_data
```

The first run on real data computes the teacher outputs and caches them in `cache/`.
The 5,000-image runs use the original single-file cache; larger pools use a folder of `.npy`
files read from disk on demand, so the full pool (~20 GB) needs little RAM.
With `--ckpt-dir`, a run saves checkpoints every `--ckpt-every` steps; starting it again with the
same arguments continues where it stopped, and finished runs are skipped.

## Long runs on Colab

Open [`notebooks/colab_run.ipynb`](notebooks/colab_run.ipynb) in Colab (File → Open notebook → GitHub),
follow the one-time setup at the top (upload `nyu_data.zip` to `MyDrive/kd_depth/`, add a read-only
`GITHUB_TOKEN` secret, pick an L4 GPU), set `RUN_NAME` and `ARGS`, and Run all. Teacher outputs,
checkpoints and results are kept on Drive, so after a disconnect Run all simply continues.

| Recipe | ID | Training signal |
|---|---|---|
| `gt` | E1 | real depth on labeled images (baseline) |
| `gt_t` | E2 | real depth + teacher, labeled images only |
| `semi` | E3 | real depth on labeled images + teacher on all images |
| `semi_edge` | E4 | E3 + edge (gradient-matching) loss |

## Evaluation protocol

- Validation: 10% of the training **rooms** are held out, used for checkpoint selection and tuning.
- Test: the standard 654 NYU test images, used only for the final results.
- Metrics: AbsRel, RMSE, δ<1.25; efficiency: parameters, FLOPs, latency.

## Workflow and citations

See [CONTRIBUTING.md](CONTRIBUTING.md) for how experiments are tracked with issues, branches
and pull requests, and [CITATIONS.md](CITATIONS.md) for the models, data and methods used.
