# Small and Fast Depth Estimation with Few Labels

CS466 Deep Learning (Fall 2026) term project, Özyeğin University.

A small student network (1–4M parameters) learns monocular depth on NYU Depth V2
from a few labeled images plus a big teacher (Depth Anything V2) on the unlabeled ones.

## Files

| File | What it does |
|---|---|
| `feasibility.py` | Shared code: data loading, teacher, `Student` model, losses, metrics, timing. Also the first full-label feasibility test. |
| `fewlabel.py` | Few-label experiment E1–E4 (v1). |
| `fewlabel_v2.py` | v2: teacher-weight ramp-up, gradient clipping, scale column. |
| `proposal.tex` | Project proposal. |
| `*_results*/` | `report.json` and example figures from each run. |
| `*.log` | Console output of each run. |

## Setup

```bash
pip install -r requirements.txt
```

Put the DenseDepth copy of NYU Depth V2 (50,688 train / 654 test images) in
`nyu_data/` (folder or zip). It is not in the repo. Teacher outputs are cached under
`*/cache/` on the first run; these are not in the repo either and are rebuilt automatically.

## Run

```bash
python3 fewlabel_v2.py --quick                       # ~3 min smoke test
python3 fewlabel_v2.py                               # E1–E4 at 5% labels
python3 fewlabel_v2.py --labeled 50,250,1250,5000 --recipes gt,semi_edge
```

Validation uses held-out training rooms. The 654 test images are kept for the final results.
