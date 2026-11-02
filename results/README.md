# Results index

Each folder has `report.json` (full config + metrics), `examples.png`, and `run.log` when the
console output was saved. From now on `report.json` also records the git commit under `"run"`.

Validation = 300 images from 25 held-out rooms (feasibility: 200 random images, not split by room).
The 654 test images have not been used yet. All runs: Depth Anything V2 Small teacher, Apple M4 Pro (MPS).

| Folder | Script | Setting | Recipe | AbsRel ↓ | RMSE ↓ | δ<1.25 ↑ |
|---|---|---|---|---|---|---|
| `feasibility` | `feasibility.py` | 2,000 imgs, all labeled, 1.5k steps, small | gt | 0.202 | 0.675 | 0.675 |
| | | | full | 0.204 | 0.677 | 0.673 |
| `fewlabel_v1` | `fewlabel_v1.py` | 250 / 5,000 labeled, 2k steps, small | E1 gt | 0.212 | 0.770 | 0.622 |
| | | | E2 gt_t | 0.252 | 0.860 | 0.572 |
| | | | E3 semi | 0.222 | 0.787 | 0.624 |
| | | | E4 semi_edge | 0.241 | 0.842 | 0.592 |
| `fewlabel_v2` | `fewlabel_v2.py` | 50 / 5,000 labeled, small | E1 gt | 0.266 | 0.898 | 0.535 |
| | | | E3 semi | 0.252 | 0.855 | 0.580 |
| | | 250 / 5,000 labeled, small | E1 gt | 0.219 | 0.781 | 0.609 |
| | | | E3 semi | 0.204 | 0.739 | 0.666 |
| `fewlabel_v2_seeds` | `fewlabel_v2.py` | 250 / 5,000, small, seeds 0,1,2 (mean) | E1 gt | 0.221 | 0.781 | 0.608 |
| | | | E3 semi | 0.211 | 0.757 | 0.648 |
| | | | E4 semi_edge | 0.209 | 0.758 | 0.653 |
| `fewlabel_v2_tiny` | `fewlabel_v2.py` | 250 / 5,000, tiny (~1M) | E1 gt | 0.243 | 0.852 | 0.566 |
| | | | E3 semi | 0.241 | 0.839 | 0.586 |
| `fewlabel_v2_base` | `fewlabel_v2.py` | 250 / 5,000, base (~4M) | E1 gt | 0.208 | 0.764 | 0.620 |
| | | | E3 semi | 0.202 | 0.740 | 0.670 |

These runs were made before the project used git, so their reports have no commit hash.
