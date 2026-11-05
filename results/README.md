# Results

Each folder has `report.json` (full config and metrics), `examples.png` and, for most runs,
`run.log` (console output). Reports made after the project moved to git also record the commit.

- **Teacher:** Depth Anything V2 Small (24.8M parameters) in every run.
- **Students:** tiny (~1M), small (3.15M, the default) and base (~4M).
- **Recipes:** E1 `gt` (labels only), E2 `gt_t` (+ teacher on labeled images), E3 `semi`
  (+ teacher on all images), E4 `semi_edge` (E3 + edge loss).
- **Metrics:** validation only. The 654 test images have not been used yet.
  "Best" is the checkpoint with the lowest validation AbsRel; "final" is the last step.

Contents:

1. [Official runs](#official-runs): 1,000 validation images, Colab L4
2. [Pilot runs](#pilot-runs): 300 validation images (not comparable with the official runs)
3. [Run history notes](#run-history-notes)

---

## Official runs

Full pool of 46,248 images, validation on **1,000 images from 25 held-out rooms**, small student
and 24k steps unless noted. The plan is in [`docs/official_runs.md`](../docs/official_runs.md).

| Reference | AbsRel ↓ | RMSE ↓ | δ<1.25 ↑ |
|---|---|---|---|
| Constant depth (lower bound) | 0.402 | 1.345 | 0.343 |
| Teacher, aligned to real depth (optimistic upper bound) | 0.110 | 0.519 | 0.881 |

### Summary: effect of the teacher by label budget

| Labeled | Images | Seeds | E1 gt | E4 semi_edge | E4 vs E1 | Folder |
|---|---|---|---|---|---|---|
| 1% | 462 | 3 | 0.199 | **0.176** | **−11.2%** | `fix_1_short` |
| 5% | 2,312 | 3 | **0.166** | 0.170 | +2.5% | `main_5_short` |
| 25% | 11,562 | 1 | **0.156** | 0.168 | +7.7% | `budgets_25_100` |
| 100% | 46,248 | 1 | **0.153** | 0.164 | +7.2% | `budgets_25_100` |

Best-checkpoint AbsRel. The teacher helps with very few labels and hurts with many.

### `fix_1_short` and `main_5_short`: main comparison (3 seeds, mean)

| Labeled | Recipe | AbsRel ↓ | RMSE ↓ | δ<1.25 ↑ | Final AbsRel |
|---|---|---|---|---|---|
| 1% | E1 gt | 0.199 ± 0.001 | 0.712 | 0.676 | 0.200 ± 0.001 |
| 1% | E4 semi_edge | **0.176 ± 0.001** | **0.705** | **0.725** | **0.181 ± 0.003** |
| 5% | E1 gt | **0.166 ± 0.001** | **0.635** | **0.744** | **0.166 ± 0.001** |
| 5% | E4 semi_edge | 0.170 ± 0.001 | 0.675 | 0.736 | 0.172 ± 0.000 |

- At 1% the teacher lowers AbsRel by 11.2% (best) and 9.5% (final step).
- At 5% it does not help: +2.5% AbsRel and 6% worse RMSE. E4 fits the labels less
  (training SILog 0.165 vs 0.067 for E1).

### `subsets_1_k1`, `subsets_1_k2`: other labeled subsets (1%, seed 0)

| Labeled subset | E1 gt | E4 semi_edge | E4 vs E1 | E4 final |
|---|---|---|---|---|
| 0 (`fix_1_short`, 3 seeds) | 0.199 | 0.176 | −11.2% | 0.181 |
| 1 (`subsets_1_k1`) | 0.202 | 0.180 | −11.1% | 0.180 |
| 2 (`subsets_1_k2`) | 0.199 | 0.177 | −10.9% | 0.180 |

The gain does not depend on which 462 images are labeled.

### `ablation_1_5`: which part of the teacher helps (seed 0)

| Labeled | E1 gt | E2 gt_t | E3 semi | E4 semi_edge |
|---|---|---|---|---|
| 1% | 0.199 | 0.195 (final 0.199) | **0.177** (final 0.180) | **0.176** |
| 5% | **0.166** | 0.174 (final 0.176) | 0.172 (final 0.176) | 0.170 |

E1 and E4 are from `fix_1_short` and `main_5_short` (3 seeds).

- The gain comes from the teacher on the **unlabeled** images (E3). The teacher on the labeled
  images only (E2) does not help.
- The edge loss (E3 → E4) adds nothing measurable.
- At 5% every teacher variant is worse than E1.

### `students_1_tiny`, `students_1_base`: student size (1%, seed 0)

| Student | Parameters | E1 gt | E4 semi_edge | E4 vs E1 | Final E1 → E4 |
|---|---|---|---|---|---|
| tiny (MobileNetV3-Small) | ~1M | 0.224 | 0.203 | −9.3% | 0.225 → 0.207 |
| small (MobileNetV3-Large, 3 seeds) | 3.15M | 0.199 | 0.176 | −11.2% | 0.200 → 0.181 |
| base (EfficientNet-B0) | ~4M | 0.190 | 0.174 | −8.5% | 0.196 → 0.184 |

- The teacher helps by about the same amount at every size.
- The small student with the teacher (0.176) beats the base student without it (0.190).
- Base E1 overfits: validation rises after 12k steps (0.190 → 0.196) while its training SILog
  falls to 0.046.

### `budgets_25_100`: many labels (seed 0)

| Labeled | Recipe | AbsRel ↓ | RMSE ↓ | δ<1.25 ↑ | Final AbsRel |
|---|---|---|---|---|---|
| 25% | E1 gt | **0.156** | **0.622** | **0.762** | **0.156** |
| 25% | E4 semi_edge | 0.168 | 0.670 | 0.743 | 0.171 |
| 100% | E1 gt | **0.153** | **0.595** | **0.781** | **0.153** |
| 100% | E4 semi_edge | 0.164 | 0.661 | 0.748 | 0.166 |

- E1 keeps improving until the last step. Each labeled image is seen only ~17 (25%) and
  ~4 (100%) times, so there is no memorization.
- E4 stays at 0.164–0.170 from 5% to 100% labels, and its training SILog stays at ~0.18.

### `main_1_5`: long training (63k steps, 3 seeds, mean)

| Labeled | Recipe | AbsRel ↓ | RMSE ↓ | δ<1.25 ↑ | Final AbsRel |
|---|---|---|---|---|---|
| 1% | E1 gt | 0.199 ± 0.002 | 0.713 | 0.676 | 0.205 ± 0.003 |
| 1% | E4 semi_edge | **0.182 ± 0.002** | **0.712** | **0.710** | 0.196 ± 0.011 |
| 5% | E1 gt | **0.165 ± 0.001** | **0.634** | **0.748** | **0.167 ± 0.001** |
| 5% | E4 semi_edge | 0.166 ± 0.001 | 0.672 | 0.744 | 0.171 ± 0.002 |

- At 1% E4 reaches 0.180–0.184 on every seed, but degrades late on two seeds
  (final 0.191 / 0.212 / 0.185) while its predicted scale drifts.
- With 24k steps (`fix_1_short`) E4 no longer degrades, so all later groups use 24k steps.

---

## Pilot runs

Validation on **300 images** from held-out rooms (`feasibility`: 200 random images, not split
by room). These runs were used to choose the setup.

### 5,000-image pool (Apple M4 Pro)

| Folder | Setting | Recipe | AbsRel ↓ | RMSE ↓ | δ<1.25 ↑ |
|---|---|---|---|---|---|
| `feasibility` | 2,000 images, all labeled, 1.5k steps | gt | 0.202 | 0.675 | 0.675 |
| | | full | 0.204 | 0.677 | 0.673 |
| `fewlabel_v1` | 250 labeled, 2k steps | E1 gt | 0.212 | 0.770 | 0.622 |
| | | E2 gt_t | 0.252 | 0.860 | 0.572 |
| | | E3 semi | 0.222 | 0.787 | 0.624 |
| | | E4 semi_edge | 0.241 | 0.842 | 0.592 |
| `fewlabel_v2` | 50 labeled | E1 gt | 0.266 | 0.898 | 0.535 |
| | | E3 semi | 0.252 | 0.855 | 0.580 |
| | 250 labeled | E1 gt | 0.219 | 0.781 | 0.609 |
| | | E3 semi | 0.204 | 0.739 | 0.666 |
| `fewlabel_v2_seeds` | 250 labeled, 3 seeds (mean) | E1 gt | 0.221 | 0.781 | 0.608 |
| | | E3 semi | 0.211 | 0.757 | 0.648 |
| | | E4 semi_edge | 0.209 | 0.758 | 0.653 |
| `fewlabel_v2_tiny` | 250 labeled, tiny student | E1 gt | 0.243 | 0.852 | 0.566 |
| | | E3 semi | 0.241 | 0.839 | 0.586 |
| `fewlabel_v2_base` | 250 labeled, base student | E1 gt | 0.208 | 0.764 | 0.620 |
| | | E3 semi | 0.202 | 0.740 | 0.670 |

### Full pool of 46,248 images (Colab L4)

| Folder | Setting | Recipe | AbsRel ↓ | RMSE ↓ | δ<1.25 ↑ |
|---|---|---|---|---|---|
| `timing_test` | 5%, 500 steps (speed check only) | E1 gt | 0.212 | 0.759 | 0.630 |
| `long_e1_e3` | 0.5% (231), 30k steps | E1 gt | 0.213 | 0.781 | 0.630 |
| | | E3 semi | 0.188 | 0.751 | 0.680 |
| | 1% (462), 30k steps | E1 gt | 0.191 | 0.690 | 0.692 |
| | | E3 semi | 0.170 | 0.698 | 0.726 |
| | 5% (2,312), 30k steps | E1 gt | 0.164 | 0.631 | 0.751 |
| | | E3 semi | 0.163 | 0.644 | 0.760 |

`long_e1_e3` (one seed) is why 1% became a main comparison point: the teacher lowered AbsRel by
~11% at 0.5% and 1%, but not at 5%. With few labels E1 memorized its images (training SILog
~0.04 while validation stopped improving); E3 did not.

---

## Run history notes

- **`main_1_5` reruns:** E1 1% (seeds 0–2) and E4 1% seed 0 finished before weights were saved
  (PR #10), so they were run again to keep their weights for the test set. The tables and
  `report.json` use the reruns. GPU training is not bit-exact, so they differ slightly from the
  first runs (E1 1% best 0.200 → 0.199, E4 seed 0 final 0.199 → 0.191). `run.log` has both.
- **`long_e1_e3`:** the session disconnected once during E3 at 5% and resumed from the
  step-17,000 checkpoint.
- **Commit hashes:** runs up to `fewlabel_v2_base` were made before the project used git, so
  their reports have none. `timing_test` records `0e01ce5`, which became `393908b` when the
  history was rewritten (same code).
