# Results

Each folder holds one run group: `report.json` (config, metrics, git commit), `examples.png`
(predictions next to the real depth) and `run.log` (console output, when saved).

All numbers on this page are **validation** numbers. The 654-image NYU test set has not been
used yet; it is evaluated once, at the end (see [`docs/official_runs.md`](../docs/official_runs.md)).

## Key findings

1. **With 1% labels, the teacher lowers AbsRel by about 11%** (0.200 → 0.178). The gain is the
   same on three different sets of labeled images and three seeds.
2. **The gain comes from the teacher on the unlabeled images.** Using the teacher only on the
   labeled images does nothing, and the edge loss adds nothing measurable.
3. **The gain holds for every student size** (1M, 3M and 4M parameters: −9%, −11%, −8.5%).
   The 3M student with the teacher beats the 4M student without it.
4. **With 5% labels or more, the teacher hurts** (+2.5% at 5%, +7–8% at 25% and 100%).
   The student with the teacher stays at AbsRel ~0.165–0.170 however many labels it gets.
5. **The teacher helps where the student memorizes its labels.** With 462 labeled images each one
   is seen ~400 times and validation stops improving early; with all labels it is seen ~4 times
   and the label-only student keeps improving to the last step.
6. **Long training hurts the teacher recipe at 1%** (63k steps: final AbsRel 0.196 ± 0.011).
   Training for 24k steps fixes it (0.181 ± 0.003), so all later runs use 24k steps.

## Setup

| | |
|---|---|
| Data | NYU Depth V2 (DenseDepth copy): pool of 46,248 training images |
| Validation | 1,000 images from 25 rooms held out of training |
| Teacher | Depth Anything V2 Small (24.8M parameters), frozen |
| Students | tiny ~1M (MobileNetV3-Small), **small 3.15M (MobileNetV3-Large, default)**, base ~4M (EfficientNet-B0) |
| Training | 24k steps, batch 8, AdamW, LR 5e-4, one-cycle schedule, Colab L4 GPU |
| Model selection | best validation checkpoint ("best"); the last step ("final") is reported next to it |

**Recipes**

| ID | Name | Training signal |
|---|---|---|
| E1 | `gt` | real depth on the labeled images only (baseline) |
| E2 | `gt_t` | real depth + teacher, labeled images only |
| E3 | `semi` | real depth on the labeled images + teacher on all images |
| E4 | `semi_edge` | E3 + edge loss (the main method) |

**Reference points** (same validation set)

| Model | AbsRel ↓ | RMSE ↓ | δ<1.25 ↑ |
|---|---|---|---|
| Constant depth (sanity check) | 0.402 | 1.345 | 0.343 |
| Teacher, aligned to real depth per image (optimistic upper bound) | 0.110 | 0.519 | 0.881 |

AbsRel and RMSE: lower is better. δ<1.25: share of pixels within 25% of the real depth, higher is better.
"Change" columns compare E4 with E1 on AbsRel; negative means the teacher helps.

## 1. Main comparison: 1% and 5% labels

Small student, 24k steps, mean over seeds 0, 1, 2.

| Labels | Recipe | AbsRel (best) | RMSE | δ<1.25 | AbsRel (final) | Change | Folder |
|---|---|---|---|---|---|---|---|
| 1% (462) | E1 | 0.199 | 0.712 | 0.676 | 0.200 ± 0.001 | | `fix_1_short` |
| | E4 | **0.176** | **0.705** | **0.725** | **0.181 ± 0.003** | **−11.2%** | |
| 5% (2,312) | E1 | **0.166** | **0.635** | **0.744** | **0.166 ± 0.001** | | `main_5_short` |
| | E4 | 0.170 | 0.675 | 0.736 | 0.172 ± 0.000 | +2.5% | |

## 2. Does it depend on which images are labeled?

1% labels, small student, 24k steps. Subset 0 is the main subset (3 seeds); subsets 1 and 2 use one seed.

| Labeled subset | E1 | E4 | Change | Folder |
|---|---|---|---|---|
| 0 | 0.199 | 0.176 | −11.2% | `fix_1_short` |
| 1 | 0.202 | 0.180 | −11.1% | `subsets_1_k1` |
| 2 | 0.199 | 0.177 | −10.9% | `subsets_1_k2` |

No: the gain is 11% on every subset.

## 3. Which part of the method helps? (ablation)

Small student, 24k steps. E2 and E3 use one seed (`ablation_1_5`); E1 and E4 are from section 1.

| Labels | E1 labels only | E2 + teacher on labeled images | E3 + teacher on all images | E4 = E3 + edge loss |
|---|---|---|---|---|
| 1% | 0.199 | 0.195 | 0.177 | **0.176** |
| 5% | **0.166** | 0.174 | 0.172 | 0.170 |

At 1%, E2 is the same as E1 and E3 is the same as E4: the gain comes from the teacher's targets on
the unlabeled images, and the edge loss does not change the result.

## 4. Student size

1% labels, 24k steps. tiny and base use one seed.

| Student | Parameters | E1 | E4 | Change | E4 final | Folder |
|---|---|---|---|---|---|---|
| tiny | ~1M | 0.224 | 0.203 | −9.3% | 0.207 | `students_1_tiny` |
| small | 3.15M | 0.199 | 0.176 | −11.2% | 0.181 | `fix_1_short` |
| base | ~4M | 0.190 | 0.174 | −8.5% | 0.184 | `students_1_base` |

The teacher helps at every size. With the teacher, base is no better than small, and small with the
teacher (0.176) beats base without it (0.190). Base E1 overfits: validation rises after 12k steps
(0.190 → 0.196) while its training loss falls to 0.046.

## 5. Label budget: when does the teacher stop helping?

Small student, 24k steps. 25% and 100% use one seed (`budgets_25_100`).

| Labels | Images | Times each labeled image is seen (E1) | E1 | E4 | Change |
|---|---|---|---|---|---|
| 1% | 462 | ~415 | 0.199 | **0.176** | **−11.2%** |
| 5% | 2,312 | ~83 | **0.166** | 0.170 | +2.5% |
| 25% | 11,562 | ~17 | **0.156** | 0.168 | +7.7% |
| 100% | 46,248 | ~4 | **0.153** | 0.164 | +7.2% |

E1 improves steadily with more labels; E4 stays at ~0.165–0.170 and its training loss on the labels
stays at ~0.18 (E1: ~0.09). At 25% and 100% E1 is still improving at the last step.

## 6. Training length

1% labels, small student, mean over 3 seeds.

| Steps | E1 best / final | E4 best / final | Folder |
|---|---|---|---|
| 63k | 0.199 / 0.205 ± 0.003 | 0.182 / 0.196 ± 0.011 | `main_1_5` |
| 24k | 0.199 / 0.200 ± 0.001 | 0.176 / 0.181 ± 0.003 | `fix_1_short` |

At 63k steps E4 reaches its best value early and then gets worse on two of three seeds while its
predicted scale drifts. At 5% the 63k results match the 24k ones (E1 0.165, E4 0.166).

## All run folders

| Folder | What | Validation | Status |
|---|---|---|---|
| `main_1_5` | E1, E4 at 1% and 5%, 63k steps, 3 seeds | 1,000 | official |
| `fix_1_short` | E1, E4 at 1%, 24k steps, 3 seeds | 1,000 | official |
| `main_5_short` | E1, E4 at 5%, 24k steps, 3 seeds | 1,000 | official |
| `subsets_1_k1`, `subsets_1_k2` | E1, E4 at 1% on other labeled subsets | 1,000 | official |
| `ablation_1_5` | E2, E3 at 1% and 5% | 1,000 | official |
| `students_1_tiny`, `students_1_base` | E1, E4 at 1% with the ~1M and ~4M students | 1,000 | official |
| `budgets_25_100` | E1, E4 at 25% and 100% | 1,000 | official |
| `long_e1_e3` | pilot: E1, E3 at 0.5%, 1%, 5%, 30k steps | 300 | pilot |
| `timing_test` | speed check, 500 steps | 300 | pilot |
| `fewlabel_v2*`, `fewlabel_v1`, `feasibility` | early tests on 5,000 images | 300 / 200 | pilot |

## Pilot runs

Earlier tests used a 300-image validation set (`feasibility`: 200 random images), so their numbers
are **not comparable** with the tables above. They were used to choose the setup of the official runs.

| Folder | Setting | Recipe | AbsRel | RMSE | δ<1.25 |
|---|---|---|---|---|---|
| `feasibility` | 2,000 images, all labeled, 1.5k steps | gt | 0.202 | 0.675 | 0.675 |
| | | full | 0.204 | 0.677 | 0.673 |
| `fewlabel_v1` | 250 / 5,000 labeled, 2k steps | E1 | 0.212 | 0.770 | 0.622 |
| | | E2 | 0.252 | 0.860 | 0.572 |
| | | E3 | 0.222 | 0.787 | 0.624 |
| | | E4 | 0.241 | 0.842 | 0.592 |
| `fewlabel_v2` | 50 / 5,000 labeled | E1 | 0.266 | 0.898 | 0.535 |
| | | E3 | 0.252 | 0.855 | 0.580 |
| | 250 / 5,000 labeled | E1 | 0.219 | 0.781 | 0.609 |
| | | E3 | 0.204 | 0.739 | 0.666 |
| `fewlabel_v2_seeds` | 250 / 5,000, 3 seeds (mean) | E1 | 0.221 | 0.781 | 0.608 |
| | | E3 | 0.211 | 0.757 | 0.648 |
| | | E4 | 0.209 | 0.758 | 0.653 |
| `fewlabel_v2_tiny` | 250 / 5,000, tiny student | E1 | 0.243 | 0.852 | 0.566 |
| | | E3 | 0.241 | 0.839 | 0.586 |
| `fewlabel_v2_base` | 250 / 5,000, base student | E1 | 0.208 | 0.764 | 0.620 |
| | | E3 | 0.202 | 0.740 | 0.670 |
| `timing_test` | 5% of 46,248, 500 steps (speed check only) | E1 | 0.212 | 0.759 | 0.630 |
| `long_e1_e3` | 0.5% of 46,248, 30k steps | E1 | 0.213 | 0.781 | 0.630 |
| | | E3 | 0.188 | 0.751 | 0.680 |
| | 1%, 30k steps | E1 | 0.191 | 0.690 | 0.692 |
| | | E3 | 0.170 | 0.698 | 0.726 |
| | 5%, 30k steps | E1 | 0.164 | 0.631 | 0.751 |
| | | E3 | 0.163 | 0.644 | 0.760 |

Runs up to `fewlabel_v2_base` were made on an Apple M4 Pro; `timing_test` and later on a Colab L4.

## Notes

- **Reruns in `main_1_5`.** E1 at 1% (seeds 0–2) and E4 at 1% (seed 0) finished before weights were
  saved, so they were run again to keep their weights for the test set. The tables use the reruns.
  GPU training is not bit-exact, so they differ slightly from the first runs (E1 1% best 0.200 → 0.199,
  E4 seed 0 final 0.199 → 0.191); `run.log` has both.
- **Per-seed values.** `main_1_5` E4 1% final: 0.191 / 0.212 / 0.185. `fix_1_short` E4 1% final:
  0.179 / 0.185 / 0.178.
- **One-seed runs.** Differences below ~3% may be noise.
- **Constant-depth reference.** It uses the median depth of the labeled images, so it moves slightly
  with the labeled set (0.406 in `budgets_25_100`).
- **Commit hashes.** Runs up to `fewlabel_v2_base` were made before the project used git and have none.
  `timing_test` records `0e01ce5`, which became `393908b` when the history was rewritten (same code).
- **Resumed run.** In `long_e1_e3` the session disconnected once during E3 at 5% and resumed from the
  step-17,000 checkpoint.
