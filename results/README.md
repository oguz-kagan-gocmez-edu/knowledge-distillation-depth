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
2. [All official results](#all-official-results): every run with every metric
3. [Pilot runs](#pilot-runs): 300 validation images (not comparable with the official runs)
4. [Run history notes](#run-history-notes)

---

## Official runs

Full pool of 46,248 images, validation on **1,000 images from 25 held-out rooms**, small student
and 24k steps unless noted. The plan is in [`docs/official_runs.md`](../docs/official_runs.md).

**Reference points** (same 1,000 validation images)

| Reference | AbsRel ↓ | RMSE ↓ | δ1 ↑ | δ2 ↑ | δ3 ↑ | log10 ↓ |
|---|---|---|---|---|---|---|
| Constant depth (median of the labeled images) | 0.402 | 1.345 | 0.343 | 0.635 | 0.812 | 0.170 |
| Teacher, aligned to real depth per image (optimistic) | 0.110 | 0.519 | 0.881 | 0.966 | 0.985 | 0.049 |

The constant-depth reference uses the median depth of the labeled images, so it moves slightly
with the labeled set (0.406 in `budgets_25_100`). The teacher reference scales and shifts each
teacher prediction to the real depth of that image, which a student never sees, so it is an
optimistic upper bound.

**Model size and speed** (measured in `feasibility`, input 256×192)

| Model | Parameters | Times smaller than the teacher | GFLOPs | Latency |
|---|---|---|---|---|
| Teacher: Depth Anything V2 Small | 24.8M | 1× | – | – |
| Student tiny (MobileNetV3-Small) | ~1M | ~25× | – | – |
| **Student small (MobileNetV3-Large)** | **3.15M** | **~8×** | **0.74** | **8.6 ms (M4 Pro GPU), 123 ms (CPU)** |
| Student base (EfficientNet-B0) | ~4M | ~6× | – | – |

The tiny and base sizes are design targets; they are measured with the test set.

### Summary: effect of the teacher by label budget

| Labeled | Images | Seeds | E1 gt | E4 semi_edge | E4 vs E1 | E4 + teacher decay | vs E1 | + GT mask | vs E1 | Folders |
|---|---|---|---|---|---|---|---|---|---|---|
| 1% | 462 | 3 | 0.199 | 0.176 | −11.2% | **0.175** | **−12.1%** | 0.189 | −5.0% | `fix_1_short`, `decay_1_5_25`, `gtmask_decay` |
| 5% | 2,312 | 3 | 0.166 | 0.170 | +2.5% | 0.163 | −1.8% | **0.159** | **−4.2%** | `main_5_short`, `decay_1_5_25`, `gtmask_decay` |
| 25% | 11,562 | 1 | 0.156 | 0.168 | +7.7% | 0.159 | +1.9% | **0.154** | **−1.3%** | `budgets_25_100`, `decay_1_5_25`, `gtmask_decay` |
| 100% | 46,248 | 1 | **0.153** | 0.164 | +7.2% | 0.157 | +2.6% | 0.157 | +2.6% | `budgets_25_100`, `decay_100`, `gtmask_decay` |

Best-checkpoint AbsRel; teacher decay runs have one seed. With a constant weight the teacher helps
with very few labels and hurts with many. Letting its weight fall to 0 over the second half
(`--teacher-decay 0.5`) keeps the gain at 1% and removes most of the harm at 5% to 100%.
Also using the real depth of every labeled image in the batch and turning the teacher off on them
(GT mask) gives the best 5% and 25% results, but loses half of the gain at 1%.

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

### `reg_1_aug`, `reg_1_wd`, `reg_1_drop`: usual regularizers vs the teacher (1%, seed 0)

One option at a time, all else as in `fix_1_short`. Training SILog is the labeled loss at the last
step (lower = the labels are fit more closely).

| Option | E1 gt | E4 semi_edge | E4 vs E1 | Final E1 → E4 | Training SILog E1 / E4 |
|---|---|---|---|---|---|
| none (`fix_1_short`, 3 seeds) | 0.199 | 0.176 | −11.2% | 0.200 → 0.181 | ~0.07 / – |
| `--aug`: zoom-in crop ×1–1.25 and color jitter (`reg_1_aug`) | **0.195** | **0.173** | −11.0% | 0.195 → 0.183 | 0.064 / 0.125 |
| `--weight-decay 0.01` instead of 1e-4 (`reg_1_wd`) | 0.198 | **0.173** | −12.9% | 0.199 → 0.175 | 0.046 / 0.139 |
| `--dropout 0.2` before the decoder head (`reg_1_drop`) | 0.204 | 0.192 | −5.7% | 0.206 → 0.206 | 0.063 / 0.151 |

- None of the three stops E1 from fitting its 462 images: its training SILog still ends at
  0.05–0.06 and validation stays at 0.195–0.204. The best, augmentation, gains about 2%, within
  the noise of a single seed.
- Weight decay changes nothing. Most weights sit before a BatchNorm layer, so shrinking them does
  not change what the network computes.
- Dropout hurts E4 (0.176 → 0.192, best at 9k steps). E4 already fits the labels loosely because of
  the teacher; adding dropout makes it underfit.
- The teacher's gain (−11 to −13%) stays the same with augmentation or weight decay, so it adds
  to them. Augmentation with the teacher gives the lowest 1% result so far (0.173).

### `decay_1_5_25`, `decay_100`: teacher weight falls to 0 over the second half (E4, seed 0)

`--teacher-decay 0.5`: after the usual ramp the teacher weight stays at 1 until 12k steps, then
falls linearly to 0 at 24k. E1 and E4 without decay are the 3-seed means of the summary table
(25% and 100%: seed 0 of `budgets_25_100`).

| Labeled | E1 gt | E4 | E4 + decay | Final AbsRel | Best at | Labeled SILog at 12k → 24k |
|---|---|---|---|---|---|---|
| 1% | 0.199 | 0.176 | **0.175** | 0.183 | 15k | 0.162 → 0.098 |
| 5% | 0.166 | 0.170 | **0.163** | 0.163 | 24k | 0.186 → 0.126 |
| 25% | **0.156** | 0.168 | 0.159 | 0.159 | 24k | 0.196 → 0.138 |
| 100% (`decay_100`) | **0.153** | 0.164 | 0.157 | 0.157 | 24k | 0.198 → 0.142 |

- At 1% the student starts to fit its labels once the teacher fades (training SILog 0.162 → 0.098)
  and validation rises again (0.175 → 0.183): without the teacher the overfitting comes back.
- At 5% to 100% validation keeps improving to the last step: here the teacher had been keeping
  the student from fitting the labels, and the decay lifts that limit.
- With decay E4 is never clearly worse than E1 (all differences within ~2%, one seed).
- At 25% the predicted scale drifts to ×0.93 at the last step.

### `gtmask_decay`: teacher only on unlabeled images (E4 + decay, seed 0)

`--pool-gt --lambda-t-lab 0` on top of `--teacher-decay 0.5`: SILog on every labeled image in the
batch (also those drawn from the pool), teacher and edge losses only on unlabeled images.

| Labeled | E1 gt | E4 + decay | + GT mask | Final AbsRel | Best at | Labeled SILog at 12k → 24k |
|---|---|---|---|---|---|---|
| 1% | 0.199 | **0.175** | 0.189 | 0.216 | 9k | 0.121 → 0.078 |
| 5% | 0.166 | 0.163 | **0.159** | 0.160 | 21k | 0.164 → 0.109 |
| 25% | 0.156 | 0.159 | **0.154** | 0.154 | 21k | 0.169 → 0.119 |
| 100% | **0.153** | 0.157 | 0.157 | 0.157 | 24k | 0.122 → 0.091 |

- At 1% the teacher on the labeled images was what kept the student from memorizing them. Without
  it training SILog falls to 0.078 and validation rises from 9k steps on (0.189 → 0.216).
- At 5% and 25% this is the best setting so far, 1–3% below E4 + decay (one seed, within noise).
- At 100% every image is labeled, so the teacher and edge losses are 0 for the whole run and it is
  E1 with a different sampler. It gets 0.157 against 0.153 for E1: two runs of nearly the same
  setup differ by ~2.6%, a direct estimate of seed-to-seed noise.

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

## All official results

Every official run with every metric, taken from the `report.json` files. Mean (± std) over seeds
when there are several. **Best** = the checkpoint with the lowest validation AbsRel ("Best at" is
its step for each seed); **final** = the last step. δ1–δ3: share of pixels within 1.25, 1.25² and
1.25³ of the real depth.

| Folder | Labeled | Recipe | Student | Steps | Seeds | AbsRel ↓ | RMSE ↓ | δ1 ↑ | δ2 ↑ | δ3 ↑ | log10 ↓ | Best at | Final AbsRel ↓ | Final RMSE ↓ | Final δ1 ↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `main_1_5` | 1% (462) | E1 gt | small | 63k | 3 | 0.199 ± 0.002 | 0.713 | 0.676 | 0.914 | 0.977 | 0.084 | 15k/21k/24k | 0.205 ± 0.003 | 0.727 | 0.664 |
| `main_1_5` | 1% (462) | E4 semi_edge | small | 63k | 3 | 0.182 ± 0.002 | 0.712 | 0.710 | 0.919 | 0.971 | 0.081 | 21k/12k/45k | 0.196 ± 0.011 | 0.743 | 0.684 |
| `main_1_5` | 5% (2,312) | E1 gt | small | 63k | 3 | 0.165 ± 0.001 | 0.634 | 0.748 | 0.940 | 0.984 | 0.071 | 30k/30k/36k | 0.167 ± 0.001 | 0.638 | 0.742 |
| `main_1_5` | 5% (2,312) | E4 semi_edge | small | 63k | 3 | 0.166 ± 0.001 | 0.672 | 0.744 | 0.929 | 0.976 | 0.074 | 18k/51k/39k | 0.171 ± 0.002 | 0.682 | 0.727 |
| `fix_1_short` | 1% (462) | E1 gt | small | 24k | 3 | 0.199 ± 0.001 | 0.712 | 0.676 | 0.914 | 0.978 | 0.084 | 15k/18k/21k | 0.200 ± 0.001 | 0.711 | 0.676 |
| `fix_1_short` | 1% (462) | E4 semi_edge | small | 24k | 3 | 0.176 ± 0.001 | 0.705 | 0.725 | 0.923 | 0.973 | 0.078 | 21k/15k/21k | 0.181 ± 0.003 | 0.705 | 0.717 |
| `main_5_short` | 5% (2,312) | E1 gt | small | 24k | 3 | 0.166 ± 0.001 | 0.635 | 0.744 | 0.941 | 0.986 | 0.072 | 18k/21k/18k | 0.166 ± 0.001 | 0.635 | 0.744 |
| `main_5_short` | 5% (2,312) | E4 semi_edge | small | 24k | 3 | 0.170 ± 0.001 | 0.675 | 0.736 | 0.930 | 0.976 | 0.075 | 18k/21k/24k | 0.172 ± 0.000 | 0.672 | 0.736 |
| `subsets_1_k1` | 1% (462) | E1 gt | small | 24k | 1 | 0.202 | 0.738 | 0.657 | 0.908 | 0.976 | 0.086 | 18k | 0.202 | 0.737 | 0.657 |
| `subsets_1_k1` | 1% (462) | E4 semi_edge | small | 24k | 1 | 0.180 | 0.718 | 0.723 | 0.924 | 0.973 | 0.079 | 21k | 0.180 | 0.721 | 0.721 |
| `subsets_1_k2` | 1% (462) | E1 gt | small | 24k | 1 | 0.199 | 0.708 | 0.684 | 0.915 | 0.975 | 0.082 | 15k | 0.201 | 0.709 | 0.679 |
| `subsets_1_k2` | 1% (462) | E4 semi_edge | small | 24k | 1 | 0.177 | 0.683 | 0.735 | 0.930 | 0.973 | 0.076 | 15k | 0.180 | 0.689 | 0.732 |
| `ablation_1_5` | 1% (462) | E2 gt_t | small | 24k | 1 | 0.195 | 0.737 | 0.676 | 0.909 | 0.969 | 0.086 | 21k | 0.199 | 0.718 | 0.685 |
| `ablation_1_5` | 1% (462) | E3 semi | small | 24k | 1 | 0.177 | 0.710 | 0.713 | 0.916 | 0.972 | 0.080 | 21k | 0.180 | 0.699 | 0.717 |
| `ablation_1_5` | 5% (2,312) | E2 gt_t | small | 24k | 1 | 0.174 | 0.690 | 0.723 | 0.925 | 0.977 | 0.077 | 21k | 0.176 | 0.685 | 0.720 |
| `ablation_1_5` | 5% (2,312) | E3 semi | small | 24k | 1 | 0.172 | 0.699 | 0.726 | 0.923 | 0.974 | 0.078 | 21k | 0.176 | 0.677 | 0.732 |
| `students_1_tiny` | 1% (462) | E1 gt | tiny | 24k | 1 | 0.224 | 0.785 | 0.614 | 0.892 | 0.970 | 0.094 | 15k | 0.225 | 0.778 | 0.618 |
| `students_1_tiny` | 1% (462) | E4 semi_edge | tiny | 24k | 1 | 0.203 | 0.744 | 0.675 | 0.905 | 0.968 | 0.086 | 12k | 0.207 | 0.747 | 0.663 |
| `students_1_base` | 1% (462) | E1 gt | base | 24k | 1 | 0.190 | 0.682 | 0.701 | 0.927 | 0.981 | 0.079 | 12k | 0.196 | 0.685 | 0.695 |
| `students_1_base` | 1% (462) | E4 semi_edge | base | 24k | 1 | 0.174 | 0.670 | 0.749 | 0.931 | 0.974 | 0.075 | 18k | 0.184 | 0.693 | 0.736 |
| `budgets_25_100` | 25% (11,562) | E1 gt | small | 24k | 1 | 0.156 | 0.622 | 0.762 | 0.947 | 0.987 | 0.069 | 24k | 0.156 | 0.622 | 0.762 |
| `budgets_25_100` | 25% (11,562) | E4 semi_edge | small | 24k | 1 | 0.168 | 0.670 | 0.743 | 0.931 | 0.976 | 0.074 | 21k | 0.171 | 0.683 | 0.733 |
| `budgets_25_100` | 100% (46,248) | E1 gt | small | 24k | 1 | 0.153 | 0.595 | 0.781 | 0.954 | 0.987 | 0.066 | 24k | 0.153 | 0.595 | 0.781 |
| `budgets_25_100` | 100% (46,248) | E4 semi_edge | small | 24k | 1 | 0.164 | 0.661 | 0.748 | 0.934 | 0.977 | 0.073 | 21k | 0.166 | 0.666 | 0.745 |
| `reg_1_aug` | 1% (462) | E1 gt | small | 24k | 1 | 0.195 | 0.673 | 0.705 | 0.926 | 0.980 | 0.079 | 24k | 0.195 | 0.673 | 0.705 |
| `reg_1_aug` | 1% (462) | E4 semi_edge | small | 24k | 1 | 0.173 | 0.663 | 0.739 | 0.932 | 0.979 | 0.075 | 18k | 0.183 | 0.668 | 0.730 |
| `reg_1_wd` | 1% (462) | E1 gt | small | 24k | 1 | 0.198 | 0.716 | 0.668 | 0.911 | 0.977 | 0.085 | 21k | 0.199 | 0.712 | 0.670 |
| `reg_1_wd` | 1% (462) | E4 semi_edge | small | 24k | 1 | 0.173 | 0.698 | 0.731 | 0.922 | 0.973 | 0.078 | 21k | 0.175 | 0.690 | 0.730 |
| `reg_1_drop` | 1% (462) | E1 gt | small | 24k | 1 | 0.204 | 0.729 | 0.662 | 0.908 | 0.976 | 0.086 | 21k | 0.206 | 0.724 | 0.665 |
| `reg_1_drop` | 1% (462) | E4 semi_edge | small | 24k | 1 | 0.192 | 0.710 | 0.687 | 0.917 | 0.974 | 0.082 | 9k | 0.206 | 0.709 | 0.680 |
| `decay_1_5_25` | 1% (462) | E4 semi_edge + decay | small | 24k | 1 | 0.175 | 0.680 | 0.735 | 0.926 | 0.974 | 0.076 | 15k | 0.183 | 0.724 | 0.720 |
| `decay_1_5_25` | 5% (2,312) | E4 semi_edge + decay | small | 24k | 1 | 0.163 | 0.631 | 0.749 | 0.945 | 0.987 | 0.071 | 24k | 0.163 | 0.631 | 0.749 |
| `decay_1_5_25` | 25% (11,562) | E4 semi_edge + decay | small | 24k | 1 | 0.159 | 0.630 | 0.752 | 0.947 | 0.986 | 0.071 | 24k | 0.159 | 0.630 | 0.752 |
| `decay_100` | 100% (46,248) | E4 semi_edge + decay | small | 24k | 1 | 0.157 | 0.628 | 0.761 | 0.948 | 0.988 | 0.069 | 24k | 0.157 | 0.628 | 0.761 |
| `gtmask_decay` | 1% (462) | E4 semi_edge + decay + GT mask | small | 24k | 1 | 0.189 | 0.736 | 0.683 | 0.914 | 0.972 | 0.085 | 9k | 0.216 | 0.887 | 0.628 |
| `gtmask_decay` | 5% (2,312) | E4 semi_edge + decay + GT mask | small | 24k | 1 | 0.159 | 0.654 | 0.753 | 0.938 | 0.985 | 0.071 | 21k | 0.160 | 0.640 | 0.760 |
| `gtmask_decay` | 25% (11,562) | E4 semi_edge + decay + GT mask | small | 24k | 1 | 0.154 | 0.625 | 0.766 | 0.948 | 0.988 | 0.068 | 21k | 0.154 | 0.626 | 0.764 |
| `gtmask_decay` | 100% (46,248) | E4 semi_edge + decay + GT mask | small | 24k | 1 | 0.157 | 0.606 | 0.769 | 0.948 | 0.987 | 0.068 | 24k | 0.157 | 0.606 | 0.769 |

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
