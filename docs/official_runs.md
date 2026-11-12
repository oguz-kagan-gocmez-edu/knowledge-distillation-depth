# Official runs

The plan for the runs that go into the report. It follows the proposal (E1–E6, 63k steps,
3 seeds for the main comparison). One change comes from the pilot (`results/long_e1_e3`):
**1% labels is a second main comparison point, also with 3 seeds.** In the pilot the teacher
helped by ~11% at 0.5% and 1%, but not at 5%. So the report needs both points to show where
distillation helps and where it stops helping.

## Shared settings

- Full pool (46,248 images), validation **1,000** images from 25 held-out rooms. The pilot used
  300, and its numbers moved by ±0.01 between evaluations.
- 24,000 steps (63,000 for `main_1_5`), evaluation every 3,000 steps, checkpoint every 1,000.
- Every report has the best checkpoint (`mean`) and the final step (`mean_last`), with AbsRel,
  RMSE, δ1–δ3 and log10.
- Model selection: the best checkpoint on validation is the main model (early stopping), and
  the final-step model is reported next to it. Both are kept in `.done.pt` since PR #10.
- The 654-image test set is used once, at the end, on both models. Validation numbers are only
  used to compare recipes and pick checkpoints.

## Runs (Colab L4, one notebook run per group)

Each group is a preset in `notebooks/colab_run.ipynb`: set `RUN_NAME` to it and **Run all**.
From `main_5_short` on, every group uses **24k steps** (see "Why `fix_1`" below). At ~0.07 s/step
a 24k-step E4 run takes ~30 min and an E1 run ~20 min (the base student takes longer). Two
notebooks with **different** `RUN_NAME`s can run at the same time on two L4 sessions.

| `RUN_NAME` | What | Steps | Runs | ~Hours | Status |
|---|---|---|---|---|---|
| `main_1_5` | E1 vs E4 at 1% and 5%, seeds 0,1,2 (4 runs rerun to keep weights) | 63k | 12 | 15 | done (`results/main_1_5`) |
| `fix_1_short` | E1 vs E4 at 1%, seeds 0,1,2 | 24k | 6 | 2.5 | done |
| `main_5_short` | E1 vs E4 at 5%, seeds 0,1,2 (same schedule as `fix_1_short`) | 24k | 6 | 2.5 | done |
| `subsets_1_k1`, `subsets_1_k2` | E1 vs E4 at 1% on two other labeled subsets (`--label-subset`) | 24k | 4 | 2 | done |
| `ablation_1_5` | E2, E3 at 1% and 5% | 24k | 4 | 2 | done |
| `budgets_25_100` | E5: E1 vs E4 at 25% and 100% | 24k | 4 | 2 | done |
| `students_1_tiny`, `students_1_base` | E6: E1 vs E4 at 1% with ~1M and ~4M students | 24k | 4 | 2.5 | done |
| `fix_1_decay` | optional: E4 at 1%, teacher weight falls to 0 over the second half | 63k | 3 | 4 | skipped (24k steps fixed the degradation) |
| `decay_1_5_25` | E4 at 1%, 5% and 25% with `--teacher-decay 0.5`: does the decay remove the harm with many labels? | 24k | 3 | 1.5 | done |
| `decay_100` | E4 at 100% with `--teacher-decay 0.5`, to complete the budget table | 24k | 1 | 0.5 | done |
| `gtmask_decay` | E4 + decay at 1–100% with `--pool-gt --lambda-t-lab 0`: real depth on every labeled image in the batch, teacher only on unlabeled ones | 24k | 4 | 1.5 | to do |
| `reg_1_aug` | E1 vs E4 at 1% with `--aug` (random zoom-in crop with depth divided by the zoom, color jitter) | 24k | 2 | 1 | done |
| `reg_1_wd` | E1 vs E4 at 1% with weight decay 0.01 instead of 1e-4 | 24k | 2 | 1 | done |
| `reg_1_drop` | E1 vs E4 at 1% with channel dropout 0.2 before the decoder head | 24k | 2 | 1 | done |

The main table of the report uses `fix_1_short` (1%) and `main_5_short` (5%). `main_1_5` stays in
the report as the long-training result that shows the late degradation.

**Why `fix_1`:** in `main_1_5`, E4 at 1% reached 0.181–0.184 AbsRel by 12k–45k steps on every
seed, but two seeds then got worse (final 0.199 and 0.212) while the predicted scale drifted.
`fix_1` checks whether lowering the teacher weight late in training, or stopping earlier, keeps
the gain until the end. If one of them works, the remaining groups use it.
Result (`fix_1_short`, 3 seeds): E1 at 24k is the same as at 63k (best 0.199, final 0.200 ± 0.001),
while E4 at 24k gives best 0.176 / RMSE 0.705 / δ1 0.725 and final 0.181 ± 0.003 (63k: 0.182 and
0.196 ± 0.011). With the same schedule the teacher lowers AbsRel by 11.2% (best) and 9.5% (final step).
So shorter training fixes the late degradation of E4 without helping E1, and the remaining groups
use 24k steps.

**Why `subsets_1`:** the three seeds of `main_1_5` share the same labeled images, so their spread
only measures training noise. Other subsets show how much the choice of labeled images matters.

## Test set

At the end, set `RUN_NAME = None` and `EVAL_RUNS` to the groups for the report, then **Run all**.
`scripts/evaluate_test.py` evaluates the best and the final model of every run on the 654 test
images (Eigen crop, 10 m cap). It reports raw metrics, metrics after median scaling (structure
only) and AbsRel/RMSE by distance (< 3 m, 3–6 m, > 6 m).

## Order

1. `main_5_short`
2. `subsets_1_k1`, `subsets_1_k2`
3. `ablation_1_5`, `budgets_25_100`, `students_1_tiny`, `students_1_base`
4. `fix_1_decay` if there is time
5. Test set (`EVAL_RUNS = ["main_1_5", "fix_1_short", "main_5_short", ...]`)
6. Analysis: AbsRel against epochs over the labeled images, from the validation curves (no new runs)

After each group: copy `results/<RUN_NAME>/` from Drive into the repo, open a PR,
and add the summary to the group's experiment issue. Keep `checkpoints/` on Drive.

**Why `reg_1`:** at 1% labels E1 memorizes its 462 images (training SILog ~0.07 while validation
stops improving). These groups compare the usual regularizers with the teacher, one at a time, so
the report can show how much each one helps and whether they add to the teacher's gain. Compare
with `fix_1_short` seed 0. `--aug` follows Eigen et al. (2014) for the zoom and Depth Anything for
perturbing only the student's input; rotation is left out because it would leave empty corners in
the teacher target. All three options keep the old run signatures when left at their defaults.

**Why `gtmask_decay`:** in E3/E4 the batch is 4 labeled images plus 4 drawn from the whole pool.
Until now the real depth of pool images was not used even when they were labeled, and the teacher
loss was applied to every image. So at 100% labels E4 saw 4 labeled images per step against 8 for
E1, and the teacher pulled against the real depth on every labeled pixel. `--pool-gt` uses the
real depth of every labeled image in the batch and `--lambda-t-lab 0` turns the teacher off on
them (Hu et al. use 0.1 there). Compare with `decay_1_5_25` and `decay_100`. At 100% every image
is labeled, so the teacher is off entirely and the run should match E1: a sanity check.

