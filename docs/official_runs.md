# Official runs

The plan for the runs that go into the report. It follows the proposal (E1–E6, 63k steps,
3 seeds for the main comparison). One change comes from the pilot (`results/long_e1_e3`):
**1% labels is a second main comparison point, also with 3 seeds.** In the pilot the teacher
helped by ~11% at 0.5% and 1%, but not at 5%. So the report needs both points to show where
distillation helps and where it stops helping.

## Shared settings

- Full pool (46,248 images), validation **1,000** images from 25 held-out rooms. The pilot used
  300, and its numbers moved by ±0.01 between evaluations.
- 63,000 steps, evaluation every 3,000 steps, checkpoint every 1,000.
- Every report has the best checkpoint (`mean`) and the final step (`mean_last`), with AbsRel,
  RMSE, δ1–δ3 and log10.
- Model selection: the best checkpoint on validation is the main model (early stopping), and
  the final-step model is reported next to it. Both are kept in `.done.pt` since PR #10.
- The 654-image test set is used once, at the end, on both models. Validation numbers are only
  used to compare recipes and pick checkpoints.

## Runs (Colab L4, one notebook run per group)

Each group is a preset in `notebooks/colab_run.ipynb`: set `RUN_NAME` to it and **Run all**.
At ~0.07 s/step a 63k-step run takes ~75 min (the base student takes longer).

| `RUN_NAME` | What | Runs | ~Hours | Status |
|---|---|---|---|---|
| `main_1_5` | E1 vs E4 at 1% and 5%, seeds 0,1,2 | 12 | 15 | done (`results/main_1_5`) |
| `main_1_5` + `RERUN_WITHOUT_WEIGHTS` | the 4 runs that kept no weights (E1 1% seeds 0–2, E4 1% seed 0) | 4 | 5 | done |
| `fix_1_short` | E1 vs E4 at 1%, 24k steps (E4 first, E1 added after) | 6 | 2.5 | E4 done, E1 to do |
| `fix_1_decay` | E4 at 1%, teacher weight falls to 0 over the second half (`--teacher-decay 0.5`) | 3 | 4 | to do |
| `subsets_1_k1`, `subsets_1_k2` | E1 vs E4 at 1% on two other labeled subsets (`--label-subset`) | 4 | 5 | to do |
| `ablation_1_5` | E2, E3 at 1% and 5% | 4 | 5 | to do |
| `budgets_25_100` | E5: E1 vs E4 at 25% and 100% | 4 | 5 | to do |
| `students_1_tiny`, `students_1_base` | E6: E1 vs E4 at 1% with ~1M and ~4M students | 4 | 6 | to do |

**Why `fix_1`:** in `main_1_5`, E4 at 1% reached 0.181–0.184 AbsRel by 12k–45k steps on every
seed, but two seeds then got worse (final 0.199 and 0.212) while the predicted scale drifted.
`fix_1` checks whether lowering the teacher weight late in training, or stopping earlier, keeps
the gain until the end. If one of them works, the remaining groups use it.
E4 at 24k steps (first result): best 0.176 / RMSE 0.705 / δ1 0.725, final step 0.181 ± 0.003, with
no late degradation on any seed. E1 is also run at 24k steps, so the comparison uses the same
schedule for both recipes.

**Why `subsets_1`:** the three seeds of `main_1_5` share the same labeled images, so their spread
only measures training noise. Other subsets show how much the choice of labeled images matters.

## Test set

At the end, set `RUN_NAME = None` and `EVAL_RUNS` to the groups for the report, then **Run all**.
`scripts/evaluate_test.py` evaluates the best and the final model of every run on the 654 test
images (Eigen crop, 10 m cap). It reports raw metrics, metrics after median scaling (structure
only) and AbsRel/RMSE by distance (< 3 m, 3–6 m, > 6 m).

## Order

1. `main_1_5` with `RERUN_WITHOUT_WEIGHTS = True`
2. `fix_1_short`, then `fix_1_decay`
3. `subsets_1_k1`, `subsets_1_k2`
4. `ablation_1_5`, `budgets_25_100`, `students_1_tiny`, `students_1_base` (with the `fix_1` setting if it helps)
5. Test set

After each group: copy `results/<RUN_NAME>/` from Drive into the repo, open a PR,
and add the summary to the group's experiment issue. Keep `checkpoints/` on Drive.
