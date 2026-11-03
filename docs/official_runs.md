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
  RMSE, δ1–δ3 and log10. The report uses the final step as the main number, so that picking
  a checkpoint on validation does not make the numbers look better.
- The 654-image test set is used once, at the end, on the final models.

## Runs (Colab L4, one notebook run per group)

At ~0.07 s/step, one run takes ~75 min (the base student takes longer).

| Group (`RUN_NAME`) | Experiment | Runs | ~Hours |
|---|---|---|---|
| `main_1_5` | E1 vs E4 at 1% and 5%, seeds 0,1,2 | 12 | 15 |
| `ablation_1_5` | E2, E3 at 1% and 5%, seed 0 | 4 | 5 |
| `budgets_25_100` | E5: E1 vs E4 at 25% and 100%, seed 0 | 4 | 5 |
| `students_1` | E6: tiny and base students, E1 vs E4 at 1%, seed 0 | 4 | 6 |
| **Total** | | **24** | **~31** (≈ 60 compute units) |

E5's 1% and 5% points come from `main_1_5`. E6 runs at 1% because at 5% the teacher made
no difference in the pilot.

`ARGS` for each group (the common part first):

```python
COMMON = "--pool -1 --n-val 1000 --steps 63000 --eval-every 3000"
ARGS = f"{COMMON} --labeled 1%,5% --recipes gt,semi_edge --seeds 0,1,2"          # main_1_5
ARGS = f"{COMMON} --labeled 1%,5% --recipes gt_t,semi"                           # ablation_1_5
ARGS = f"{COMMON} --labeled 25%,100% --recipes gt,semi_edge"                     # budgets_25_100
ARGS = f"{COMMON} --labeled 1% --recipes gt,semi_edge --student tiny"            # students_1 (part 1)
ARGS = f"{COMMON} --labeled 1% --recipes gt,semi_edge --student base"            # students_1 (part 2)
```

For E6, run the two student sizes with different `RUN_NAME`s (`students_1_tiny`,
`students_1_base`), because a report holds one student size.

## Order

1. `main_1_5`: the main result. It takes ~15 h, so expect at least one disconnect.
   **Run all** resumes from the last checkpoint.
2. `ablation_1_5`
3. `budgets_25_100`
4. `students_1_tiny`, `students_1_base`
5. Test-set evaluation of the final models.

After each group: copy `results/<RUN_NAME>/` from Drive into the repo, open a PR,
and add the summary to the group's experiment issue.
