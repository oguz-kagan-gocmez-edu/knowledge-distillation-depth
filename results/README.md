# Results index

Each folder has `report.json` (full config + metrics), `examples.png`, and `run.log` when the
console output was saved. From now on `report.json` also records the git commit under `"run"`.

Validation = 300 images from 25 held-out rooms (feasibility: 200 random images, not split by room).
The 654 test images have not been used yet. All runs: Depth Anything V2 Small teacher. Runs up to `fewlabel_v2_base` were made on an Apple M4 Pro (MPS);
`timing_test` and later ran on a Colab L4 GPU.

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

| `timing_test` | `fewlabel_v2.py` | 2,312 / 46,248 labeled (5%), 500 steps, small (speed check only) | E1 gt | 0.212 | 0.759 | 0.630 |
| `long_e1_e3` | `fewlabel_v2.py` | 231 / 46,248 (0.5%), 30k steps, small | E1 gt | 0.213 | 0.781 | 0.630 |
| | | | E3 semi | 0.188 | 0.751 | 0.680 |
| | | 462 / 46,248 (1%), 30k steps, small | E1 gt | 0.191 | 0.690 | 0.692 |
| | | | E3 semi | 0.170 | 0.698 | 0.726 |
| | | 2,312 / 46,248 (5%), 30k steps, small | E1 gt | 0.164 | 0.631 | 0.751 |
| | | | E3 semi | 0.163 | 0.644 | 0.760 |

| `main_1_5` | `fewlabel_v2.py` | 462 / 46,248 (1%), 63k steps, val 1,000, seeds 0,1,2 (mean) | E1 gt | 0.199 | 0.713 | 0.676 |
| | | | E4 semi_edge | 0.182 | 0.712 | 0.710 |
| | | 2,312 / 46,248 (5%), 63k steps, val 1,000, seeds 0,1,2 (mean) | E1 gt | 0.165 | 0.634 | 0.748 |
| | | | E4 semi_edge | 0.166 | 0.672 | 0.744 |
| `fix_1_short` | `fewlabel_v2.py` | 462 / 46,248 (1%), **24k steps**, val 1,000, seeds 0,1,2 (mean) | E1 gt | 0.199 | 0.712 | 0.676 |
| | | | E4 semi_edge | 0.176 | 0.705 | 0.725 |
| `main_5_short` | `fewlabel_v2.py` | 2,312 / 46,248 (5%), 24k steps, val 1,000, seeds 0,1,2 (mean) | E1 gt | 0.166 | 0.635 | 0.744 |
| | | | E4 semi_edge | 0.170 | 0.675 | 0.736 |
| `subsets_1_k1` | `fewlabel_v2.py` | 462 / 46,248 (1%), other labeled subset (`--label-subset 1`), 24k steps, seed 0 | E1 gt | 0.202 | 0.738 | 0.657 |
| | | | E4 semi_edge | 0.180 | 0.718 | 0.723 |
| `subsets_1_k2` | `fewlabel_v2.py` | 462 / 46,248 (1%), third labeled subset (`--label-subset 2`), 24k steps, seed 0 | E1 gt | 0.199 | 0.708 | 0.684 |
| | | | E4 semi_edge | 0.177 | 0.683 | 0.735 |
| `ablation_1_5` | `fewlabel_v2.py` | 462 / 46,248 (1%), 24k steps, seed 0 | E2 gt_t | 0.195 | 0.737 | 0.676 |
| | | | E3 semi | 0.177 | 0.710 | 0.713 |
| | | 2,312 / 46,248 (5%), 24k steps, seed 0 | E2 gt_t | 0.174 | 0.690 | 0.723 |
| | | | E3 semi | 0.172 | 0.699 | 0.726 |
| `students_1_tiny` | `fewlabel_v2.py` | 462 / 46,248 (1%), tiny student (~1M), 24k steps, seed 0 | E1 gt | 0.224 | 0.785 | 0.614 |
| | | | E4 semi_edge | 0.203 | 0.744 | 0.675 |

`main_1_5` is the first official run group (see `docs/official_runs.md`). Its validation set has 1,000 images,
so its numbers are not comparable with the 300-image runs above. Final-step AbsRel (mean ± std over seeds):
1% E1 0.205 ± 0.003, E4 0.196 ± 0.011 (0.191 / 0.212 / 0.185); 5% E1 0.167 ± 0.001, E4 0.171 ± 0.002.
At 1% the teacher lowers best-checkpoint AbsRel by 8.7% on every seed (best 0.180–0.184 vs 0.196–0.202),
but E4 can degrade with long training while its scale drifts (seed 1). At 5% it gives no gain and a 6% worse RMSE.
E1 1% seeds 0–2 and E4 1% seed 0 finished before weights were saved (PR #10), so they were run again
to keep their weights for the test set. The table and `report.json` use the reruns. GPU training is not
bit-exact, so the reruns differ slightly from the first runs (E1 1% best 0.200 → 0.199, E4 seed 0
final 0.199 → 0.191). `run.log` has both. All 12 runs now have their best and final weights.

`fix_1_short` trains for 24k instead of 63k steps. Final-step AbsRel: E1 0.200 ± 0.001, E4 0.181 ± 0.003
(0.179 / 0.185 / 0.178). E1 is the same as at 63k, while E4 no longer degrades late: the teacher lowers
AbsRel by 11.2% at the best checkpoint and 9.5% at the final step. The remaining groups use 24k steps.

`main_5_short` (5%, 24k steps): final-step AbsRel E1 0.166 ± 0.001, E4 0.172 ± 0.000. As at 63k steps, the
teacher does not help at 5% (+2.5% AbsRel, 6% worse RMSE), so this is not caused by late degradation.
E4 fits the labels less (training SILog 0.165 vs 0.067 for E1).

`subsets_1_k1` (1%, another set of 462 labeled images, one seed): the teacher lowers AbsRel by 11.1%
(0.202 → 0.180, final step the same), the same gain as on the main subset (11.2%).

`subsets_1_k2` (1%, a third set of labeled images, one seed): 0.199 → 0.177 (−10.9%), final step
0.201 → 0.180. Over the three labeled subsets the gain is 11.2%, 11.1% and 10.9%, so it does not
depend on which images are labeled.

`ablation_1_5` (one seed; compare with E1/E4 from `fix_1_short` and `main_5_short`): at 1%, using
the teacher only on the labeled images (E2) gives 0.195 (final 0.199), the same as E1 (0.199). Using
it on the whole pool (E3) gives 0.177 (final 0.180), the same as E4 (0.176). So the gain comes from
the teacher's targets on unlabeled images, and the edge loss adds nothing measurable. At 5% every
teacher variant is worse than E1 (0.166): E2 0.174, E3 0.172, E4 0.170.

`students_1_tiny` (1%, ~1M-parameter student, one seed): 0.224 → 0.203 (−9.3%), final step
0.225 → 0.207 (−8%). Both recipes are ~0.025 worse than with the small student, but the teacher's
gain is about the same. The tiny student with the teacher is close to the small student without it.

Runs up to `fewlabel_v2_base` were made before the project used git, so their reports have no commit hash.
`timing_test` records commit `0e01ce5`, which became `393908b` when the history was rewritten (same code).

`long_e1_e3` (one seed, best checkpoint on validation): the teacher lowers AbsRel by ~11% at 0.5% and 1%
labels, but at 5% the two recipes are the same within noise. With few labels E1 memorizes its images
(training SILog falls to ~0.04 while validation stops improving); E3 does not. E3's RMSE is not better
at 1% and 5%. The session disconnected once during E3 at 5% and resumed from the step-17,000 checkpoint.
