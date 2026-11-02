# Workflow

How work is done in this repository, so that no result gets lost and every number can be traced.

## One experiment = one issue

1. Open an issue with the **Experiment** template: hypothesis, setup, command.
2. If the experiment needs code changes, make a branch named after the issue:
   `exp/12-metric-pseudo-labels`. Other prefixes: `fix/`, `docs/`, `chore/`.
3. Commit the code **before** running. Each `report.json` records the commit
   (`"run": {"git_commit": ..., "git_dirty": ...}`); `git_dirty: true` means the run used
   uncommitted code and is not reproducible.
4. Write results into `results/<run_name>/` (use `--out`), and save the console output:
   `python scripts/fewlabel_v2.py ... 2>&1 | tee results/<run_name>/run.log`
5. Open a pull request that says `Closes #12`, with the result table filled in.
   CI must pass (lint + smoke test).
6. Merge into `main`. Closed issues and merged PRs together form the experiment log for the report.

Small changes (typos, proposal edits, adding a log) can go straight to `main`.

## Milestones

Tag the state of the repository at each deliverable, and make a GitHub Release for it:

| Tag | Deliverable |
|---|---|
| `proposal` | Proposal (Oct 30, 2026) |
| `final` | Final report and code (Dec 27, 2026) |

## Rules

- Never commit data, caches or checkpoints (`.gitignore` covers `data/`, `cache/`, `*.npz`, `*.pt`).
- Tune only on the validation rooms. The 654 test images are used once, for the final tables.
- Every number in the report must come from a committed `report.json`.

## Checks

```bash
make lint    # ruff
make smoke   # every script on tiny synthetic data, CPU, ~2 min
```
