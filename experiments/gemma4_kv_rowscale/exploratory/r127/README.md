# R-127 scripts (EXPLORATORY)

The post-run reviewer of E-74 (R-127, reviewer, 2026-11-18) wrote these scripts. They were copied here from the session
scratchpad. Their numbers are exploratory: no rule was frozen for them before they ran.

- `q1.py`: recomputes the E-74 decision values from `results/e74-mac-20261118/metrics_e54.json`.
- `q2.py`: checks the V8 failure of run 1 (layer 29) and the A2 floor.
- `q3.py`, `q3.json`: 5 cells of the E-54 captures (PTH only): the error at scale offsets j = 0, 3 and 6, with the row
  maximum or the top 4 elements kept exact, and with the BOS row exact. This tests the hypothesis that the vLLM rule
  stores the row maximum exactly. The review is `dossier/REVIEW_E-74_POSTRUN_2026-11-18.md`.

The scripts can hold absolute paths of a temporary folder; they are kept as a record, not as reusable tools.
