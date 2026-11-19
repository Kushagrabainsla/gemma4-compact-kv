# E-73: does the engine follow the model to a second operating point (15 GiB pool, 8K prompts)? (preregistration)

Status: RUN 2026-11-17 (D-127, the D-118 rule waived). Q1 STEP-CONFIRMS, Q2 SHIFT-MATCHES (R-124 ACCEPT_WITH_CHANGES).
The result is in the Section "Result" at the end. (Preregistration text below: it waited for
the monthly Modal credit renewal under the D-118 rule, until we waived the rule in A2.)
Idea: I-182 (v2). Origin: E-71 and E-72 (one point, 15 GiB D3H: R_step 1.376, repeated to 0.06%, clean transfers);
the E-73 sizing and tier check (Section 1 of the journal). We, 2026-11-17: "I think we should do the item 1 ...
this will our v2", "start the free sizing now", "go ahead!", "Yes, write E-73". Labels: FACT, INTERPRETATION,
HYPOTHESIS. Thresholds are project heuristics.

## 1. Questions
- Q1 (prediction): at a second operating point, 15 GiB D pool and D3Q prompts (median 8,192 tokens), is the engine's
  R_step (compact over stock decode tokens per work step) inside the frozen model band?
- Q2 (shift): from D3H (E-71 and E-72) to D3Q, does the engine's R_step fall by the amount that the model predicts?
- Q3 (bias, report): is the compact excess over the model (+0.94% at D3H) the same at D3Q?

## 2. Rubber duck
- FACT (sizing, `results_e73_sizing.json`): of 9 candidate points, 15 GiB D3Q has a prediction clearly below D3H
  (r_step about 1.22 against 1.36), a tight seed range, 8 to 10 running requests and 10 to 14 preemptions per window,
  and a prefill need of about 570 tokens/s (one P gives about 1,800). Larger pools are near or above the P limit.
- FACT (tier check, `e73_tier_check.py`): the D3Q prediction is the same with a 200 GB, 64 GiB or 10 GB CPU tier;
  seed 0 has no missed reload in the model.
- FACT (`prediction_interval_e73.json`, `e73_interval.py`): r_step interval [1.16940, 1.22858] over seeds 0 to 4 and
  depths 2, 3, 8; seed 0 depth 3: 1.22853 (at the top again). Report only: Q_graph [1.053, 1.170], Q_eager
  [1.018, 1.129]; seed 0: 1.170 and 1.129. The model predicts 8.25 stock and 10.13 compact running requests (seed 0).
- FACT (E-71, E-72): the engine follows seed 0 closely: stock 5.458 against the model's 5.454; compact 7.509 and 7.514
  against 7.438 (+0.94%). R_step at D3H: 1.37571 (E-71), 1.37657 (E-72 clean).
- INTERPRETATION: if the compact excess of +0.94% is a constant bias of the model, the engine gives about
  1.22853 x 1.0094 = 1.2401 at D3Q, which is 0.0008 below the band top. So the Q1 word may again pass or fail by a very
  small margin. Q2 tests the shift between the two points, which a constant bias does not change. Q2 is the
  question that a reviewer cares about most: "does the model predict how the gain changes?"
- What would make E-73 useless: P cannot keep D saturated (W4); the D3Q sequence in the engine differs from the
  model's (a Mac test checks it); a lease expires (W3).

## 3. Hypotheses
- H1: R_step is inside [1.15770, 1.24087].
- H2: the engine shift is 0.75 to 1.25 times the model shift.
- H0: R_step is below the band (the model overstates the gain at D3Q), or the engine shift is outside 0.75 to 1.25.

## 4. Setup (frozen)
- As E-72 C2 and E-71: one Modal container, two A100 80GB (P on GPU 0, D on GPU 1); vLLM 3966924d5, nixl 1.5.0;
  eager; BF16 KV; LBHNC; prefix caching off; `max_model_len` 144,000; D pool 15 GiB (6,144 blocks); P budget 1,024;
  D budget 2,048; D MultiConnector of NIXL and the offload connector (64 GiB, `offload_prompt_only` False,
  `prefix_cache_retention_interval` None); UCX error mode none; the byte-view back-port on compact engines;
  `kv_lease_duration` 600 on P and D (E-72).
- Workload: D3Q, seed 0: the E-64 generator with `random.Random("64-D3Q-0")`, prompt lognormal median 8,192, output
  lognormal median 2,048, sigma 0.6, the clips of E-71, the model's skip rule on the usable pool, and the length limit
  144,000. P keeps 2 prefilled requests ready (BUF 2); D keeps 3 requests waiting (WAIT 3); `ignore_eos`; greedy.
- Arms, in this order: S (stock cache, hooks OFF, as E-71 S1 and S2), then C (compact cache, VIEWS hooks, as E-71 C).
  No A/A arm: E-71 measured an A/A spread of 0.03% and E-72 a compact repeat of 0.06%.
- Window: work steps [2,000, 22,000) of each arm, as E-71.

## 5. Metrics
- R_step = (C decode tokens per work step) / (S decode tokens per work step).
- Engine shift = 1.37614 - R_step, where 1.37614 is the mean of the two D3H engine values (E-71, E-72 clean).
  Model shift = 1.36380 - 1.22853 = 0.13527 (seed 0, depth 3, at D3H and D3Q).
- Excess = R_step / 1.22853 - 1 (report; compare with +0.94% at D3H).
- Reported: running requests per step (against 8.25 and 10.13), saturation, preemptions, completed, q depth, step
  time, the throughput ratio against Q_graph and Q_eager, the P lease lines.

## 6. Validity checks
- W1: all roles on A100 80GB. W2: the code hashes match (`e73_code_manifest.json`).
- W3: no lease expiry: 0 "Releasing expired KV blocks" and 0 "Potentially invalid KV blocks" lines in the P logs,
  and the line `'kv_lease_duration': 600` present in both P logs.
- W4: D saturated in at least 95% of the window steps in both arms.
- W5: both arms complete the window, with 0 early finishes and at least 1 completed request.
- W6: the D pool is 6,144 blocks in both arms.
- W7: C has the compact spec and LBHNC on P and D; S has the stock spec; the hooks have no problems in every role.

## 7. Decision rule (frozen)
- If a check fails: INVALID (the failed checks are named).
- Q1: STEP-CONFIRMS if R_step is inside [1.15770, 1.24087] (the interval widened by 1% on each side, the E-71 floor;
  there is no A/A arm); STEP-ABOVE above it; STEP-BELOW below it.
- Q2: s = engine shift / model shift. SHIFT-MATCHES if 0.75 <= s <= 1.25; SHIFT-SMALLER if s < 0.75 (the engine gain
  falls less than the model predicts); SHIFT-LARGER if s > 1.25.
- Q3: report the excess; no word.
- The words are reported side by side. The v2 wording uses Q2 as the main statement of model validity across the
  two points, and Q1 as the point check.

## 8. Budget (Modal prices of 2026-11-12)
- Two A100 80GB, 8 cores, 160 GiB: $6.65/h. Predicted arm time (engine-calibrated): S about 1,520 s, C about 1,650 s,
  plus two engine builds per arm (about 250 s) and the snapshot: about 3,700 s, about $6.9.
- Limits: arm 2,650 s each (D watchdog 2,250 s after the build); container 5,700 s, so both arms fit even at their
  limits. Cost cap $11.09 for one try (expected about $6.9).
  If S fails, C does not run.
- Credits: about $10.56 left (2026-11-17), below 2 x $11.09; the run waits for the monthly renewal ($30), when the D-118
  rule holds. A live credit read comes before the request.

## 9. Threats
1. Q2 uses the D3H engine values from E-71 and E-72 (other sessions and hosts; E-72 repeated E-71 to 0.06%).
2. One seed per point; two points on one axis (prompt length) at one pool; eager; one model.
3. Seed 0 is at the top of the model interval at both points; a constant model bias moves Q1, not Q2.
4. The D3Q window has more preemptions; the offload reload is then a larger part of the result (E-69 and E-70 showed
   that it works).

## 10. Files
`README_E73.md`, `e73_interval.py`, `prediction_interval_e73.json`, `run_e73.py` (routes the E-71 driver: the D3Q
sequence, the lease, the arms S and C), `stub_e73.py`, `analyze_e73.py`, `e73_stage.py`, `modal_e73.py`, `test_e73.py`,
`e73_code_manifest.json` (written by `e73_stage.py --manifest`).

## Amendments

### A1 (2026-11-17, before any run; the pre-run review R-123, `dossier/REVIEW_E-73_PRERUN_2026-11-17.md`)
R-123 gave RUN with no blocker and three optional fixes. All three are applied; the decision rule does not change.
- **F1. The hook mode is recorded and checked.** `run_e73.recording_build` wraps the E-71 build and records the mode
  that the build sets (`rec["mode"]`). W7 now also needs `mode` OFF on both S roles and VIEWS on both C roles. A
  test covers the wrapper and a wrong S mode.
- **F2. Preemptions (Section 2).** For seed 0 at depth 3 the model gives 8 (stock) and 9 (compact) preemptions in the
  window; "10 to 14" was the median over seeds 0 to 4. At D3H the engine had 6 against the model's 3, so about 16 to
  20 are possible here. This is a report, not a check.
- **F3. When Q1 and Q2 disagree (Section 7).** Q2 is the main statement. If R_step is in [1.15770, 1.20705), the words
  are STEP-CONFIRMS and SHIFT-LARGER: the engine is inside the seed range but falls more than the model predicts.
  If R_step is in (1.24087, 1.27469], the words are STEP-ABOVE and SHIFT-MATCHES: the model predicts the change
  correctly, with the same small constant excess as at D3H. The result reports both words and these sentences.

### A2 (2026-11-17, before any run; our waiver of the D-118 rule)
We, 2026-11-17: "lets bypass the 2x credit rule, and go ahead with whatever credit is left, so that atleast we
get our v2 with this results added". A live credit read showed $19.55 of the $30 monthly credit used, so about
$10.45 left, below the A1 cap of $11.09. So the cap must stay below the credits left:
- container limit 5,000 s (was 5,700 s), so the cost cap is $9.80, about $0.65 below the credits left;
- arm limit 2,350 s (was 2,650 s) and D watchdog 2,050 s (was 2,250 s; `run_e73.WATCHDOG_S`), so both arms fit at
  their limits (2 x 2,350 + 120 = 4,820 s against 4,850 s); the watchdog allows 93 ms per work step for 22,000 steps,
  against the predicted 69 to 75 ms.
The decision rule does not change. The expected cost stays about $6.9. 5 tests pass with the new limits.

## Result (2026-11-17; post-run review R-124 ACCEPT_WITH_CHANGES, `dossier/REVIEW_E-73_POSTRUN_2026-11-17.md`)
Run `results/e73-20261117T020634Z` (D-127; two A100-SXM4-80GB, Modal Azure eu-south, 3,469 s: S 1,677 s, C 1,784 s;
about $6.51; cap $9.80). W1 to W7 pass.

| Arm | Tokens per work step | Model (seed 0) | Engine vs model | Tokens/s | Step | Saturated | Preemptions / completed |
|---|---|---|---|---|---|---|---|
| S stock | 8.1922 | 8.246 | -0.62% | 120.7 | 67.9 ms | 99.99% | 12 / 62 |
| C compact | 10.1326 | 10.130 | +0.02% | 134.0 | 75.6 ms | 100% | 16 / 79 |

- **Q1 STEP-CONFIRMS.** R_step 1.23686, inside [1.15770, 1.24087] (margin 0.004 to the top). The pass rests on the
  1% floor again: R_step is 0.68% above the model's 5-seed top (1.22859).
- **Q2 SHIFT-MATCHES (the main statement).** The engine drop from D3H (1.37614) is 0.13928; the model drop is
  0.13527; the ratio is 1.030. The ratio stays within 0.75 to 1.25 for any D3H engine value from 1.338 to 1.406 (E-71
  or E-72 alone gives 1.026 to 1.033). The test confirms the direction and the size of the drop, not 1% accuracy.
- **Q3 (report).** The excess over the model is +0.68% (D3H +0.94%). The per-arm errors changed between the points:
  at D3H, stock matched the model and compact was +0.95%; at D3Q, compact matched and stock was -0.62%. So the
  per-layout bias is not constant; the layout difference (compact about 0.7 to 0.9% above the model relative to
  stock) looks stable. HYPOTHESIS: the engine preempts more than the model at both points (12 and 16 against 8 and 9
  here); not tested.
- **Clean run.** 0 lease lines and the lease line present in both P logs; D: 0 failed transfers, 0 expired; 0 early
  finishes; queue depth 1.7 and 1.6 of 2; P 553 and 610 tokens/s; reloads 13 and 18 with 96% and 99.7% hits.
- **Throughput (report).** R_tput 1.110, inside Q_graph [1.053, 1.170] and Q_eager [1.018, 1.129]; not resolved (A/A
  2.7% in E-71; S ran first).

**Allowed wording (R-124, for E-71 to E-73):** "On two A100 80GB at a 15 GiB pool and one seed per point, the compact
cache gave 1.376 times the stock decode tokens per work step at D3H. A second compact run repeated this to 0.06%. At a
second point (D3Q, median 8,192-token prompts) it gave 1.237, against 1.228 for the model's seed 0. The drop between
the points, 0.139, was 1.03 times the model's drop of 0.135. The ratio stays within 0.75 to 1.25 for any D3H value from
1.338 to 1.406. Compact runs about 0.7 to 0.9% more than the model predicts, and each per-arm error differs by point.
Without the 1% floor, R_step is above the model's seed range at both points."

**Not claimable:** exact prediction or a constant bias; other seeds, pools, CUDA graphs or hosts; a throughput gain
finer than about 3%; more than two points on one axis.
