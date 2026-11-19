# E-71: does the compact K=V cache give the predicted decode throughput on a real 31B decode node? (preregistration)

Status: RUN 2026-11-16 (D-125). Word STEP-CONFIRMS (by 0.002, on the 1% band floor; R-120 ACCEPT_WITH_CHANGES).
Preregistered 2026-11-14 and amended by A1 (after R-118 and R-119), A2 and A3 before any 31B data. The result is in
the Section "Result" at the end. Sections 4 to 8 hold the first design; A1 replaces the parts it names.
Idea: I-182 (v2). Origin: E-64 and E-67 (CPU models), E-68 (measured read times, exploratory), E-69 (CPU reload works
on one engine), E-70 stages 1 and 2 (NIXL transfer and the reload on a NIXL-fed decode engine work, on the 12B). The
user, 2026-11-14: "we'll do the 31B test".
Labels: FACT, INTERPRETATION, HYPOTHESIS. Thresholds are project heuristics.

## 1. Question
On a real prefill/decode pair of Gemma 4 31B engines (vLLM 3966924d5, NIXL with the byte-view back-port, the offload
connector on D), does the compact cache raise D's decode throughput over the stock cache by the ratio that the E-67
model (with the E-68 read table) predicts for the same request sequence and pool? "No": the measured ratio is below
the prediction band, or below 1.

## 2. Rubber duck
- FACT (sizing, `e71_predict.py`, `results_predict_sizing.json`, exploratory): at the D pool of a 31B engine on one
  80 GB A100 (15 GiB = 6,144 blocks), with offload of all blocks, the model predicts compact/stock decode throughput
  1.31 [1.24, 1.43] for D3 (32K-token prompts) and 1.29 [1.25, 1.34] for D3H (16K-token prompts; median and range over
  5 seeds). One prefill engine must deliver about 1,300 (D3) or 1,030 (D3H) prompt tokens per second; the E-64 time
  model puts 31B prefill at about 2,000 tokens per second on an A100. So D3H leaves a 2x margin; D3 does not.
- At this pool the model predicts few preemptions (0.3 to 0.4 per 1,000 steps for D3H), so the measured gain is mostly
  the capacity effect (more running requests per step), with the offload connector as the recovery path.
- What would kill it early: P cannot keep D saturated (then D measures P); the 31B compact engine with NIXL fails
  (the 12B worked); the NIXL transfer of 31B requests is slow (stage 1: loads done within about 4 s for 8 requests).

## 3. Hypotheses (H1 replaced by A1: R_step inside the band)
- H1: the measured C/S decode-throughput ratio lies inside the prediction band, and it is at least 1.25.
- H0: the measured ratio is below the band (the model overstates the engine), or below 1.

## 4. Setup (frozen)
- Hardware: one Modal container with two NVIDIA A100 80GB (P on GPU 0, D on GPU 1); the E-70 harness (two in-process
  engines in two processes, NIXL with `ucx_error_handling_mode` none, the byte-view back-port on compact engines).
- Model: `google/gemma-4-31B-it` rev `842da379...` (in `msthesis-e69-weights`). Engines: the E-57/E-69 settings
  (TRITON_ATTN, V2 runner, eager, BF16 KV, prefix caching off, LBHNC), `kv_cache_memory_bytes` 15 GiB on D (6,144
  blocks) and on P, `max_model_len` 131,072 + 16,384 on D (the E-64 clip), `max_num_seqs` 16 on D.
- D connector: MultiConnector of NIXL and the offload connector (CPU tier 64 GiB, `offload_prompt_only` False),
  `prefix_cache_retention_interval` None (E-70 stage 2). P connector: NIXL.
- Workload: D3H, the E-64 generator (`random.Random("64-D3H-0")`; prompt lognormal median 16,384, sigma 0.6, clipped
  [256, 131,072]; output lognormal median 2,048, sigma 0.6, clipped [16, 16,384]); `ignore_eos`; greedy. Prompt token
  ids: the E-57 synthetic generator (ids in [1,000, 100,000)), distinct per request. A request whose stock full-ISL
  reservation does not fit the empty pool is skipped (the E-64 rule).
- Arrival: saturated. A driver keeps P prefilling the next requests in order and keeps 2 prefilled requests waiting on
  D at all times (the E-64 backlog is never empty).
- Arms (same container, in this order): S (stock), C (compact), S again (A/B/A, for drift and the A/A noise floor).
- Measurement on D only: decode tokens per second over a window of D engine steps [2,000, 22,000) of each arm (after
  a warm-up of 2,000 steps), from D's step outputs.
- Window choice (FACT, `e71_window_scan.py`, `results_window_scan.json`, exploratory): the predicted D3H ratio over 5
  seeds is 1.27 [1.19, 1.30] for a 10,000-step window, 1.31 [1.27, 1.32] for 20,000, 1.28 [1.25, 1.29] for 30,000 and
  1.28 [1.18, 1.29] for 40,000. A short window holds few requests, so its ratio depends on which requests are in
  flight; 20,000 steps is tight and near the long-run value at an affordable time.
- Prediction (frozen before the run, `prediction_frozen.json`, from `e71_predict.py --window`): the E-67 model, all200,
  L20, tag `meas`, pool 6,144, D3H seed 0, warm-up 2,000 steps, horizon 22,000 steps: **Q_pred 1.324, band [1.274,
  1.374]** (heuristic: plus or minus 0.05, about the seed spread at this window). The model's running decodes per
  step: stock 5.46, compact 7.44; preemptions 3 and 3.

## 5. Metrics (the primary metric replaced by A1: R_step)
- Primary: R = (decode tokens/s of C) / (mean decode tokens/s of the two S runs), over the window.
- Noise floor: the A/A spread |S1 - S2| / mean(S); R is resolved only if the A/A spread is at most 3%.
- Secondary: D saturation share (steps with a waiting request on D), running requests per step, preemptions and
  offload reloads on D, NIXL load time per request, P prefill tokens per second, the step time.

## 6. Validity checks (replaced by A1)
- V1: both GPUs are A100 80GB. V2: the code hashes match. V3: D is saturated in at least 95% of the window steps in
  every arm. V4: the A/A spread is at most 3%. V5: the compact spec and LBHNC on both compact engines. V6: no request
  fails a NIXL load, and every arm completes the window.

## 7. Decision rule (frozen; replaced by A1 before any run)
- ENGINE-CONFIRMS: V1 to V6 pass; R inside the band and R at least 1.25.
- ENGINE-CONFIRMS-LOW: R inside the band and below 1.25 (the model is right, the gain is below the gate).
- ENGINE-BELOW: R below the band (the model overstates the engine gain).
- ENGINE-ABOVE: R above the band.
- INVALID: a V check fails (V3 failing means P limits D: report it, then a D3Q follow-up needs a new approval).

## 8. Budget (Modal prices of 2026-11-12; replaced by A1)
- Two A100 80GB $5.00/h + 8 cores $0.38/h + 160 GiB $1.28/h = $6.66/h. Per arm: two 31B engine starts (about 4 min,
  in parallel), the warm-up and window (22,000 D steps at about 60 to 70 ms: 22 to 26 min). Three arms and the
  container: about 90 min, about $10 per try. Time cap 110 min per try ($12.50 with the startup).
- A 12B rehearsal of the streaming driver on two L40S first (about 20 min, cap $2.50).
- Cap: $15 of Modal credits for one rehearsal try and one main try (about $25 of credits left on 2026-11-14); a
  second main try needs a new approval.

## 9. Threats
1. One seed, one pool, one workload (D3H, not D3): the claim is a model validation at one point.
2. Synthetic prompt ids and `ignore_eos` (no output quality claim).
3. Eager mode; one Modal host (gVisor, transport unknown); A100 type as Modal gives it.
4. The NIXL back-port at the pin (main fixed the check).
5. The model's "steps" and the engine's steps differ in what a step holds (chunked prefill of re-admissions on D);
   the window is matched in steps, not seconds.

## 10. Files (to write)
`e71_predict.py` (done; `--window` adds the frozen prediction), `run_e71.py` (P and D streaming roles), `e71_stage.py`,
`modal_e71.py`, `analyze_e71.py`, `test_e71.py`, `make_manifest.py`, `code_manifest.json`.

## Amendments

### A1 (2026-11-14, before any run; the pre-run review R-118, `dossier/REVIEW_E-71_PRERUN_2026-11-14.md`)
R-118 gave AMEND_THEN_RUN with three blockers. No data existed when A1 was written. A1 replaces the parts it names.
A different-model re-check of A1 (R-119, AMEND_THEN_RUN: the smoke gate, the stall timer, two checks that cannot fail,
the start margin, the band floor) came before any run; its fixes are part of A1 and are marked "R-119".

**B1. Engine start.** The pinned vLLM refuses to start if one `max_model_len` request does not fit the pool. Under async
scheduling each sliding group needs `cdiv(1023 + 2 x budget, 16) + 1` blocks. With 147,456 tokens, stock D and every P
failed. A1 sets `max_model_len` 144,000 on both roles and P `max_num_batched_tokens` 1,024. The largest value that stock
D at budget 2,048 accepts is 145,216; 144,000 keeps a margin of 38 blocks (R-119 F4). `requests()` skips a request with prompt plus output above `max_model_len`, and its
skip probe uses the usable pool, 6,143 blocks. The seed-0 sequence does not change for the first 2,000 requests
(`scratchpad/e71check/seq_a1.py`). The test checks the start arithmetic for both roles, both layouts and both models.

**B2 and B3. The primary statistic and the frozen prediction.**
- The frozen prediction was a CUDA-graph model at one seed. The engine is eager, and one trajectory is chaotic.
- The new primary statistic is time-free: **R_step** = (decode tokens per work step of C) / (the mean of S1 and S2).
- The frozen prediction is `prediction_interval.json` (`e71_interval.py`): the E-67 model (all200, L20, `meas`), pool
  6,143, window [2,000, 22,000), D3H seeds 0 to 4, backlog depths 2, 3 and 8, length limit 144,000 (the values at
  145,216 were the same to every printed digit).
- **r_step interval [1.308, 1.364]** (seed 0, depth 3: 1.364). The depth changes r_step by less than 0.0002, so the
  interval is the seed spread (R-119 F5). The engine runs seed 0, which is at the top of the interval. So
  STEP-CONFIRMS allows the model to overstate the engine by up to about 4 to 5%. The result must say this.
- Report only: Q_graph [1.272, 1.325] and Q_eager [1.225, 1.276]. Q_eager adds the E-59 eager costs to the model step:
  +1.5 ms for both layouts and a further +2.2 ms for compact.
- N1: the model tier is 200 GB, the engine tier 64 GiB. R-118 showed that Q does not change with a 68.7 GB tier.

**Decision rule (replaces Section 7).**
- The band is the r_step interval, widened on each side by half the A/A spread of the step statistic, with a floor
  of 1% (`h = max(aa/2, 0.01)`; band `lo x (1 - h)`, `hi x (1 + h)`; R-119 F6: two runs can agree by chance).
- STEP-CONFIRMS: all checks pass and R_step is inside the band. The model predicts the engine's capacity gain.
- STEP-BELOW: R_step is below the band. The model overstates the capacity gain in the engine.
- STEP-ABOVE: R_step is above the band.
- INVALID: a check fails.
- Secondary (a report, no word): the throughput ratio R against Q_graph and Q_eager ("below", "inside", "above"), the
  throughput A/A spread, and whether R is at least 1.25. The 1.25 gate of Section 7 is now a report line only.

**Checks (replace Section 6).** V1 the GPU names (A100 and 80GB; the product name goes in the result). V2 the code hashes.
V3 D saturated in at least 95% of the window steps in S1, C and S2. V4 the A/A spread of tokens per step at most 3%.
V5 the compact spec and LBHNC on C0 and C. V6 every arm completes the window (a sanity check; it cannot fail when D
ends ok, R-119 F3). V7 in C0, S1, C and S2 no request ends with a finish reason other than "length", and at least 1
request completes in the window. V8 the D pool is 6,144 blocks in S1, C and S2 (main only; the bytes per block are not
compared). The smoke arm C0 must end ok in both roles.

**Driver changes (R-118 F1 to F5, N3, N7).**
- F1: a short 31B compact smoke arm C0 runs first (warm-up 50, window 300 work steps, limit 900 s). It checks the start,
  the NIXL load of a 31B compact request and one window before the paid A/B/A arms. R-119 B1: a failed NIXL load ends
  a request with finish reason "error" and the arm still ended ok. Now D stops at the first request that ends without
  its length, and the session runs S1 only if C0 has 0 early finishes and more than 0 decode tokens.
- F2: D stops with "P starved" if P fails, or if, after the first request, nothing waits on D and no prefilled request
  is ready for 300 s (R-119 F2: the timer starts when D is starved, not at the last add). Each D watchdog
  (counted after the engine build) is shorter than its arm limit: main 2,000 s in 2,400 s, C0 400 s in 900 s.
  D writes `progress.json` every 500 work steps.
- F3: D counts early finishes and completed requests (V7). The D pool is recorded (V8).
- F4: D counts the tokens of every step in the window, idle steps included, and starts the clock before the first
  window step.
- F5: D logs the depth of the prefilled queue. P keeps at most 2 prefilled requests that D has not taken. Not done:
  the P "blocked share" and the F6 early abort on a slow step rate (the 2,000 s watchdog, 91 ms per step, bounds the
  loss to one arm).
- B3 and N7: D keeps 3 requests in the not-allocated waiting states (WAITING, PREEMPTED), the model's `keep_waiting` 3.
- N3: `detokenize=False` on D.

**Rehearsal (R-118 F8, mandatory).** The 12B rehearsal on two L40S runs the same four arms with a short plan (6 GiB pool,
`max_model_len` 32,768, warm-up 300, window 1,500; C0 warm-up 50, window 300). It passes if all arms end ok, the checks
V2 and V5 to V7 pass, `q_depth_mean` is logged and an R_step is printed. The rehearsal word has no meaning for the
claim. If the rehearsal fails, the main try does not start.

**Budget (replaces Section 8; Modal prices of 2026-11-12).** Main: two A100 80GB, 160 GiB, 8 cores, $6.65/h. The
container time cap is 120 min plus a 5 min start, so the cost cap is $13.86. Expected time: 4 builds and 3 windows of
22,000 steps at 60 to 85 ms, about 80 to 106 min. If the engine is slower, the session deadline skips S2 and the word is
INVALID. Rehearsal: two L40S, cap $2.53. **Total cap $17** of Modal credits for one rehearsal and one main try. A
second main try needs a new approval.
Credits (live read, `modal billing summary`, 2026-11-14): $4.45 used this month of the $30 monthly credit, so about
$25.5 left. The D-118 rule (credits at least 2 x the stage's worst case) holds for the rehearsal ($5.06) but not for the
main try ($27.72). The main try needs our waiver of that rule; the credits cover its worst case 1.84 times.

### A2 (2026-11-16, before any data; an operational change)
D-125 was decided by us on 2026-11-16 ("Approve, cap $17", after a live credit read of about $25.7). The first
rehearsal waited 25 minutes in the Modal queue for two L40S and never started (no cost). We: "Move to A100". The
rehearsal now runs on two A100 80GB, the GPU type of the main try. The plan of the rehearsal does not change (12B,
6 GiB pool, `max_model_len` 32,768). V1 in the rehearsal now asks for A100 80GB. The rehearsal cap is $3.07, and the
total is $3.07 + $13.86 = $16.93, inside the $17 cap. The decision rule of the main try does not change.

### A3 (2026-11-16, after the rehearsal, before any 31B data)
Rehearsal `results/e71-reh-20261116T200322Z` (two A100-SXM4-80GB, 12B, about $1.74, 1,128 s with the queue): all four
arms ended ok in both roles; V1, V2, V4, V5, V6, V8 pass; V3 and V7 fail; R_step 1.12 (no meaning for the claim).
- **V7 could never pass.** C0 runs 350 work steps, and a request with the median output of 2,048 tokens needs about
  2,000. So C0 completed 0 requests. A3: the "at least 1 completed request" part of V7 applies to S1, C and S2 only;
  "no early finish" still applies to all four arms. With A3 the rehearsal gives V7 true (a re-analysis of a copy).
- **V3 failed in the 12B (S1 0.965, C 0.885, S2 0.92).** The 12B pool often has room for every waiting request. V3 is
  not part of the rehearsal pass rule, and the model gives the 31B pool `pool_bound` 1.0. No change.
- **Time.** The 12B steady step was 58 ms per work step (C0: 68.8 ms, slower in the ramp). The 31B is likely slower,
  and the A1 watchdog (2,000 s for 22,000 steps) allows 91 ms. A3 adds a speed gate after C0: the C0 time per work step
  times 22,000 must be at most the D watchdog less 100 s, and three such arms plus three builds (250 s each) must fit
  the time left. Else the session stops after C0 (about $1.5). A3 also sets the D watchdog to 2,250 s (gate limit about
  98 ms per step), the arm limit to 2,650 s and the container limit to 130 min. The main cost cap is $14.97.
- The rehearsal pass rule holds with A3: all arms ok, V2 and V5 to V7 pass, `q_depth_mean` logged (about 1.3 of a
  buffer of 2, so P kept up), and an R_step printed.
- Budget: rehearsal $1.74 (credit read) plus the main cap $14.97 is $16.71, inside the $17 cap. Credits before the main
  try: about $23.9 (live read). The D-118 rule is waived for the main try (D-125).
- The decision rule of the main try (R_step against the r_step band) does not change.

## Result (2026-11-16; post-run review R-120 ACCEPT_WITH_CHANGES, `dossier/REVIEW_E-71_POSTRUN_2026-11-16.md`)
Run `results/e71-main-20261116T202539Z` (two A100-SXM4-80GB, Modal Azure eu-south, 5,370 s, about $10.07). The C0
smoke arm and the speed gate passed (C0 step 79.6 ms; projected arm 1,751 s against a 2,250 s watchdog).

| Arm | Decode tokens per work step | Model (seed 0) | Tokens/s | Step (ms) | Saturated | Preemptions | Completed |
|---|---|---|---|---|---|---|---|
| S1 stock | 5.4576 | 5.46 | 78.7 | 69.4 | 99.65% | 6 | 41 |
| C compact | 7.5092 | 7.44 | 101.3 | 74.2 | 99.99% | 6 | 63 |
| S2 stock | 5.4593 | 5.46 | 76.6 | 71.3 | 99.73% | 6 | 41 |

- **Word: STEP-CONFIRMS.** R_step 1.3757. Band [1.2952, 1.3775] (the frozen interval [1.3083, 1.3638] widened by
  h = max(A/A/2, 0.01) = 0.01). V1 to V8 pass. The rule did not change after the run started (R-120).
- **The word rests on the 1% floor and passed by 0.002.** Without the floor (h = A/A/2 = 0.016%), the word would be
  STEP-ABOVE. The measured value is 0.87% above the top of the model's 5-seed range and above the seed-0 value (1.364).
- The stock arms reproduce to 0.03% with the same preemption count, so run noise of stock is small. There is no
  compact repeat, so the compact noise is unmeasured.
- The whole excess comes from compact: it ran 0.9% more requests per step than the model (7.51 against 7.44); stock
  matched the model to 0.03%. The engine had 6 window preemptions per arm, the model 3 for seed 0 (3 to 8 over seeds
  1 to 4). INTERPRETATION: consistent with trajectory differences; not proven.
- Report only: the throughput ratio R 1.304, inside Q_graph [1.272, 1.325] and above Q_eager [1.225, 1.276] by 0.028.
  The throughput A/A spread is 2.7% (R 1.287 to 1.322 by which stock arm is used), so "above Q_eager" is not resolved.
  Compact steps were 5.5% slower than stock in eager mode.
- P kept D fed: q depth mean 1.26 to 1.36 of 2; saturation above 99.6% in every arm. The CPU tier reloads hit 99.8%
  to 99.9% of preempted lookups and took about 1 s per arm.
- R-120 F8, traced on 2026-11-16 (pinned source `nixl/pull_worker.py:665-680`, `base_worker.py:3384-3430`,
  `base_scheduler.py:86-88, 202-241`; script `e71_lease_check.py` with `results/e71-main-20261116T202539Z/lease_check/`; q-file times from the Modal
  volume):
  - FACT: P keeps the blocks of a prefilled request for a lease of `kv_lease_duration`, default 30 s, from the end of
    the prefill. D extends the lease with heartbeats (every 5 s, by 20 s), but only for requests that D has added to
    its own scheduler (`on_new_request`).
  - FACT: in every arm, each expired lease was released 30 to 32 s after P wrote the q file, so no lease was ever
    extended. P released S1 21, C 28 and S2 24 leases. D read 19, 28 and 22 of them 15 to 247 s after the release
    (median 118, 75, 122 s); two per stock arm were never read (after the window). So D read 36%, 35% and 42% of
    the requests it added after P had freed their blocks. The message "Potentially invalid KV blocks ... may have
    expired" is P's notice of such a late read.
  - INTERPRETATION: the harness keeps up to 2 prefilled requests in its file queue, outside D's scheduler, and D
    adds one only when its own queue of 3 has room (about one admission every 25 to 35 s). So a prefilled request
    can wait more than 30 s before D can heartbeat it. Not verified: whether heartbeats work for requests that are
    already in D's queue.
  - Effect: after the release, P can give those blocks to its next prefill, so D may have loaded another request's
    KV for these requests. The primary statistic does not depend on the KV values: the output length is fixed
    (`ignore_eos`, `max_tokens`), prefix caching is off, and the attention cost does not depend on the values. Both
    layouts had the same effect (35% to 42%). So the word and the throughput report stand. E-71 shows nothing about
    NIXL transfer correctness; E-70 stage 1 (O-32) covers that, at the 12B.
  - Fix for a future run: set `kv_lease_duration` on P (for example 600 s) in `kv_connector_extra_config`, or keep
    the prefilled requests inside D's scheduler.

**Allowed wording (R-120):** "On a real 31B prefill/decode pair at one pool (6,144 blocks) and one seed, the compact
cache gave 1.376 times the decode tokens per work step of stock. The pre-registered word is STEP-CONFIRMS, but it
passed by 0.002 and depends on a 1% band floor. The measured value is 0.87% above the top of the model's 5-seed range
and the model's seed-0 value. Compact ran 0.9% more requests per step than the model. Throughput gain was 1.30 in
eager mode (A/A 2.7%), inside the graph model interval and above the eager one. The model is consistent with the
engine's capacity gain to about 1%, not proven exact."

**Not claimable:** other seeds, pools, workloads (D3), CUDA graphs, other hosts or real prompts; output quality; NIXL
transfer correctness (35% to 42% of the requests were read after the P lease expired; R-120 F8 above); a throughput
claim finer than about 3%.

**Cost:** rehearsal about $1.74, main about $10.07; E-71 total about $11.81 of Modal credits (cap $17).
