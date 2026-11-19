# E-72: does the E-71 compact result repeat with clean transfers, and is the compact output exact on the P/D path? (preregistration)

Status: RUN 2026-11-17 (D-126). Q1 REPEATS, Q2 STEP-CONFIRMS-CLEAN, Q3 EXACT (R-122 ACCEPT_WITH_CHANGES). The result is
in the Section "Result" at the end.
Idea: I-182 (v2). Origin: E-71 (STEP-CONFIRMS by 0.002 on the 1% floor; R-120: the compact noise is unmeasured; R-120
F8 traced: 35% to 42% of the requests were read after P's 30 s NIXL lease had expired). We, 2026-11-16: "Go ahead
with the $5 run plan". Labels: FACT, INTERPRETATION, HYPOTHESIS. Thresholds are project heuristics.

## 1. Questions
- Q1 (repeat): with the same request sequence, does a second compact run (C2) give the same decode tokens per work step
  as E-71 C (7.50915)?
- Q2 (clean): with a NIXL lease long enough that no lease expires, is C2 over the E-71 stock mean still inside the
  frozen E-71 band?
- Q3 (exact): on the real P/D path (NIXL, the offload connector on D), does the compact engine give the same output
  tokens as its stock twin EMULV (the stock cache with the same view hooks), on 8 real needle prompts?

## 2. Rubber duck
- FACT (E-71): S1 and S2 gave 5.45755 and 5.45925 tokens per work step (A/A 0.03%), C 7.50915. R_step 1.3757 against
  the band [1.29524, 1.37748]. The step count does not depend on KV values (fixed output lengths, prefix caching off).
- FACT (R-120 F8 trace): P keeps a prefilled request's blocks for `kv_lease_duration` (default 30 s, pinned source
  `nixl/base_scheduler.py:86-88`); D heartbeats extend it only for requests in D's scheduler. In E-71 every expired
  lease was released 30 to 32 s after P wrote the q file; the harness holds up to 2 prefilled requests outside D.
  A lease of 600 s covers the longest observed q-to-read time (277 s) about 2 times.
- FACT (E-57): COMPACT equals EMULV bit for bit on one engine (432 of 432 items); it is not bit-identical to plain
  stock (the hooks change the order of operations). So the exactness reference is EMULV, not stock.
- FACT (R-118 F5): a compact P at budget 1,024 that holds 5 prefilled requests cannot admit the next request at 9 of
  100 positions of the seed-0 sequence. D keeps 3 requests waiting, so short P delays do not starve D. The EV arm has
  no workload.
- INTERPRETATION: D's schedule is driven by its pool when its waiting queue is never empty (E-71 C: saturated in
  99.99% of the window steps). So C2 should repeat C closely if saturation holds.
- What would make E-72 useless: a lease still expires (then Q2 and Q3 are not clean); P cannot keep D saturated with
  the long lease; the probe phase fails after the window (the window record is written first, so Q1 survives).

## 3. Hypotheses
- H1: |C2 / C - 1| is at most 0.5%.
- H2: C2 / 5.45840 is inside [1.29524, 1.37748].
- H3: the D output tokens of the 8 probes are identical between C2 and EV.

## 4. Setup (frozen)
- Hardware and software as E-71 main: one Modal container, two A100 80GB (P on GPU 0, D on GPU 1), vLLM 3966924d5,
  nixl 1.5.0, eager, BF16 KV, LBHNC, prefix caching off, `max_model_len` 144,000, D pool 15 GiB, P budget 1,024,
  D budget 2,048, D MultiConnector of NIXL and the offload connector (64 GiB, `offload_prompt_only` False,
  `prefix_cache_retention_interval` None), UCX error mode none, the byte-view back-port on compact engines.
- New: `kv_lease_duration` 600 in the NIXL `kv_connector_extra_config` of P and of D.
- Arms, in this order (one session):
  - EV: stock cache with the VIEWS hooks on P and D (EMULV); no workload; the probe only.
  - C2: compact cache with the VIEWS hooks on P and D (as E-71 C); the E-71 workload (D3H seed 0, WAIT 3, BUF 2,
    window work steps [2,000, 22,000)), then the probe.
- Probe (after the window in C2, at once in EV): D writes `probe_go`; P stops the workload, finishes what it has, then
  prefills the first 8 needle prompts of `gemma4_kv_tasks/data_e56.json` (Gemma 4 31B token ids, about 7,666 tokens)
  one at a time; D aborts all workload requests, waits until it has none, then decodes the probes one at a time
  (batch 1; greedy, seed 0, `max_tokens` 64, EOS allowed). Recorded per probe: the D output token ids, the text, the
  finish reason, and whether the gold passphrase is in the text. One at a time makes the batch the same in both arms.

## 5. Metrics
- Q1: d = C2 tokens per work step / 7.50915 - 1.
- Q2: R_clean = C2 tokens per work step / 5.45840 (the E-71 stock mean).
- Q3: the number of probes with identical D output token ids (of 8); the first differing position if any.
- Reported: saturation, preemptions, completed requests, q depth, step time, the P lease lines, gold answers.

## 6. Validity checks
- W1: all roles on A100 80GB. W2: the code hashes match (`e72_code_manifest.json`).
- W3: no lease expiry: 0 "Releasing expired KV blocks" and 0 "Potentially invalid KV blocks" lines in the P logs of
  both arms.
- W4: C2 D saturated in at least 95% of the window steps. W5: C2 completes the window with 0 early finishes and at
  least 1 completed request. W6: the D pool is 6,144 blocks in both arms.
- W7: 8 of 8 probes complete in both arms, and each probe's NIXL load was done (no load failure).
- W8: C2 has the compact spec and LBHNC on P and D; EV has the stock spec with the VIEWS mode on P and D.

## 7. Decision rule (frozen)
- Q1 (needs W1, W2, W4, W5, W6): REPEATS if |d| <= 0.5%; DRIFTS if 0.5% < |d| <= 2%; DIFFERS if |d| > 2%.
- Q2 (needs W1 to W6, W8): STEP-CONFIRMS-CLEAN if R_clean is inside [1.29524, 1.37748]; STEP-ABOVE-CLEAN above it;
  STEP-BELOW-CLEAN below it. If W3 fails: Q2 INVALID (a lease expired).
- Q3 (needs W1, W2, W3, W6, W7, W8): EXACT if 8 of 8 probes have identical D token ids; NOT-EXACT otherwise. If W3 or
  W7 fails: Q3 INVALID.
- The words are reported side by side; there is no combined word.

## 8. Budget (Modal prices of 2026-11-12)
- Two A100 80GB, 8 cores, 160 GiB: $6.65/h. EV: one pair of 31B engine builds (about 250 s) and the probe (about
  60 s). C2: builds, 22,000 work steps at about 74 ms (E-71 C), the probe: about 1,950 s. With the snapshot and the
  start: about 2,400 s, about $4.4.
- Limits (A1: EV 700 s, probe watchdog 420 s): EV arm 600 s (D watchdog 300 s after the build); C2 arm 2,450 s
  (D watchdog 2,100 s after the build);
  container 3,100 s. Cost cap $6.29 for one try. If EV fails, C2 does not run.
- Credits: about $13.9 left (2026-11-16, after E-71). The D-118 rule (credits at least 2 x the cap) holds: $12.58.
- No rehearsal: the credits do not hold a rehearsal and the main try under the D-118 rule. EV is the smoke arm of the
  new probe path, and the C2 workload loop is the E-71 code. Mac stub tests cover the file protocol.

## 9. Threats
1. Q2 compares C2 with the E-71 stock arms of another session (another host may differ). Q1 measures the
   cross-session repeat of compact; the stock A/A in E-71 was 0.03%.
2. One seed, one pool, one workload, eager, one host: E-71 limits stay.
3. Q3 uses 8 prompts of one length and 64 output tokens, batch 1; it says nothing about batched decode numerics.
4. The probe runs after the workload in C2 and on fresh engines in EV; the engines are otherwise idle.
5. If the lease setting does not reach the NIXL scheduler, W3 fails (the check exists for this).

## 10. Files
`README_E72.md`, `run_e72.py` (the P and D roles; imports `run_e71`), `e72_stage.py`, `analyze_e72.py`,
`modal_e72.py`, `stub_e72.py`, `test_e72.py`, `e72_code_manifest.json` (written by `e72_stage.py --manifest`).

## Amendments

### A1 (2026-11-16, before any run; the pre-run review R-121, `dossier/REVIEW_E-72_PRERUN_2026-11-16.md`)
R-121 gave AMEND_THEN_RUN with two blockers. No data existed. A1 replaces the parts it names.
- **B1. An EV failure no longer stops C2.** The session stops after EV only if an EV engine did not start (no
  `engine_seconds`). W3 is split: W3c (C2: 0 lease lines, the lease line set, P ok) gates Q2; W3 (both arms) gates Q3.
  So an EV probe failure costs Q3 only. A retry would not be possible under the D-118 rule.
- **B2. Q3 power.** The needle question asks for "the passphrase only", so with EOS allowed each probe would give about
  5 to 10 tokens. A1 sets `ignore_eos` (64 tokens per probe, 512 in all) and records the logprob of each sampled token
  (`logprobs` 1). Reported, not a word: the number of probes with bit-equal logprobs, the tokens compared, and
  `num_cached_tokens`. The frozen Q3 rule on tokens does not change.
- **F1.** After the abort, D steps until its scheduler holds no request (limit 60 s), and D steps while it waits for a
  probe file, so the frees and the release notices reach P. The P loops stop on `d_done`.
- **F2. Q2 is noise-limited.** The band top (1.37748) is only 0.13% above E-71 C: STEP-ABOVE-CLEAN starts at C2
  above 7.5188, while REPEATS allows 0.5%. So Q1 REPEATS together with Q2 STEP-ABOVE-CLEAN is possible, and it means
  the same gain within noise. Report both numbers.
- **F3.** W7 cannot see a silent local recompute of a probe (the policy "fail" catches a failed load). The D records
  hold `num_cached_tokens` for a report.
- **F4.** The mode is read back from the hooks (`splice.mode`), and W8 also needs a non-empty `hooked` list with no
  `problems` in every role, so a failed hook gives INVALID, not NOT-EXACT.
- **F5. Positive control for W3.** W3 and W3c need the line `'kv_lease_duration': 600` in the P log (vLLM prints its
  non-default arguments at the start), so an empty log cannot pass.
- **F6.** The probe watchdog is 420 s (Section 8 said 300 s); the EV arm limit is 700 s (the build and the probe).
- **F7.** Credits read live before the request (2026-11-16, `modal billing summary`): $15.70 of the $30 monthly
  credit used, so about $14.30 left. The cap $6.29 holds the D-118 rule ($12.58). The design is committed before
  the run, and the manifest is regenerated after every code edit.

## Result (2026-11-17; post-run review R-122 ACCEPT_WITH_CHANGES, `dossier/REVIEW_E-72_POSTRUN_2026-11-17.md`)
Run `results/e72-20261117T004531Z` (D-126, we: "Approve, cap $6.29"; two A100-SXM4-80GB, Modal Azure uk,
2,030 s: EV 214 s, C2 1,807 s; about $3.74). W1 to W8 pass.

| Field | C2 (E-72) | C (E-71) |
|---|---|---|
| Decode tokens per work step | 7.51385 | 7.50915 |
| Running requests per step | 7.515 | 7.510 |
| Saturated | 99.99% | 99.99% |
| Preemptions / completed / added | 6 / 62 / 79 | 6 / 63 / 79 |
| Step time, tokens/s | 74.5 ms, 100.85 | 74.2 ms, 101.25 |

- **Q1 REPEATS.** d = +0.063%. The compact result repeats in a new session on another host.
- **Q2 STEP-CONFIRMS-CLEAN.** R_clean 1.37657, inside [1.29524, 1.37748] by 0.001. 0 lease lines in both arms;
  the P and D logs show `'kv_lease_duration': 600` and D reports 0 expired and 0 failed transfers. Without the 1%
  floor, R_clean is 0.94% above the interval top. Q1 and Q2 together show that the E-71 excess over the model is
  reproducible (compact runs about 0.9% more requests per step than the model), not noise. Q2 uses the E-71 stock
  arms of another session and region (not tested).
- **Q3 EXACT.** 8 of 8 probes have identical D tokens (512 compared) and bit-equal sampled-token logprobs against
  EMULV (the stock cache with the same view hooks; not plain stock). Both arms answer the gold passphrase 8 of 8. Only
  the first 4 to 6 tokens of each probe are a real answer; `ignore_eos` then forces other text, which still tests the
  numeric identity.
- **NIXL evidence (R-122).** `num_cached_tokens` (0) proves nothing: at the pin it is P's prefix-cache count. The D
  logs show "Avg prompt throughput: 0.0 tokens/s" and "External prefix cache hit rate: 100.0%" in the probe windows,
  successful NIXL transfers of 1,212.5 MB (C2) and 1,412.5 MB (EV) each, all 16 probes ended "length" under the
  "fail" policy, and each probe took about 4.7 s (the 64-token decode alone). A silent local recompute is excluded,
  but only 7 transfers per arm appear in the stats windows, so 8 of 8 loads are not counted from the logs. W7 in
  `analyze_e72.py` checks the finish reason and the count only.

**Allowed wording (R-122, for E-71 plus E-72):** "On two A100 80GB at one pool (6,144 blocks) and one seed, the
compact cache gave 1.376 times the stock decode tokens per work step. A second compact run repeated this to 0.06%.
With a 600 s NIXL lease, no lease expired and the result held, inside the frozen band by 0.001 (the band has a 1%
floor). Compact has about 0.9% more requests per step than the model, and that excess is reproducible. On the P/D
path, the compact engine gave the same 64 output tokens, with bit-equal logprobs, as its stock EMULV twin on 8 needle
prompts at batch 1."

**Not claimable:** other seeds, pools, workloads, CUDA graphs or hosts; exactness against plain stock; batched-decode
numerics; answer quality beyond the first few tokens; "verified by `num_cached_tokens`"; "8 of 8 loads counted in the
logs".

**Cost:** about $3.74 of Modal credits (cap $6.29); about $10.56 left.
