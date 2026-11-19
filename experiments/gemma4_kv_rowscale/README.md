# E-74: one FP8 scale per row for the compact K=V cache, on the stored Gemma 4 31B activations (preregistration)

Status: PREDECLARED 2026-11-18; amendment A1 2026-11-18 (R-126, before any E-74 data). RUN 2026-11-18: run 1 INVALID
(V8 only); amendment A2 after run 1 (one validity check, our OK); GO after A2. Post-run review R-127 (reviewer):
CONFIRM_WITH_CORRECTIONS (`dossier/REVIEW_E-74_POSTRUN_2026-11-18.md`). CLOSED. Section 13 has the record. Free (the
Mac CPU, no GPU, no spend). Design review (S2): R-126 (reviewer): AMEND_THEN_RUN
(`dossier/REVIEW_E-74_PRERUN_2026-11-18.md`); every finding is answered in A1 (the last section). S3: Section 12.
Idea: I-182 (the FP8 limit of the arXiv paper 2611.23035). Decision: D-128 (we, 2026-11-18: "Yes, start the
plan"). The run needs our OK. Journal entry: Section 6.1 E-74.
Labels: FACT (a source, a log or a computation was read), INTERPRETATION (reasoning), HYPOTHESIS (not tested). All
thresholds are project heuristics; they are the E-54 thresholds, so that the two results can be compared.

## 1. Question
On the stored real Gemma 4 31B activations of E-54, does the compact row [K_rot | V_unrot | V_rot] (640 values) with
ONE FP8 scale per row (per token and KV head, scale = amax of the row / 448, no calibration) give an attention-output
error no larger than the standard cache with vLLM's `fp8_per_token_head` scales (1,024 values, one K scale and one V
scale per token and KV head)?
"No" is the KILL of Section 7: a pooled error ratio above 1.5 against that baseline, for the one-scale row and for
its two-scale fallback.
(A1, R-126 B1) The comparison with the stock FP8 cache at KV scale 1.0 is reported, not decided: that cache has one
grid phase per layer, so a ratio against it measures its luck (R-74) more than the row design.

## 2. Rubber duck
- Where the question comes from (FACT): a reader of the paper asked by email on 2026-11-18 if one scale per row would
  tie the K scale and the V scale "by construction", or if the rotary part of the key makes one scale too coarse. The
  paper says that the compact row fails at KV scale 1.0 (E-53) and passes with tied, calibrated per-layer scales that
  need an accuracy of 0.1% to 0.25% (E-54). No run of this project tested a scale per row.
- What vLLM does (FACT, vLLM 3966924d566340e971a6da313e790d05492aa505, read on 2026-11-18; R-126 checked each line):
  `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py` lines 169 to 275: for each (token, KV head) the write
  kernel computes `k_scale = max(max|K| / QUANT_MAX, 1e-6)`, stores `clamp(K * (1 / k_scale))` as FP8, and does the
  same for V with its own `v_scale` (QUANT_MAX is the FP8 maximum, 448 for E4M3).
  `vllm/v1/attention/backends/triton_attn.py` lines 414 to 475: the page holds [K (hs) | K scale (4 bytes) | V (hs) |
  V scale (4 bytes)], and the two scale caches are strided float32 views with their own offsets.
  `vllm/v1/attention/ops/triton_unified_attention.py` lines 41 to 57 and 499 to 589: the read kernel casts the codes
  to the dtype of Q (BF16 here), computes `S = dot(Q, K) * (scale * k_scale[token])` in float32 and
  `acc += dot((P * v_scale[token]).to(V.dtype), V)`. (A1, R-126 B3) The per-tensor FP8 path rounds too: it casts
  codes x scale to the dtype of Q and computes `dot((P * p_scale).to(V.dtype), V) / p_scale` (lines 53 to 57 and 590
  to 599).
- The candidate ROW1 (INTERPRETATION, algebra): store the codes of r / s, with r = [K_rot | V_unrot | V_rot] and
  s = max(max|r| / 448, 1e-6) per (token, KV head). The K view is codes 0 to 511 and the V view is codes 128 to 639,
  both with the scale s. The query is the folded query of the paper, [q_rot | (q w)_unrot], rounded to BF16. Because
  the two views share one scale, the scale is tied with no calibration. HYPOTHESIS (engine, not tested here): because
  the scale caches are strided views, both can point to one 4-byte slot after the row (644 bytes against 1,032 for
  the standard per-token-head page, a factor of 0.62; the slot is 0.6% of the row).
- The risk that the reader names (INTERPRETATION): w is 0.060 to 0.065 (FACT, E-54 README Section 4.6), so K_rot is
  about 16 times smaller than V. With one scale, the K_rot codes are about 16 times smaller than the V codes. FP8 E4M3
  keeps 3 mantissa bits down to |code| = 2^-6, and below that the codes are subnormal (step 2^-9). So ROW1 loses
  accuracy only where K_rot codes are subnormal. The run counts these codes. (A1, R-126 F3) The reviewer expects
  fewer than 1% of them, because FP8 is a floating-point format and K_rot codes of about 1 to 40 are normal numbers.
- (A1, R-126 B1) Why R_A is expected near 1 (INTERPRETATION): K_rot is about w times V_rot, so the largest value of
  a row is almost always in the V part. Then the ROW1 scale equals the V scale of PTH, and the V codes of the two
  arms are the same. Only the K path differs: PTH rounds K on its own grid, and ROW1 reads K_unrot as w times the V
  code. So a GO follows nearly from the algebra and the number format. The run tests that on real values, where a
  sink row or a strong channel can break it, and it gives the numbers that the reader asked for.
- The fallback ROW2 (INTERPRETATION): the E-54 tied row with a scale per row: codes of [K_rot / ks | V_unrot / vs |
  V_rot / vs], vs = max(max(max|V|, max|K_rot| / |w|) / 448, 1e-6) and ks = float32(w x vs), with the plain permuted
  query. K_rot then uses the full code range. ROW1 against ROW2 answers the reader's question about the rotary part.
  (A1, R-126 F2) HYPOTHESIS (engine, not tested here): ROW2 can also use one scale slot, if the softmax scale of the
  layer carries the factor w (line 547 multiplies the score scale and the K scale).
- What is different from E-54: no calibration and no frozen scale file for the new arms. The scale of a row comes from
  that row only. So every row has its own FP8 grid phase. HYPOTHESIS: this removes the phase lottery of one scale per
  layer (R-74), but it also removes the gain of a calibrated phase. (A1, R-126 B2) The BOS row is the same vector in
  every prompt, so its phase is one draw, not seven. The paired scale factors of Section 4.2 measure that noise.
- Expected (HYPOTHESIS, written before any ROW or PTH value was computed): R_A (ROW1 / PTH) 0.90 to 1.15; ROW1 /
  STOCK 0.80 to 1.20; ROW1 / TIED 1.2 to 1.7 (the calibrated tied row of E-54 stays better); ROW1 / ROW2 0.95 to
  1.10; fewer than 5% of the K_rot codes subnormal. (A1) The reviewer's prediction: R_A 0.97 to 1.05, ROW1 / STOCK
  0.85 to 1.10, ROW1 / TIED 1.2 to 1.6, subnormal codes under 1%.
- What can make the result wrong: a wrong copy of the vLLM write rule; a Mac CPU pipeline that differs from the E-54
  GPU pipeline (checks V2 and V8); a control that cannot fail (checks V5 to V8, and the CPU tests of Section 12).
- Not in E-74: an engine build, a kernel run, a memory measurement and a model-level test. E-74 stays at the
  attention level. A GO does not show that the compact saving and the FP8 saving stack in an engine.

## 3. Hypotheses
- H1 (primary): GO (Section 7): ROW1 has a median R_A of at most 1.25 over the 8 scale factors, every factor at
  most 1.5, the kernel emulation at most 1.25, no layer above 2 and no test prompt above 1.5.
- H0: KILL: the median R_A is above 1.5 for ROW1 and for ROW2.

## 4. Setup (frozen)

### 4.1 Hardware, software and data
- The Mac CPU (Apple silicon), torch from `uv run --no-project --with torch` (the version goes in the run record),
  float32, one process. No GPU and no network.
- Primary data: the 7 E-54 test prompts (6 prompts of 8,192 tokens and 1 of 32,768; 256 query rows each; the 10
  global layers): `experiments/gemma4_kv_scaletie/results/e54-h100-20261102/capture/` (70 files, 7.89 GB) with
  `e54/capture_run.json` (the SHA-256 of each file) and `e54/metrics.json` (the E-54 errors per cell).
- Replication data (reported, not in the rule): the 4 E-53 prompts of 8,192 tokens:
  `experiments/gemma4_kv_accuracy/results/e53-h100-20261101/capture/` (40 files, 3.36 GB). These prompts calibrated
  the E-54 scales, so the calibrated arms are in sample there. PTH, ROW1 and ROW2 have no calibration.
- The capture holds q and k after RoPE, v after its norm and the engine output o, in BF16, from a BF16 cache.
- Frozen scales of the E-54 arms: `experiments/gemma4_kv_scaletie/scales_e54.json` (SHA-256
  `8efe6e9076e28e011ff560106665998434b9403a1ddd9f653bdd98e5c718bd43`).

### 4.2 Arms (per prompt and global layer; the reference is the float32 causal GQA attention on the BF16 values)

| Arm | Storage | Role |
|---|---|---|
| STOCK | standard FP8 cache, ks = vs = 1.0 (E-54 code) | reported baseline; replication control (V2) |
| CALSTD_JOINT | standard FP8 cache, the frozen E-54 joint scales (E-54 code) | reported; replication control (V2) |
| TIED | the E-54 tied row with the frozen per-layer scales (E-54 code) | reported; replication control (V2) |
| PTH | standard FP8 cache with the vLLM per-token-head rule: a K scale and a V scale per (token, KV head) | the decision baseline |
| ROW1 | the compact row with one scale per row (Section 2); folded BF16 query | the candidate |
| ROW2 | the compact row with vs per row and ks = float32(w x vs); plain permuted query | the fallback (PIVOT) |
| ROW2STD | standard FP8 cache with the ROW2 scales of each row (K with ks, V with vs) | algebra control (V7) |
| ROW1_E5M2 | ROW1 with E5M2 codes (one mantissa bit less) | sensitivity control (V6) |
| ROW1_ID | (A1, B4) ROW1 with no cast: x times (1 / s) times s | algebra control (V8) against the E-54 BF16C record |
| ROW1_Q32 | (A1, F1) ROW1 with the folded query in float32 | reported: the cost of the BF16 query |
| PTH_X0, ROW1_X0 | (A1, B2) PTH and ROW1 with position 0 (the BOS row) exact | reported |
| ROW1_KROT, ROW1_KUN, ROW1_V | (A1) ROW1 with only K_rot, only the K use of V_unrot, or only the V view from the codes | reported: where the error comes from |
| PTH_K, PTH_V | (A1) PTH with only K or only V from the codes | reported |

- The cast of PTH, ROW1 and ROW2 copies the vLLM write kernel: x times (1 / scale) in float32, clamp at +-448, round
  to nearest even to E4M3 (`analyze_e53.fp8`). The scale is a float32 value with the floor 1e-6.
- (A1, R-126 B2) Paired scale factors: PTH, ROW1 and ROW2 run 8 times, with every row scale multiplied by
  2^(j / 8), j = 0 to 7. j = 0 is the vLLM rule. A factor of 1 or more cannot saturate. The same j is used for the
  two arms of a ratio. All other arms use j = 0.
- Two read arithmetics for PTH, ROW1 and ROW2 at j = 0: (1) float32: the attention on codes x scale; (2) the kernel
  emulation (KEM): S = (q . codes_K) x k_scale in float32, then (P x v_scale) rounded to BF16, then the product with
  the V codes. (A1, R-126 B3) STOCK_KEM, reported, rounds P to BF16 (its codes are exact in BF16 at scale 1.0). KEM
  rounds the normalized P; the kernel rounds the P of an online softmax and divides by the unrounded sum, so KEM is a
  statistical match, not a bit-exact one. KEM cannot move a verdict much: its rounding is about 0.2% against FP8
  errors of about 3%.

## 5. Metrics
- err_X(p, l): ||o_X - ref|| / ||ref|| (float32); errk_X for the kernel emulation; err_X(p, l, j) for factor j.
- P_X: the RMS over the 10 layers of the median over the 7 prompts of err_X(p, l) (the E-54 pooled statistic).
- Primary (A1): R_A(j) = P_ROW(j) / P_PTH(j) for j = 0 to 7; R_A_med = the median over j; R_A_max = the largest;
  R_A_kem at j = 0. The same for ROW2.
- Guards at j = 0: g_A(l), the median over the prompts of err_ROW / err_PTH per layer; R_A(p), the per-prompt ratio
  of the RMS over the layers.
- Reported only: P of every arm; the 8 values of R_A; ROW1, ROW2 and PTH against STOCK (pooled, per layer and per
  prompt, with the R-74 null of a random phase: median 1.02, 95th percentile 1.19, 99th percentile 1.26), against
  TIED and against CALSTD_JOINT; ROW1 / ROW2; the ratios per prompt class (plain 8K, chat, long); the share of
  subnormal codes (|code| below 2^-6, not zero) in K_rot, V_unrot and V_rot of ROW1 and in K of PTH; the share of rows
  whose largest value is in K_rot; the arms of Section 4.2 marked "reported"; every statistic again on the 4 E-53
  prompts; the bytes per row (644 against 1,032).
- Minimum effect that matters: a pooled ratio of 1.25 (as E-54). The computation is deterministic: no A/A run. The
  spread over the 8 scale factors is the noise measure (A1, B2).

## 6. Validity checks (a failed check makes the run INVALID)
- V1: every capture file has the SHA-256 of its run record (70 files of E-54, 40 files of E-53), and the scales file
  has its frozen SHA-256.
- V2 (replication; A1, F5): on the E-54 prompts, the pooled P of STOCK, CALSTD_JOINT and TIED of this run is within
  0.2% of the value from the E-54 `metrics.json`, and every cell error of these three arms is within 1% of its E-54
  value. The run stops after the first cell if that cell is outside 1%.
- V3: the identity K_unrot = w x V_unrot holds within 1e-2 in every cell, the module w of every global layer is one
  BF16 value on the unrotated dims (as E-54 V3), and (A1, F2) w is above 0.
- V4: the reference and every primary ratio are finite, and the kernel emulation has only finite values.
- V5 (A1, F6): at j = 0, PTH, ROW1 and ROW2 have no value above 448 x (1 + 1e-5) before the clamp, and the largest
  |code| of a row is 448 in at least 99.9% of the rows of PTH (K and V) and of ROW1.
- V6 (sensitivity): P_ROW1_E5M2 / P_ROW1 >= 1.5.
- V7 (algebra control of ROW2): 0.90 <= P_ROW2 / P_ROW2STD <= 1.10.
- V8 (algebra control of ROW1; A1, B4): on the E-54 prompts, every cell error of ROW1_ID is within 1% of the E-54
  BF16C value of that cell, and the pooled value is within 0.2%.

## 7. Decision rule (frozen; project heuristics; A1 after R-126 B1 and B2)
A row design X (ROW1 or ROW2) PASSES if all of these hold: R_A_med of X is at most 1.25; R_A_max is at most 1.5;
R_A_kem is at most 1.25; every layer has g_A at most 2.0; every test prompt has R_A(p) at most 1.5; all values are
finite. X FAILS if its R_A_med is above 1.5.

| Outcome | When | Meaning and action (we confirm each step) |
|---|---|---|
| INVALID | a check of V1 to V8 fails | no result; fix the cause; a rerun is free but needs a note of the change |
| GO | ROW1 passes | one scale per row gives the compact row the attention accuracy of an uncalibrated standard FP8 cache (the vLLM per-token-head mode); it does not show parity with the calibrated tied row of E-54, which is expected to stay better; send the numbers to the reader; add one paragraph to a v2; the next question is the engine design (the scale slot and the write), then a model-level test, and each needs a new plan |
| PIVOT | ROW1 does not pass and ROW2 passes | the rotary part needs its own derived scale; report it with the decomposition |
| KILL | ROW1 fails and ROW2 fails | a scale per row does not help the compact row; check the decomposition for a bug first; the FP8 route stays with the calibrated tied scales of E-54; send the numbers to the reader |
| INCONCLUSIVE | every other result | report each failed condition; no claim; we decide |

- FACT (logic): GO, PIVOT and KILL cannot hold together.
- A GO is a kernel-level statement about the attention error. It is not a claim that the two savings stack, and it
  is not a claim that FP8 on the compact row is harmless for the model output.
- (A1) R-126 estimates a GO as the likely outcome (60 of 60 synthetic draws without a sink row). A GO is then a
  confirmation on real values, and the reported numbers (PTH against STOCK and TIED, the decomposition, the subnormal
  share) carry most of the new information.

## 8. Controls and baselines
- Decision baseline: PTH (what vLLM ships as a dynamic FP8 KV mode). Reported baseline: STOCK (what vLLM does by
  default for this checkpoint), with the R-74 null band.
- Replication control: the three E-54 arms against the E-54 record (V2).
- Algebra controls: ROW1_ID (V8) and ROW2STD (V7). Sensitivity control: ROW1_E5M2 (V6). Scale rule: V5.
- Negative controls (CPU tests, Section 12): a wrong w in the folded query, a V view with the wrong offset, a scale
  from the wrong row and a scale that is too small must each fail a test.
- A/A: not needed (no timing; deterministic).

## 9. Budget
- Money: $0. No Pod, no Modal call, no download.
- Time: the timing test of Section 12 on synthetic tensors of the real sizes first; cap 3 hours of Mac CPU.

## 10. Threats to validity
1. Attention level only: 11 prompts, 10 layers, 256 query rows per prompt, one checkpoint (FP8 weights, BF16 cache).
2. The kernel emulation copies the arithmetic of the pinned Triton kernel from its source. No kernel ran.
3. The E-54 prompts were seen by E-54. The new arms have no fitted parameter, but the thresholds were chosen with
   knowledge of the E-54 errors of STOCK and TIED.
4. A dynamic scale changes the error of the BOS sink row and of a few strong channels at random. The pooled statistic
   can hide a bad layer; the per-layer guard and the 8 scale factors are the answer to that.
5. The engine statements of Section 2 (one shared scale slot, the write of the 640-value row) are a HYPOTHESIS. E-74
   does not test them.
6. (A1) The expected outcome is a GO that follows nearly from the algebra. E-74 is a confirmation and a source of
   numbers, not a hard test of a risky claim.

## 11. Files
- `analyze_e74.py`: step 1 metrics (per cell), step 2 checks and the frozen rule. It imports the E-54 and E-53 code.
- `test_e74.py`: CPU tests on synthetic data (Section 12). `time_e74.py`: the timing test on synthetic tensors.
- `results/e74-mac-<date>/`: `metrics_e54.json`, `metrics_e53.json`, `summary.json`, `decision.txt`.

## 12. Verification record (stage S3)
See the end of this section for the record after A1. Tests (`test_e74.py`, synthetic data, no capture file): the PTH
cast against a direct copy of the vLLM rule; (A1, B4) with w = 2^-4 and no rotated part, the K codes of PTH and the
V_unrot codes of ROW1 are bit-identical and the outputs agree; ROW1 with no cast against the reference; ROW2 against
ROW2STD; the kernel emulation against float32; the negative controls; the paired scale factors; the rule on synthetic
GO, PIVOT, KILL, INCONCLUSIVE and INVALID inputs; the guards.

Record after A1 (FACT, 2026-11-18, the Mac, torch 2.15.0, 4 threads): `test_e74.py`: 13 tests pass (2.2 s). The
timing test `time_e74.py` (synthetic tensors of the real sizes, all 17 + 4 arms): 7.9 s for an 8,192-token cell and
37.4 s for the 32,768-token cell. Estimate: 60 x 7.9 + 10 x 37.4 = about 850 s for the E-54 set and 40 x 7.9 = about
320 s for the E-53 set, about 20 minutes with the file reads, inside the 3-hour cap. No capture file was read.

## 13. Run record
Run 1 (2026-11-18, the Mac CPU, torch 2.15.0; we: "Yes, run it"; `results/e74-mac-20261118/`): all 110 cells
finished (70 of E-54, 40 of E-53), exit code 0. Frozen word: INVALID (V8). Only V8 failed. The author read only the
check values before this record. (R-127 C4) The run-1 summary file already held the decision statistics.
- FACT (V2 passed): the pooled P of STOCK, CALSTD_JOINT and TIED is 0.999999, 1.000001 and 1.000002 times the E-54
  record, and the largest cell difference is 4.1e-5 of the cell value. The Mac CPU reproduces the H100 numbers.
- FACT (the V8 failure): the pooled ROW1_ID value is 1.000001 times the E-54 BF16C record (inside 0.2%). In 63 of 70
  cells the per-cell ratio is 0.99997 to 1.0001. The 7 cells outside 1% are all in layer 29, one per prompt (ratios
  1.66 to 1.82). In layer 29, w = 0.0625 = 2^-4, so the BF16 fold of the query is exact, and the BF16C error is only
  float32 noise: 1.7e-7 to 3.1e-7 (other layers: 2.1e-3 to 1.0e-2). ROW1_ID adds one more float32 rounding (x times
  1 / s times s), and its error there is 3.1e-7 to 5.4e-7. The largest absolute difference between the two check
  values in any of the 70 cells is 2.4e-7.
- INTERPRETATION: the code is correct; the check is wrong. A relative limit cannot work where the reference error is
  at the float32 noise floor. This was predictable: the E-54 README (Section 4.6) states that w is a power of 2 in
  layer 29.
- Amendment A2 (applied 2026-11-18 after run 1; we: "Yes, apply A2"; a post-run change of one check, labelled
  as such; no decision value was read before it): V8 per
  cell becomes |ROW1_ID - BF16C| <= 0.01 x BF16C + 1e-6. The floor 1e-6 is about 2,000 times smaller than the smallest
  BF16C error outside layer 29 and about 30,000 times smaller than the FP8 errors (about 3e-2). The pooled V8 limit
  stays. With A2, no cell is recomputed: step 2 (the checks and the rule) runs again on the stored metrics.

Result after A2 (2026-11-18; `summary.json`, `decision.txt`). Frozen word: GO after A2, a post-run change of one
validity check. The decision statistics are identical to run 1 (word INVALID by V8): `summary_run1_before_A2.json`
already held them. The author did not read them before A2 (R-127 C4). Checks V1 to V8 pass. Post-run review (S9):
R-127 (reviewer): CONFIRM_WITH_CORRECTIONS (`dossier/REVIEW_E-74_POSTRUN_2026-11-18.md`); the reviewer recomputed every
decision value and found no bug; its corrections C1 to C4 are applied below and marked "(R-127 Cn)".
- FACT (the decision values): ROW1: R_A median 1.021 over the 8 factors (0.981 to 1.062), KEM 1.061; per layer
  g_A 0.892 to 1.180; per prompt 1.027 to 1.092. ROW2: R_A median 1.012 (0.977 to 1.030), KEM 1.030.
- FACT (pooled P at j = 0, the vLLM rule, E-54 prompts): PTH 0.0187, ROW1 0.0199, ROW2 0.0193, STOCK 0.0432,
  CALSTD_JOINT 0.0256, TIED 0.0289. So at j = 0, ROW1 is 0.460 times STOCK, 0.776 times CALSTD_JOINT and 0.686 times
  TIED. The E-53 prompts give the same picture at j = 0 (ROW1 0.446 times STOCK and 0.678 times TIED; R_A at most
  1.041). (R-127 C2) These comparisons hold only at j = 0: the median over j of P_ROW1 is 0.0427, which is 0.99 times
  STOCK and 1.48 times TIED, and ROW1 is worse than TIED at 6 of the 7 offsets j > 0. The R-74 null band does not
  apply to these ratios (it models a random phase of one scale per layer).
- FACT (the scale rule and the risk of the reader's question): in every cell, the largest value of every row is in
  the V part (the share in K_rot is 0), and the ROW1 scale equals the PTH V scale in every row. Subnormal K_rot codes:
  0.44% on average, at most 0.81%. Decomposition (pooled P): ROW1 with only K_rot from the codes 0.0062, only the K
  use of V_unrot 0.0157, only the V view 0.0124; PTH with only K 0.0148, only V 0.0112. ROW1 / ROW2 1.031. A float32
  folded query lowers the ROW1 error by 3.0%. The BOS row exact: PTH 0.0177, ROW1 0.0186.
- FACT (not predicted): the paired factors at j > 0 make every per-row arm much worse: PTH 0.0187 at j = 0, and
  0.0376, 0.0533, 0.0541, 0.0358, 0.0280, 0.0962 and 0.0473 at j = 1 to 7; ROW1 and ROW2 follow PTH (R_A 0.98 to 1.06
  at every j).
- The author's hypothesis (before R-127): at j = 0 the largest value of each row maps to the code 448 with no
  rounding, so the strongest element of every row is stored exactly; at j > 0 it maps to 448 / g, which E4M3 rounds.
  EXPLORATORY test by R-127 (5 of 70 cells, PTH only; `exploratory/r127/q3.py`): mostly supported. In 4 of 5 cells,
  the row maximum kept exact removes most of the loss at j = 3 and j = 6, and only the maximum moved to the j = 6 grid
  at j = 0 recreates most of the j = 6 error (the K maximum is the main cause); in 1 cell, 2 to 4 outlier channels
  matter; the BOS row exact alone does not fix it. INTERPRETATION: the dynamic amax rule of vLLM stores the outlier of
  every row exactly, and a per-layer scale cannot do that. Not tested on all cells (the next free check).
- (R-127 C1) So the 8 factors are a scale-offset sensitivity, not a measure of grid-phase noise: j = 0 is the vLLM
  rule and a special storage, and the median over j mixes different storages. The paired ratio R_A is still a fair
  comparison of ROW1 and PTH at each j (both lose the exact maximum together; the V_unrot codes are identical), and
  j = 0 alone gives the same word (R_A 1.062, KEM 1.061).
- (R-127 C3) Predictions against measurements: R_A 0.90 to 1.15 expected, 1.021 measured (held). FALSIFIED: ROW1 /
  TIED 1.2 to 1.7 (the author) and 1.2 to 1.6 (R-126), 0.686 measured at j = 0; ROW1 / STOCK 0.80 to 1.20 (the
  author) and 0.85 to 1.10 (R-126), 0.460 measured at j = 0; P_PTH 0.035 to 0.045 (R-126), 0.0187 measured. The GO
  text of Section 7 says that TIED is "expected to stay better": at j = 0 it did not. Subnormal K_rot codes under 5%
  (R-126: under 1%), 0.44% measured (held).
- Safe claims (R-127): the GO is correct under the frozen rule after A2; ROW1 matches the vLLM per-token-head cache
  (R_A 1.02, at most 1.06) on these activations; the rotary part does not make one scale too coarse (no row has its
  maximum in K_rot; 0.4% of the K_rot codes are subnormal); at the exact vLLM rule, ROW1 has a lower attention error
  than STOCK and than the calibrated TIED on this data, and that advantage is exploratory and depends on the exact
  storage of the row maximum. Not to claim: that per-row scaling beats calibration in general, that the savings stack
  in an engine, that FP8 is harmless at the model level, that the one-slot engine route works, or that the GO was a
  hard test.
- Run time: 1,739 s for the E-54 set and 569 s for the E-53 set (about 38 minutes).

## Amendments
A1 (2026-11-18, after R-126, before any E-74 data; no ROW or PTH value on real data existed):
- B1: the rule uses only the PTH-referenced ratios; every STOCK-referenced number is reported with the R-74 null band
  (Sections 1, 5, 7 and 8). The reviewer's gate R_B(ROW) <= 1.25 x R_B(PTH) is the same condition as R_A <= 1.25.
- B2: 8 paired scale factors for PTH, ROW1 and ROW2; the decision value is the median, and every factor must be at
  most 1.5; PTH_X0 and ROW1_X0 are reported (Sections 2, 4.2, 5 and 7).
- B3: the text on the per-tensor path is corrected; STOCK_KEM rounds P to BF16; KEM is called a statistical match
  (Sections 2 and 4.2).
- B4: V8 (ROW1_ID against the E-54 BF16C record) and a new CPU test (Sections 4.2, 6 and 12).
- F1: ROW1_Q32 is reported. F2: the ROW2 engine route is a HYPOTHESIS with the one-slot variant; V3 requires w > 0.
  F3: the reviewer's bound is added to the risk text. F5: V2 is 0.2% pooled and 1% per cell, with a stop after the
  first cell. F6: V5 also checks that the largest code of a row is 448. F7: the wording of Section 7. F8: the GO text
  says "an uncalibrated standard FP8 cache". Question 7, item 2: the decomposition arms are reported.

A2 (2026-11-18, AFTER run 1; we: "Yes, apply A2"): the V8 cell limit becomes |ROW1_ID - BF16C| <= 0.01 x
BF16C + 1e-6 (`analyze_e74.v8_cell_ok`; a new CPU test; 14 tests pass). The reason is in Section 13: in layer 29
(w = 2^-4) the BF16C reference error is float32 noise, so a relative limit cannot hold. The pooled V8 limit, every
other check and the decision rule are unchanged. No cell was computed again; step 2 ran again on the stored metrics.
The run-1 summary and word are kept as `summary_run1_before_A2.json` and `decision_run1_before_A2.txt`. The author
read only check values before A2, but the run-1 summary file already held the decision statistics, and they are
identical after A2 (R-127 C4). The result is reported as "GO after A2".
