# E-65: does the compact K=V read save decode time at large batch and long context? (A100 microbenchmark + E-64 model; preregistration)

Status: CLOSED 2026-11-10 with the free Mac bracket (D-117; Sections 14 and 15). The GPU part did not run (no Secure
capacity). Predeclared 2026-11-10; S2 R-101 AMEND_THEN_RUN, applied as A1; R-102 CONFIRM_WITH_CORRECTIONS.
Idea: I-182 (v2 of the paper). Journal entry: Section 6.1 E-65. Origin: E-64 (Section 1). We, 2026-11-10:
"runpod has some credits left, if you want you can do the gpu measurement".
Labels: FACT, INTERPRETATION, HYPOTHESIS. Thresholds are project heuristics unless a source is given.

## 1. Question
E-64 found that the throughput gain of the compact cache depends on one assumed cost: does the compact KV read save
decode-step time? With the saving (W-A) long prompts reach 1.27 to 1.39; without it (W-B) 1.13 to 1.17. On one A100
SXM, over batches of 1 to 64 and contexts of 1K to 64K, how much of the byte saving of the global read becomes step
time, and with that measured step time (and the measured E-59 prefill time) in the E-64 model, which workloads reach a
1.25x decode-throughput gain? "No": no workload is GO under the measured model.

## 2. Rubber duck
- What is known (FACT): E-58 (weight-free, one A100) measured the 10 global-layer read at 3 x 32K: stock 5.076 ms,
  compact 4.582 ms (ratio 0.90 with block 48; bytes 0.625); at 2 x 64K the ratio was 0.97. The stock read ran at 78% of
  the DRAM peak, the compact read at 52% (distinct bytes). E-59 (the engine, CUDA graphs) measured 50.73 against 50.52
  ms per step at batch 3 and 32K: the read saves about 0.49 ms and the hook transform costs about 0.46 ms.
- What is not known: the read at large batch. With few requests the 3D decode kernel has few programs (192 on 108 SMs
  at 3 x 32K); with more requests it may become bound by DRAM bandwidth, and then the compact read could approach the
  byte ratio. Or another limit (the L2-to-SM traffic of the two overlapping views, which is the same in both layouts)
  keeps the saving small at every load. E-64 assumed one or the other.
- A new free fact (old data, read before this design): the E-59 traces time every prefill chunk. A chunk of 2,048
  tokens costs 0.614 + 0.0364 x (context / 1,024) s for stock and 0.639 + 0.0391 x (context / 1,024) s for compact
  (`prefill_fit.json`; residuals below 2%). So E-64 W-A (0.5 s per chunk) was too low, W-B (1.0 s) was close, and the
  compact build is about 6% slower in eager prefill (the hooks and the cache path).
- Why a microbenchmark: no 62 GB model download and no engine start; E-58 ran in 72 s of session time. The decode
  step outside attention (the GEMMs, norms, sampling) is the same in both layouts, so it enters as one fixed number
  from E-59.
- What would kill it early: the compact read stays near the stock time at every load (Phi near 0), and then W-M
  gives about the W-B labels. This is a useful answer: v2 then states a measured "no throughput gain".
- Order: the Mac code and tests (done), the different-model review, a rehearsal on one RTX A6000 (Ampere, 48 GB;
  the A40 has no stock now), the A100 run, the Mac part 2 (about 1 hour on one core at nice 10).

## 3. Hypotheses (written before any run)
- H1 (primary, W-B-like): Phi < 0.5 (Section 5), and no workload is GO under W-M. Prediction: Phi 0.1 to 0.35; the
  long-prompt workloads give Q_M 1.03 to 1.18; prompt median 1K is KILL.
- H0 (W-A-like): Phi >= 0.75, and the long-prompt workloads (D3, M1, M2) are GO under W-M.
- Secondary predictions: the read ratio C48 / S32 falls with the batch in the 3D path (B <= 8) and is between 0.70 and
  0.95 at B = 64; the hook cost of 10 layers stays below 0.6 ms at every batch; the sliding read of 50 layers grows
  about linearly with the batch.

## 4. Setup (frozen)

### 4.1 Hardware
- Main run (S6): 1 x A100 SXM 80GB, Secure Cloud, on demand, CUDA floor 13.0 (the E-57 to E-59 GPU type). Live read
  2026-11-10 (`list-gpu-types`, product POD, Secure): $1.79/h, stock LOW in US-MD-1 and US-WA-1. Container disk 30 GB,
  volume 20 GB (no model).
- Rehearsal (S5): 1 x RTX A6000 48GB (Ampere), Secure, $0.59/h, stock LOW in EU-SE-1 (same read). REH=1 caps the grid
  at B x L <= 2^17 tokens.

### 4.2 Software
- Image `vllm/vllm-openai@sha256:f255c4fad2f30ff5ee80db14d3a03afe9ae0f10d3470339864ff04745fe44cb3` (vLLM
  0.33.0rc1.dev229+g3966924d5, torch 2.14.0+cu130, Triton 3.8.0), the image of E-49 to E-59. No model download.
- Code reused unchanged: `../gemma4_kv_microbench/bench_e58.py` (the layouts, the kernel arguments, the reference
  attention), `../gemma4_kv_engine/compact_engine.py` and `compact_layout.py`, `../gemma4_kv_model/splice_e55.py`,
  `../gemma4_kv_capacity_model/e64.py` and `sweep.py`.
- Changed: `../gemma4_kv_capacity_model/poolsim.py` returns `prefill_chunks` (the tokens and the start context of each
  prefill chunk of a step). It adds a field only; a test recomputes one stored E-64 cell exactly.

### 4.3 The GPU grid (`bench_e65.py`)
- Points: B in {1, 2, 4, 8, 16, 32, 64} x L in {1,024, 4,096, 16,384, 65,536} with B x L <= 2^19, plus the tie point
  (3, 32,768): 25 points. B requests of L tokens each, one decode token per request.
- G (decides): the raw `unified_attention` read of 5 global layers (32 query heads, 4 KV heads, head 512), each with
  its own cache, in S32 (stock row 1,024, block 32, page 262,144 B) and C48 (compact row 640 through the two views,
  block 48, page padded to 262,144 B), filled from the same logical rows (E-58). One CUDA graph per layout; 3 rounds
  of 50 replays, the layout order alternating per round. seq_threshold_3D = 8 (the engine value with CUDA graphs), so
  B <= 8 uses the 3D path and B >= 16 the 2D path, as in the engine.
- SL: 5 sliding layers (32 query heads, 16 KV heads, head 256, block 16), each with min(L, 1,024) tokens per request;
  one graph, 50 replays. The same for both layouts.
- H: the hook transform (`ViewsSplice`, mode VIEWS) on 5 modules that do no attention, at batch B; one graph, 50
  replays.
- V1 per point: the S32 and C48 outputs of global layer 0.

### 4.4 The measured model W-M (`analyze_e65.py`)
- Decode step of a batch of B requests with mean context Lbar: stock T_OTHER + 2 G_S(B, Lbar) + 10 SL(B); compact
  T_OTHER + 2 G_C(B, Lbar) + 10 SL(B) + 2 H(B).
- T_OTHER = 50.733 ms (E-59 S_g) - 2 G_S(3, 32K) - 10 SL(3): the non-attention part of the E-59 stock step, fixed
  over B. So the model gives the E-59 stock step exactly at its point.
- Interpolation: G piecewise linear in B within each L row (linear extrapolation past the last B of a row), then
  piecewise linear in L between the rows (the 1,024 row below 1,024 tokens; linear extrapolation past 65,536). SL and
  H piecewise linear in B (their L = 1,024 values).
- Prefill: a chunk of n tokens from context c0 costs n / 2,048 x (a + b x (c0 + n / 2) / 1,024) s, with a and b per
  layout from `prefill_fit.json` (E-59 traces; Section 2). A mixed step costs its decode part plus its prefill part (as
  E-64).

### 4.5 Part 2 (`e65_model.py`, the Mac)
- The E-64 grid unchanged: U (unified node: D1 to D4), D (decode node: D1 to D4, M1, M2); pools 6,727 / 8,724 /
  11,314 / 14,672 / 19,027 blocks; seeds 0 to 4; horizon 200,000 and warm-up 40,000 steps; the E-64 engine model and
  generators; the E-64 inclusion filter (stock pool-bound share >= 0.5, max_num_seqs share <= 0.1).
- Arm M decides. Report-only arms: M_max (the decode part uses the longest context of the batch), M_pf_equal (the
  compact prefill uses the stock fit, as a production build with the transforms folded into the weights),
  M_other_prof (T_OTHER = 50.733 - 10.67 ms, the E-59 profile), M_gemm (T_OTHER grows 0.4% per request past 3).

## 5. Metrics
- Primary (decides): Q_M per workload and seed = the median over the counted pools (at least 3 of 5) of r_M, the
  compact / stock decode tokens per second under W-M.
- Reported (mechanism): per grid point, the read ratio G_C / G_S, rho = the W-M step ratio compact / stock, rho_A =
  the E-64 W-A step ratio at the same point, and phi = (1 - rho) / (1 - rho_A): the share of the W-A step saving that
  the measured step keeps (1 = W-A, 0 = no saving, negative = slower). Phi = the median phi over the points with B x
  L >= 2^18 (5 points).
- Secondary: the global-read share of the stock step, the hook cost per batch, the sliding read per batch, the round
  spreads, the report-only arms, r_step.
- MDE: the labels use the frozen thresholds; a read difference below 2% of the read is "no change" (MEASUREMENT.md).
- Repeats: 3 rounds of 50 graph replays per layout and point (the A/A is the round spread, V3).

## 6. Validity checks (a failed check makes the run INVALID)
- V1: at every point, the C48 output equals the S32 output (bit-identical, or a largest difference of 0.002).
- V2: the main run is on an A100 with vLLM 0.33.0rc1.dev229+g3966924d5.
- V3: at every point with B x L >= 16,384, the round spread (max / min - 1) of each layout is at most 3%.
- V4 (tie to E-58): at (3, 32K), 2 G_S and 2 G_C are within 5% of the E-58 reads 5.076 and 4.582 ms.
- V5: every planned point ran with no error.
- V6 (hook tie): at (3, 32K), 2 H is within 50% of the E-58 graph hook time 0.396 ms.
- Part 2: no run with zero decoding (E-64 V2); conservation (asserted in every step).
- Report only: telemetry (clocks and throttle reasons); the model attention at the tie against the E-59 profile.

## 7. Decision rule (frozen; project heuristics)
- REHEARSAL: a rehearsal or a stub run.
- INVALID: a check of V1 to V6 fails (the line names it); part 2 does not run.
- Per workload: Q_M per seed (Section 5); seed label GO if Q_M >= 1.25, KILL if Q_M < 1.10, else MID. The workload
  label is the seed label if all 5 seeds agree, else MID.
- One line: `E-65: U-D1 <label> | ... | D-M2 <label>`, and from part 1 `E-65 GPU: <word> | Phi ... | checks`.
- What the v2 paper may say: GO, a throughput gain for that workload "in a model with step times measured on one
  A100"; MID, the range; KILL, the capacity gain does not become throughput there. The E-64 labels stay as they are
  (assumed arms); E-65 adds the measured arm.

## 8. Controls and baselines
- Baseline: S32, the stock layout with the same kernel and graph settings, at every point.
- A/A: the 3 rounds per layout and point (V3).
- Positive control: V4 reproduces the E-58 reads at the E-58 shape; the `compact_fast` stub plant (C48 at the byte
  ratio) gives read ratio 0.625 and a higher Phi (test).
- Negative control: V1 (the same logical K and V give the same output in both layouts).

## 9. Budget
- Main run: 1 x A100 SXM at $1.79/h. Boot watchdog 1,500 s (`pod_cmd.py --cap 1500`), session watchdog 900 s,
  AUTO_STOP 240 s after the session, SSH limit 600 s. Cost cap $0.75 (25 min). Plan (INTERPRETATION, from E-58):
  create to SSH 4 to 8 min; setup 1 min; the benchmark about 4 to 7 min (25 points; Triton compiles about 2 min; the
  largest point about 10 s); copy 1 min. Pod about 15 min, about $0.45.
- S5 rehearsal: 1 x RTX A6000 at $0.59/h, REH=1. Boot watchdog 900 s; cost cap $0.25. VERIFY: (1) every rehearsal
  point runs with no error; (2) V1 on the GPU at every point; (3) the CUDA graph capture of the raw reads, the
  sliding reads and the hooks; (4) the 2D path at B >= 16 (no scratch error); (5) the round spread; (6) the time of
  each point, scaled to the A100 plan; (7) the session files, the code hashes, the REHEARSAL line, AUTO_STOP.
- Total cap $1.00. Part 2 runs on the Mac (free).
- Balance (live, 2026-11-10, `balance.py`): $2.64, no other spend running. The protocol floor is the cap plus $35, so
  the check fails; we decide (as D-95, D-98, D-101, D-103). If the balance falls below the planned cost before
  a create, no Pod is created.
- What the time cap cuts first: the points run by L, then B; the largest 64K points run last. A missing point fails
  V5 (INVALID), so the plan keeps a margin of about 2x.

## 10. Threats to validity
1. Isolation: the reads run without the rest of the step, so the overlap of attention with other kernels is not
   reproduced. The E-59 step at batch 3 ties the model at one point (T_OTHER).
2. T_OTHER fixed over B: the 31B GEMMs are bound by the weight read up to batch 64 on an A100 (INTERPRETATION:
   arithmetic intensity B flop per byte, below the A100 ridge of about 150), but norms and sampling grow a little.
   M_gemm reports +0.4% per request. A larger T_OTHER lowers the KV share and the gain.
3. Mean context: the model uses the mean context of a batch. In the 2D path with fewer programs than SMs, the longest
   sequence can set the time. M_max reports the other end.
4. Contiguous blocks and synthetic values: as E-58 (the kernel time does not depend on the values; an engine pool
   interleaves blocks).
5. Sliding layers: a cache of min(L, 1,024) tokens with no window argument stands for the windowed read of the
   engine (the same bytes). The same in both layouts.
6. The prefill fit comes from one A100 host (E-59), batch 3, eager prefill, contexts to 30K; longer contexts are a
   linear extrapolation. The compact prefill cost includes the run-time hooks (M_pf_equal reports a folded build).
7. One GPU type and one host; a different A100 host than E-58 and E-59 (V4 and the tie check the offset).
8. The E-64 limits stay (F7, F8 of `README_E64.md`): a model of the engine, synthetic workloads, a saturated decode
   node, KV transfer time ignored.
9. Not blind: the author knows the E-58 and E-59 numbers (Section 2) and the E-64 result.

## 11. Files
| File | Role |
|---|---|
| `bench_e65.py` | the GPU benchmark (one process; `--stub` on the Mac; `--rehearsal`) |
| `analyze_e65.py` | part 1: the checks, the W-M tables (`fit.json`), phi and Phi |
| `e65_model.py` | part 2: the E-64 grid under W-M; the labels (the Mac) |
| `prefill_fit.py`, `prefill_fit.json` | the prefill weights from the E-59 traces (old data) |
| `session_e65.sh`, `boot_end.py`, `stop_pod.py`, `versions.py` | the Pod session (`STUB=1` on the Mac, `REH=1` for the rehearsal) |
| `make_manifest_e65.py` | the SHA-256 of each file that runs, at the design commit |
| `test_e65.py` | the S3 tests |

## 12. Verification record (stage S3)
- Tests (2026-11-10): `uv run --no-project --with numpy --with torch --with pytest python -B -m pytest -q -p
  no:cacheprovider experiments/gemma4_kv_decode_scaling/ experiments/gemma4_kv_capacity_model/` gives 26 passed (8
  new): the grid (25 points, the caps); the stub benchmark (8 points, V1 identical) and its analysis (REHEARSAL, the
  tables); the `compact_fast` plant; the model interpolation (exact for bilinear tables, the extrapolation, the clamp
  below 1,024); the failed checks V1, V2, V3, V5; the prefill term; the poolsim prefill chunks (their sum equals the
  prefill tokens, unified and decode node); one stored E-64 cell recomputed exactly; a part 2 cell and the label rule.
- The stub session (`STUB=1 sh session_e65.sh` on the Mac, 5 s): the bench ends ok with 8 points, the code hashes,
  and an "E-65 GPU: REHEARSAL" line with no failed check.
- Not testable on the Mac (VERIFY in S5): the Triton kernels, the graph capture, the 2D path, the times.
- After A1 (2026-11-10): 27 tests pass; the stub session runs 9 points (3 of them one layout at a time) with no
  failed check. Frozen file hashes: `code_manifest.json` (13 files, `make_manifest_e65.py --working`).
- S4 (2026-11-10): D-116 (we: "GPU run too"). Live balance $2.64, no other spend running; `balance.py --need
  1.00` fails the $35 floor, which D-116 accepts. The account SSH key `runpod-masters-thesis` is registered.
- S5 attempt 1 (2026-11-10, about 21:10 UTC): four Secure create calls failed for capacity (HTTP 400 "no longer any
  instances available"), although the catalog read showed LOW stock: RTX A6000 in EU-SE-1 (CUDA floor 13.0, a 20 GB
  volume; then 13.2, no volume), RTX A5000 in CA-MTL-1 (13.2, 40 GB disk; then 13.0, 20 GB disk), L4 (13.0). No Pod
  was created (list-pods read after the calls); no cost. The rehearsal is not done. We: "Retry Secure later"
  (Secure Cloud only); the retry runs when the Mac bracket ends.
- S5 attempt 2 (2026-11-10, after the bracket, about 23:00 UTC): a fresh catalog read showed LOW Secure stock for 15
  GPU types; two creates failed the same way: A40 in EU-SE-1 (13.2, 30 GB) and RTX A4500 in EU-RO-1 (13.0, 30 GB).
  Six of six Secure creates failed in two hours, for five GPU types in four data centers. The request has the form
  that worked for E-46 (an image by digest), and the error is the capacity text, not 402 (balance) or 422 (body);
  the Runpod docs note that a 400 on every candidate can also mean a request problem, which stays open. No Pod was
  created; no cost. The GPU part is not run.

## 13. Pre-run review R-101 (2026-11-10, reviewer)
AMEND_THEN_RUN. R-101 read the pinned vLLM source (full SHA 3966924d566340e971a6da313e790d05492aa505) and ran the real
`e65_model.py` on planted G tables (pools 8,724, 14,672, 19,027; seeds 0 and 1; horizon 60,000). Scripts:
a temporary folder `r101/`.
- B1 (FACT from its simulation): the GPU result cannot reach GO. With the measured prefill, even the full byte saving
  (read ratio 0.65) gives decode-node D3 1.22, M2 1.09, D2 1.09 and U-D3 1.05; a ratio of 0.92 gives 1.15, 1.02, 1.03
  and 1.02. Mechanism: compact holds about 1.4 times the batch, so it preempts about twice as often, and each
  preemption prefills again at 0.6 s plus 0.036 s per 1K of context per 2,048-token chunk. So the GPU run can only
  move labels between KILL and MID. Fix: bracket part 2 on the Mac with planted ratios 1.0 and 0.625 before any
  spend, and let us decide if the mechanism number is worth the money.
- B2 (FACT from the source): the engine calls the sliding layers with window_size (1023, 0) and the full context;
  then the decode tile is 32 and, in the 3D path, only 1 or 2 of 16 segments overlap the window. The bench (no
  window, a 1,024-token cache) is faster than the engine at B <= 8 (about 2.2 against about 5.6 ms for 50 layers at
  the tie). T_OTHER absorbs the gap, which is wrong at B >= 16. Fix: the engine window and a shared out-of-window block.
- B3 (FACT from its simulation): about 25% of the decode-node D3 steps have B > 8 and a mean context above 40K, and
  30% of M2 steps a mean context above 65K, but the 65,536 row stops at B = 8 (the 3D path). Fix: rows at 32K to
  B = 16, (16, 64K), (2, 128K), (4, 128K); layouts one after the other at the largest points.
- B4 (FACT): the tie point runs last, so a V4 failure is seen only after all the money is spent. Fix: the tie first;
  V5 fatal only for the core points.
- Findings: F1 the gate arm should need M, M_pf_equal and M_gemm all at the threshold (the compact prefill includes
  the run-time hooks); F2 T_OTHER fixed over B favours GO; F3 SL and H need rounds and V3; F4 V1 needs a relative
  tolerance and a non-zero output; F5 V6 cannot fail; F6 L2 reuse at (1, 1,024); F7 the pinned thresholds and scratch
  match (vLLM main differs: do not read main for the pinned behaviour); F8 Phi is well defined; F9 the rehearsal GPU
  tests mechanics only, AUTO_STOP_DELAY 90 s.
- R-101 predictions (before data): Phi 0.10 to 0.30; Q_M decode-node D3 1.10 to 1.20, M2 1.00 to 1.10; no GO.
- State: amended as A1 (below). We chose "GPU run too" (D-116, journal Section 10).

## 14. The Mac bracket (A1 B1; 2026-11-10; reported, decides nothing)
FACT (`results/bracket/e65_bracket_r1.0.json` and `_r0.625.json`; part 2 with planted tables, 1,786 s and 2,258 s at
nice 10): median Q_M over the 5 seeds (seed range in brackets).

| Workload | r = 1.0 (no read saving) | r = 0.625 (full byte saving) |
|---|---|---|
| U-D1 | 1.030 KILL | 1.047 KILL |
| U-D2 | 1.055 KILL | 1.097 MID |
| U-D3 | 1.062 MID [1.054, 1.100] | 1.102 MID [1.093, 1.137] |
| U-D4 | 1.079 KILL | 1.137 MID |
| D-D1 | 1.033 KILL | 1.051 KILL |
| D-D2 | 1.071 KILL | 1.109 MID |
| D-D3 | 1.067 MID [1.028, 1.179] | 1.117 MID [1.108, 1.289] |
| D-D4 | 1.090 MID | 1.147 MID |
| D-M1 | 1.115 MID | 1.178 MID |
| D-M2 | 1.102 MID [0.971, 1.249] | 1.203 MID [1.032, 1.337] |

INTERPRETATION: with the measured E-59 prefill, no workload is GO even if the compact read saves all its bytes; the
GPU result lies between the two columns. R-101 B1 holds on the full grid.

## 15. Post-run check R-102 and the close (2026-11-10; D-117, we: "Close with bracket")
R-102 (reviewer): CONFIRM_WITH_CORRECTIONS. Scripts: a temporary folder `r102/`.
1. FACT: all 20 labels and medians recomputed from the 500 cells match Section 14; no cell is excluded.
2. FACT: single seeds pass 1.25. The largest per-seed median of a gate arm: r = 0.625, M 1.337, M_pf_equal 1.358,
   M_gemm 1.334 (D-M2 seed 1; D-D3 seed 3 1.289); r = 1.0, M 1.249, M_pf_equal 1.267, M_gemm 1.246. The 5-seed rule
   makes these workloads MID. So the claim holds for the 5-seed median, not for every seed.
3. FACT: M_pf_equal alone at r = 0.625 gives at most 1.226 (D-M2).
4. FACT (full-horizon reruns of D-M2, D-M1, D-D3 at r = 0.625): no hook cost 1.207 / 1.184 / 1.123; no hook and
   T_OTHER 20 ms 1.151 / 1.135 / 1.070 (a lower T_OTHER lowers the gain); no hook and ratio 0.5 (below the byte
   ratio) 1.245 / 1.203 / 1.143; no hook and zero prefill and recompute cost 1.401 / 1.279 / 1.349 (GO).
5. INTERPRETATION: the bracket is fair on the read time (0.625 is the byte ratio; the planted stock read runs at
   about 1.6 TB/s, as E-58; 40 ms is about the BF16 weight-read floor on an A100).
6. FACT (the weak point): the gain is cut from the capacity ratio (1.35 to 1.45 steps) to about 1.20 by the
   preemption recompute (D-M2, pool 8,724, seed 0: 9 stock preemptions, 148 compact). `poolsim.py` recomputes the
   whole context of a preempted request with no prefix-cache hit. vLLM enables prefix caching by default, and E-63 to
   E-65 model it off (the E-57 engine setting). So the conclusion depends on this assumption.
7. FACT: the prefill mapping is sound (42 stock points, 28 compact; residuals at most 1.9%). The fit covers contexts
   of 1K to 27K, batch 3, eager prefill, one request; the model extrapolates it linearly to 131K. The compact / stock
   prefill ratio (1.057) has no noise estimate (2 rows).

Close (D-117): E-65 is closed with the bracket. The GPU part did not run (six of six Secure creates failed for
capacity); the package stays ready to run.
Allowed wording (R-102): "In a CPU model of the vLLM scheduler, the Gemma 4 31B compact K=V cache did not reach a 1.25x
decode-throughput gain in any of 10 workloads (5-seed median). Prefill chunk times came from one A100 (E-59, eager,
2,048-token chunks, contexts to 28K, extrapolated linearly); decode read times were planted. If the compact read saves
all its bytes, the best median is 1.20x (decode node, M2), and some single seeds exceed 1.25x; with no read saving the
medians are 1.03 to 1.12x. The model recomputes preempted requests in full, with no prefix cache; with zero recompute
cost, three decode-node workloads exceed 1.25x. This is a model result, not a GPU measurement."
Do not say: "measured prefill" or "measured decode time" without these limits; "never reaches 1.25" or "in every
seed"; that the GPU result is bounded by this model; that the compact cache has no benefit (the capacity gain stays).
Open question (not decided): the prefix-cache hit of a re-admitted request under memory pressure.

## 16. Update from E-66 (2026-11-10)
E-66 answers the open question of Section 15: CLAIM-STANDS, confirmed by R-104 (`README_E66.md` Sections 10, 11). Section
15 stays as the record. In its allowed wording, replace the sentence "The model recomputes preempted requests in full,
with no prefix cache; with zero recompute cost, three decode-node workloads exceed 1.25x." with this text:
"In a model of the GPU-memory prefix cache, 0.75% of re-admitted preempted requests hit the cache (0.38% of discarded
tokens), and a block-level reference gave a hit rate of about 2.5% on 6 cells; with zero-cost recovery, three
decode-node workloads exceed 1.25x."

## Amendments

### A1 (2026-11-10, after R-101, before any GPU run; we: "GPU run too")
Changes (they replace the text of Sections 4 to 9 where they differ):
- B1: the free Mac bracket runs first (`make_bracket.py`, `run_bracket.sh`; results in `results/bracket/`): part 2
  with planted tables at read ratio 1.0 and 0.625, tied to E-58 (stock read) and the E-59 profile (sliding 5.6 ms of
  50 layers at the tie; T_OTHER 40.06 ms). Its labels are reported as the bracket of the GPU result; they decide
  nothing.
- B2: the sliding layers use the engine call: window (1,023, 0), seqused = L, a block table over the full context in
  which the entries before the window point at one shared block (`SlidingCache`). SL now depends on L, so the model
  interpolates SL like G (per L row).
- B3: L rows 1K, 4K, 16K, 32K, 64K, 128K with B x L <= 2^19 (33 core points with the tie), plus the extra points
  (16, 64K) and (8, 128K) (2^20 tokens; the 2D path at long context). Above 2^19 tokens the two layouts are built and
  timed one after the other (3 rounds each), so only one is resident.
- B4: the tie point runs first. On the main run, V4 is checked at once; if it fails, the tie runs once more; if it
  fails again, the bench stops (INVALID, little money spent). Then the core points by tokens, the extra points last.
  V5 fails only for a missing core point; a missing extra point is reported as "extrapolated".
- F1, F2 (the gate): a seed label is GO only if Q_M, Q_M_pf_equal and Q_M_gemm are all at least 1.25; KILL only if all
  three are below 1.10; else MID. The 5-seed rule stays.
- F3: SL and H have 3 rounds each; V3 also covers them (spread at most 5% where the median is at least 0.2 ms).
- F4: V1 passes if the outputs are identical, or the largest difference is at most 2% of the output scale; the scale
  must be above 0.
- F5: V6 is report only.
- F6: the point (1, 1K) is left out of the G rows (its caches fit in the 40 MB L2).
- F9: the rehearsal runs 8 points (`REH_POINTS`: the 3D and 2D paths, and the one-after-the-other path at (2, 64K)
  with a threshold of 2^16); AUTO_STOP_DELAY 90 s.
- Not changed: the question, the metric r_M, the thresholds 1.25 and 1.10, the 5-seed rule, the E-64 grid, V2, V4,
  the T_OTHER rule, the prefill fit, the budget caps.
- Freeze: no commit was made for the design, so the design is frozen by the SHA-256 of each file that runs, written
  before the rehearsal (`make_manifest_e65.py --working`; the list goes in Section 12).
- Verification: 27 tests pass (A1 adds the plan order, the sliding block table, the zero-output V1 failure, V3 on
  SL, the core and extra V5 rule); the stub session runs 9 points with no failed check.
