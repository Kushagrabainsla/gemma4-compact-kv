# E-57: the compact K=V cache built into vLLM: measured KV capacity, decode throughput and output identity (preregistration)

Status: PREDECLARED 2026-11-04; amended A1 2026-11-04 (before any run) and A2 2026-11-05 (after the S5 rehearsal,
before the main run; see Amendments). S2: R-82 AMEND_THEN_RUN, applied as A1. S3: 37 tests pass (Section 12). S4:
D-101 (we, "Yes, run both"). S5: the 12B rehearsal on one A40 passed (EQ EXACT). S6 (2026-11-05, one A100):
BUILD-PIVOT (Section 13). S9: R-83 CONFIRM_WITH_CORRECTIONS (Section 14).
Idea: I-182. Decision: D-100 (we, 2026-11-04, "Lets close the biggest gap": build the compact cache into vLLM and
measure the KV capacity, the throughput and the outputs against the stock cache; the Mac work first, one GPU session
after a new S4). Journal entry: Section 6.1 E-57 (to add at S2).
Labels: FACT (a source, a log or a computation was read), INTERPRETATION (reasoning), HYPOTHESIS (not tested).
All thresholds are project heuristics unless a source is given.

## 1. Question

In vLLM 0.33.0rc1.dev229+g3966924d5 on one A100 80GB, with the 16-bit Gemma 4 31B weights, does a compact K=V cache in
the 10 global layers (640 values per KV head and token, read by the stock Triton kernel through two strided views):
- compute exactly the arithmetic that E-56 validated (the same outputs as an emulation on the stock cache),
- keep the final answers of the stock engine (the E-56 margin), and
- hold more long requests in the same KV pool and decode more tokens per second there?

A "no" answer: the build outputs differ from the emulation (a bug), the accuracy falls, the engine footprint ratio is
below 1.15 at 64K tokens, or the decode rate at the pool limit does not rise.

## 2. Rubber duck

- Why a build now. The memory saving of I-182 is arithmetic, and no engine run measured it (R-81 C1, C2). E-56 tested
  the final answers with an emulation: hooks gave a stock cache the values that the compact row returns. A build
  answers two questions that the emulation cannot: does the engine allocate less memory per request, and is the
  build arithmetic the validated arithmetic.
- Why no new kernel. The views layout of E-52 reads the row with the stock kernel. The code map
  (`dossier/DESIGN_E-57_VLLM_CODE_MAP_2026-11-04.md`) shows that the stock Triton write op and the Triton attention
  kernel take separate K and V strides, so they accept the two views. FACT (`triton_unified_attention.py` 428 to 485 at
  3966924d5): each token finds its own block (`seq_offset // BLOCK_SIZE`), so a 48-token block works.
- Why the A100. On SM80, vLLM runs TRITON_ATTN in every layer for Gemma 4 (E-56 log). The FA4 path of the H100 reads
  the views in a microbenchmark only (E-52), and its compiled kernel is outside the vLLM tree (code map Section 6.2).
  The A100 is also the E-56 platform, so the weights, the image and the data are the same.
- The page plan. A compact page (5,120 B per token) does not divide the sliding page (16,384 B per token). Gate G1 found
  that the R-72 plan (a 1,024-token global block) is the weakest, and that B3 is better for BF16: the largest
  multiple of 16 tokens whose compact page fits the sliding page. For the 31B that is 48 tokens (245,760 B), and the
  stock page unifier pads it to 262,144 B (6.25% waste). FACT (code map Section 4): the patch is only the spec; the
  unifier, the allocator and the scheduler stay stock.
- The bit-for-bit twin. E-56's BF16C arm kept the natural dim order. The views layout permutes the dims (the K view is
  [rotated | unrotated]), so the dot products add their terms in a different order. So the build is compared with
  EMULV: the same hook transform as the build (`ViewsSplice` mode VIEWS) on the stock cache. INTERPRETATION (from the
  code; EQ tests it, A1 wording per R-82 N7): EMULV and COMPACT give the kernel the same q, k and v values in the same
  order, and FACT: the Triton kernel's tile sizes and segment counts do not depend on the block size (`triton_attn.py`
  96 to 200; `triton_unified_attention.py` 810 to 1022). So the outputs must be identical if both engines schedule
  the same batches. A Mac test checks this with a paged-cache emulation (Section 12).
- The same batches. The schedule depends on the token budget and on the free KV blocks. The output phase caps the engine
  at 16 sequences and sends the 7.7K-token needles in groups of 6, so the pool never limits a step (FACT,
  arithmetic: at most about 4,800 of 6,143 blocks). V8 checks the failed allocations and the preemptions.
- The capacity measurement. The engine log line "Maximum concurrency" uses a startup bound that keeps sliding blocks
  for 2 x 2,048 tokens in flight (`vllm_cfg.py` 616 to 627; vLLM PR #47728). FACT (Section 12): this bound reproduces
  the E-56 line (3.10x at 10,240 tokens). It understates the decode capacity, because a decoding request keeps about 65
  sliding blocks per group, not 321. So the capacity metric is the engine's own steady block count of one decoding
  request (C1, A1: the median over its last 32 decode steps), read from the block pool. The log lines are reported.
- The admission rule (A1, R-82 B2). FACT (`config/scheduler.py` 190; `kv_cache_manager.py` 529 to 547;
  `single_type.py` 215 to 227 at 3966924d5): `scheduler_reserve_full_isl` is on by default, so the scheduler admits a
  waiting request only if its full prefill reservation (the global blocks of the prompt plus at most 321 blocks per
  sliding group) fits the free blocks now. So b requests decode together only if b - 1 steady footprints and one
  prefill peak fit the pool. The prefill peak minus the steady footprint is about 1,275 blocks in both layouts, so the
  concurrency 1 + (free - peak) / steady has the same ratio as the steady footprints. The batch is its integer part.
- The integer effect. The pool holds only a few long requests (Section 3). The batch ratio is an integer ratio, so the
  throughput result depends on where the pool limit falls. The footprint ratio is the continuous metric; the throughput
  rows report the batches next to the ratio.
- The noise. E-56 DET (the same engine twice) was bit-identical. E-57 reruns 32 GSM8K items (DET). For the decode rate,
  engine A repeats its accepted batch once at each length (the A/A spread). The overhead line is the larger of 5% and
  3 times that spread.
- The controls. GROSS (the query without w: the unrotated scores grow about 16 times) must break the answers. It shows
  that the hooks reach the kernel. NAIVE (the 50% "K equals V" design: w V_rot in place of K_rot) is a realistic design
  error; its effect is reported, not decided (R-80 C3 asked for a control that can separate).
- Mode OFF with the compact cache would write k and v into overlapping views and corrupt the row. The run forbids it.
- What E-57 does not test: the FA4 path, FP8, prefix caching, CUDA graphs, tensor parallelism, the 26B-A4B model, the
  load-time folds (the build applies w and the output order at run time, as EMULV).

## 3. Hypotheses

- EQ (the build check): COMPACT gives the same output token ids, the same reference log-probabilities and the same probe
  log-probabilities as EMULV, bit for bit (prediction: EXACT).
- H1 (the E-56 replication): the accuracy change of COMPACT against STOCK over 432 items is inside +/-3 points (the
  paired Agresti-Min 95% interval).
- CAP: the ratio of the steady decode footprints (stock / compact) is 1.338 at 32,768 tokens and 1.403 at 65,536
  tokens (FACT, arithmetic for 65 sliding blocks per group; 1.337 and 1.401 for 66).
- TPUT (HYPOTHESIS, A1): the gate model (Section 2; `run_e57.StubEngine`) predicts the accepted batches 3 -> 4 at 32K
  and 2 -> 2 at 64K (concurrency 3.61 -> 4.83 and 2.05 -> 2.88). So T is about 1.3 at 32K and about 1.0 to 1.15 at 64K
  (the same batch; the compact step can read fewer KV bytes if the overlap of the two views stays in the L2 cache).
  Before A1 the prediction was 4 -> 5 and 2 -> 3 (1.2 and 1.5 times); it used the decode footprint for admission.
- Reported predictions (A1): the peak ratio about 1.15 (32K) and 1.23 (64K); the startup log ratio about 1.23 at
  66,048 tokens.
- OVH (HYPOTHESIS): at the same batch, the compact step time is within 5% of the stock step time.

## 4. Setup (frozen)

### 4.1 Hardware
- 1 x A100 80GB (SXM or PCIe), Secure, on demand, from a live catalog read at S4 (E-56: SXM in US-MD-1, $1.79/h).
- The rehearsal (S5): 1 x 48 GB GPU of the Ampere or Ada class (A40, RTX A6000 or L40S), from a live read at S4.

### 4.2 Software
- Image `vllm/vllm-openai@sha256:f255c4fa...` (the E-56 image; vLLM 0.33.0rc1.dev229+g3966924d5), Triton 3.8.0.
- Model `google/gemma-4-31B-it` at revision `842da3794eaa0b77d5f08bae87a17459d91ff475` (the E-56 checkpoint, 62.6 GB).
- Rehearsal model `google/gemma-4-12B-it` at revision `43c9d014b231474fa7212dbb48a762566c2b255d` (23.95 GB). FACT (its
  config): 8 global layers with K=V, 1 global KV head of 512, 8 sliding KV heads of 256, window 1,024; class
  `Gemma4UnifiedForConditionalGeneration`, which runs `Gemma4ForCausalLM` of `gemma4.py`, so the same attention code.
- Engine (both layouts): eager, in process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`), the V2 model runner, BF16 KV cache,
  no prefix caching, seed 0, `max_model_len` 66,048, `max_num_batched_tokens` 2,048, `max_num_seqs` 16,
  `kv_cache_memory_bytes` 15 GiB (the same pool in both engines: 6,144 blocks of 2,621,440 B), no image or audio.

### 4.3 Data
`experiments/gemma4_kv_tasks/data_e56.json` (SHA-256 `4f8511c89b77481ee5f2b39927de41c99b6da63680ccd8c5f6bcccd6431db672`,
unchanged): 400 GSM8K test problems and 32 needle prompts of 7.7K tokens in chat format, the GSM8K reference solutions
as scoring sequences, and the probe (the first 1,536 tokens of the health slice HB1). The capacity prompts are
synthetic token ids in [1,000, 100,000) from a fixed seed (`run_e57.cap_prompt`).

### 4.4 The build (`compact_engine.py`, installed before the engine starts)
1. The spec: `GPUModelRunner.get_kv_cache_spec` returns, for each `FullAttentionSpec` with head sizes 512 of a K=V
   model, the same spec with `state_content_bytes` 1,280 (640 BF16 values) and the B3 block (48 for the 31B, 96 for
   the 12B). The stock unifier pads the global page to the sliding page.
2. The views: wrappers of `triton_attn.unified_attention` and `triton_attn.triton_reshape_and_cache_flash` replace a
   128-value value cache (the split of a 640-value row at 512) with the V view: the K view with its storage offset
   plus 128 values. Stock rows pass through. Both engines install the wrappers.
3. The hooks: `ViewsSplice` on the 10 global `Attention` modules. VIEWS: q' = [q_rot | bf16(q_unrot x w)] (after
   RoPE; w is the per-dim k_norm weight), key' = [K_rot | V_unrot], value' = [V_unrot | V_rot], and the output back to
   the model order. NAIVE: key' = [bf16(w V_rot) | V_unrot]. GROSS: q' = [q_rot | q_unrot].
4. Counters: the V-view fixes and pass-throughs, the failed KV allocations (`KVCacheManager.allocate_slots` returns
   None) and the preemptions (`Scheduler._preempt_request`).

### 4.5 Arms
| Arm | Engine | Mode | Role |
|---|---|---|---|
| STOCK | A (stock cache) | OFF | the baseline |
| EMULV | A | VIEWS | the bit-for-bit twin of COMPACT on the stock cache |
| DET | A | OFF | determinism: STOCK again on the first 32 GSM8K items |
| COMPACT | B (compact cache) | VIEWS | the build |
| GROSS | B | GROSS | the liveness control (64 GSM8K, 12 needles) |
| NAIVE | B | NAIVE | the design-error control (reported) |

### 4.6 Workload and order
- Engine A: the warm-up (512 probe tokens, not scored); the pre-flight (OFF and VIEWS on the probe; exit 3 on a problem
  of V1, V2 or V4); STOCK; the health gate (V5; exit 3); EMULV; DET; the capacity phase.
- Engine B: the warm-up (VIEWS); the pre-flight (VIEWS, GROSS, NAIVE; exit 3 on a problem); COMPACT; the capacity phase
  with the equal-batch rows; GROSS and NAIVE while the time allows.
- An arm: the 400 scoring sequences (prompt log-probabilities), the 400 GSM8K answers (greedy, at most 640 tokens,
  stop at `<turn|>`), the 32 needle answers in groups of 6 (greedy, at most 128 tokens, with the prompt
  log-probabilities from token 1,025).

### 4.7 The capacity phase (at L = 32,768 and 65,536 tokens; as amended by A1)
- C1 (the footprint): one request of L - 64 prompt tokens and 64 decode tokens. The steady footprint is the median of
  the pool blocks in use over its last 32 decode steps (the free blocks of the block pool). The prefill peak is the
  largest number over all steps. (R-82 B1: the first decode steps still hold the last prefill chunk's sliding blocks.)
- C2 (the batch): rows of b requests of L - 256 prompt tokens (at most 512 decode tokens, no end token). A row is clean
  if no preemption happens and all b requests run and decode together for a window of 32 steps or more. The window
  starts at the fourth such step (the async lag, R-82 B3), holds at most 128 steps, and drops the steps above 5 times
  its median (counted). The decode rate is b / the mean step time of the window. A deferred admission (a failed
  allocation while the previous request frees its prefill blocks) is allowed and reported.
- The search: the first batch is b0 = 1 + floor((free blocks - prefill peak) / steady footprint), at most 16 (the gate
  model, Section 2). Search down to the first clean row. After the window of a clean row, one probe request is added:
  if the engine admits it within 4 steps, b + 1 runs next. Each batch runs once. The accepted batch is the largest
  clean batch; the limit shows as a refused probe or a failed next batch.
- A row ends early: at the first preemption, at a finish before the window, when a waiting request is refused for 16
  steps while every running request decodes, or after the window and the probe. Then the run aborts its requests.
- C2 repeat (engine A): the accepted batch once more in mode OFF (the A/A spread of the step time).
- EMULV row (engine A, R-82 F3): the accepted batch in mode VIEWS (the hook cost on the stock cache).
- C3 (engine B): the batch of engine A's accepted C2 row (the overhead at equal work). If engine B has a clean C2 row
  at that batch, that row is C3.
- The order (A2): C1 and C2 at both lengths first (the decision rows), then the A/A repeats (engine A) or C3 (engine
  B), then the EMULV rows (engine A).

## 5. Metrics
- EQ: the share of items with identical output ids, of identical score arrays, and the probe log-probabilities of VIEWS
  in both engines; the final-answer changes and the mean reference-NLL change of COMPACT against EMULV.
- H1: the accuracy change of COMPACT against STOCK over the 432 items, with the paired Agresti-Min interval (E-56 code).
- Reported: the final-answer changes and the text changes against STOCK; the reference-NLL change with the E-56 ratio
  interval; the same rows for EMULV and NAIVE; the GROSS and NAIVE accuracies; the long-context needle NLL.
- CAP: R(L) = steady footprint(stock) / steady footprint(compact) from C1 (the decision metric). Reported (A1, R-82
  F2): the predicted R; the prefill-peak ratio; the concurrency c = 1 + (free - peak) / steady of each engine and its
  ratio; the startup log line "Maximum concurrency" of each engine and its ratio (the admission bound).
- TPUT: T(L) = rate(compact) / rate(stock) at the accepted C2 batches, with the batches and how each limit showed.
- OVH: O(L) = step time(C3) / step time(stock C2), with the A/A spread. Reported (A1, R-82 F3): the hook cost H =
  EMULV row / stock C2 and the cache cost K = C3 / EMULV row. Only clean rows at the same batch count (R-82 N9).
- Reported (A1, R-82 F4): the schedule fingerprints of EMULV and COMPACT for each phase (scoring, GSM8K, needles).

## 6. Validity checks (a failed check makes the run INVALID; V9 is recorded)
- V1: engine A built the stock global spec (1,024-value rows); engine B applied the compact spec to every hooked layer,
  its global group has 1,280 content bytes, the B3 block and a page padded to the sliding page, and its global tensor
  is [B, H, block, 640]. No hook problem, and no KV-shared global layer (A1, R-82 F8; both configs have
  `num_kv_shared_layers` 0).
- V2: every arm of engine B fixed the V view in reads and writes; no arm of engine A did.
- V3: the two engines have the same pool blocks and the same bytes per block.
- V4: the pre-flight gaps (the engine attention output against a float32 reference on 16 query rows) are at most
  0.02 in every hooked layer for OFF and VIEWS; VIEWS does not separate from the stock reference (at most 0.02); no
  hook error; finite probe scores.
- V5: STOCK passes the E-56 health gate (GSM8K and needle accuracy 0.75 or more, reference NLL 3.0 or less).
- V6: STOCK, EMULV, DET and COMPACT finished; both engines have a steady footprint and an accepted (clean) C2 row at
  both lengths; both runs ended with the status ok.
- V7: if GROSS ran, its GSM8K accuracy is at least 20 points below COMPACT on the same items, or its needle accuracy at
  least 25 points below.
- V8: no failed allocation and no preemption in STOCK, EMULV, DET and COMPACT.
- V9 (recorded): DET equals STOCK bit for bit. If not, EXACT is not expected, and EQ is read at the NEAR level. (A2:
  DET runs the first 32 GSM8K items only, so its batches change in composition when its queue empties, and greedy
  outputs can change at near ties. So the output-id comparison mixes the batch composition with run-to-run noise. S5
  gave ids 30 of 32 and score arrays 96 of 96 while EQ was EXACT. V9 stays recorded; it is not a failed check.)

## 7. Decision rule (frozen; project heuristics)
- EQ: EXACT if all output ids, all score arrays and the probe scores are identical; else NEAR if at most 3 final answers
  differ and the 95% interval of the mean reference-NLL change is inside +/-0.002 nats per token (A1, R-82 F1: the
  E-56 GO3 margin; the E-56 benign reorder BF16C against BF16 gave 2 final changes and -9.1e-4 [-1.56e-3, -2.5e-4]);
  else FAIL.
- H1: GO1 if the interval is inside [-3, +3] points; KILL1 if its upper end is below 0; else INCONCLUSIVE1. (R-82 F9
  noted that KILL1 is a significance rule; it stays, because H1 replicates the E-56 rule.)
- CAP: GO-CAP if R(32K) >= 1.25 and R(64K) >= 1.30; KILL-CAP if R(64K) < 1.15; else PARTIAL-CAP.
- TPUT: TPUT-GAIN if T >= 1.15 at one length and T >= 0.97 at both; TPUT-NONE if T < 1.05 at both; else TPUT-PARTIAL.
- OVH: OVH-OK if O is at most max(1.05, 1 + 3 x |A/A spread|) at both lengths; else OVH-HIGH; NA-OVH if a clean C3 or
  stock row is missing.
- The decision (A1, R-82 F9: one word for each case; the first rule that applies decides):
  - REHEARSAL: a rehearsal run.
  - INVALID: a check of V1 to V8 fails. Fix and rerun with a new approval.
  - BUILD-FAIL (a bug or a schedule difference, not a decision about the idea): EQ FAIL. The schedule fingerprints and
    the score arrays tell which.
  - BUILD-KILL (KILL): KILL1 or KILL-CAP. The engine build does not deliver the saving; the thesis reports the negative
    result.
  - BUILD-GO (GO): GO1, GO-CAP, TPUT-GAIN or TPUT-PARTIAL, and OVH-OK (EQ is EXACT or NEAR here). The thesis reports
    the measured capacity, the throughput rows and the build.
  - BUILD-PIVOT (PIVOT): every other case (for example PARTIAL-CAP, INCONCLUSIVE1, TPUT-NONE, OVH-HIGH or NA-OVH).
    Next: the page plan P4 or the load-time folds, then a new test.
- `analyze_e57.py` prints one line: `E-57: <word> | EQ | H1 | CAP | TPUT | OVH | failed checks`.

## 8. Controls and baselines
- The baseline: STOCK, the unchanged engine on the same Pod, the same weights and the same pool.
- The decomposition: STOCK to EMULV changes the arithmetic only; EMULV to COMPACT changes the cache only.
- A/A: DET (outputs), the C2 repeat (the decode rate).
- Positive controls: GROSS (sure break; V7), NAIVE (a realistic design error; reported).
- The E-56 link: COMPACT against STOCK repeats the E-56 BF16C against BF16 comparison in the views order.

## 9. Budget
- Main run: 1 x A100 80GB at the live price (E-56: $1.79/h). Cost cap $2.50. Caps: the boot watchdog at 3,780 s
  (`pod_cmd.py --cap 3780`), the session watchdog at 3,600 s, AUTO_STOP 300 s after the session, an SSH limit of 600 s.
  Worst case: 10 min to SSH plus the boot watchdog: 73 min, $2.18.
- Time per step (estimates, INTERPRETATION; A1 per R-82 F6, A2 from S5 and the E-56 step times): create to SSH 4 to
  12 min; setup and the 62.6 GB download about 4 min (E-56: 220 s); each engine start about 1 min; each output arm
  about 6.3 min (the scoring 0.4, GSM8K about 4.1 at 16 sequences, from the E-56 82 s at 64 sequences, the needles
  about 1.8); DET 0.4 min; the capacity phase about 9 min (A: C1, C2, the repeat and the EMULV row at two lengths; a
  32K row about 1 min and a 64K row about 1.6 min) and about 5 min (B); GROSS 2.4 min; NAIVE 6.3 min; analysis 1 min.
  Session about 43 min without NAIVE and 49 min with it (the deadline gives about 57 min); Pod about 55 min, about
  $1.65 at $1.79/h.
- The session deadline (A1, R-82 F5) is the earlier of 60 min after the session start and 240 s before the boot
  watchdog stops the Pod (its "armed cap" line in `/workspace/boot_watchdog.log`). Engine A gets 52% of the time after
  the setup (45% before A1).
- What the time cap cuts first: NAIVE, then GROSS (then V7 is not run), then the EMULV row, the A/A repeat and C3
  (reported rows). The decision arms, C1 and the accepted C2 rows must finish, or the result is INVALID.
- S5 rehearsal: the 12B on a 48 GB GPU at the live price, REH=1 (48 GSM8K items, 8 needles, L = 8,192 and 16,384,
  a 12 GiB pool; the gates record and do not stop). Cost cap $0.75, boot watchdog 1,800 s. VERIFY: (1) both engines
  start; engine B has the compact spec (block 96, 1,280 content bytes, the padded page) and engine A the stock spec;
  (2) the V-view fix counters (B > 0, A = 0); (3) equal pools; (4) the pre-flight gaps for every hooked layer; (5) EQ
  EXACT on the 12B; (6) the footprints, the accepted batches, the windows of 32 steps or more, the A/A repeat; (7)
  GROSS breaks the answers; (8) the time per arm and per capacity row; (9) the session files, the code hashes and the
  REHEARSAL line; (10, A1) `limit_mm_per_prompt` 0 works for `Gemma4UnifiedForConditionalGeneration` (R-82 N8); (11,
  A1) the capacity rows end as designed (the steady footprint, the probe, the end reasons, the drain to idle) and the
  session deadline line names the boot watchdog end.
- Balance floor: the protocol rule is the cap plus $35. If the live balance is lower, we decide (as D-95, D-98).

## 10. Threats to validity
1. The 31B on one A100 leaves a 15 GiB pool, so the batches are small (2 to 6) and the batch ratio is an integer
   ratio. The footprint ratio carries the capacity claim; the throughput rows are specific to this pool.
2. A larger GPU, FP8 weights or tensor parallelism change the pool and the batches; the footprint ratio stays the
   same for the same block plan (arithmetic).
3. Prefix caching is off. With it on, B3 aligns the hits to 48 tokens (gate G1); not tested.
4. The build applies w and the output order at run time. The load-time folds round differently in BF16 (R-81 P3) and
   need their own check.
5. The synthetic capacity prompts do not change the footprint or the step time of a decode step (the attention reads
   every cached token whatever the token ids are).
6. One Pod, one session: no clock control. The A/A repeat bounds the step-time noise.
7. O-19: the instruct model scores raw text poorly in vLLM; the output check uses chat-format tasks, as E-56.
8. The Triton kernel only. The FA4 path reads the views in a microbenchmark only (E-52).
9. The admission rule (A1). In a 15 GiB pool the prefill reservation of one long request (321 sliding blocks per
   group) is large next to the decode footprint, so the batches are small and the batch ratio rounds: at 64K the
   predicted concurrency is 2.05 (stock) and 2.88 (compact), and both batches are 2. A larger pool, or a smaller
   prefill reservation (fewer tokens in flight), moves the batch ratio toward the footprint ratio.
10. The decode path has no attention-level check (R-82 F10, accepted). EQ and H1 test it end to end. If EQ is not
    EXACT, identical score arrays with different output ids place the fault in the decode path, and the schedule
    fingerprints tell a schedule difference.
11. H1 is a replication on the same 432 items (R-82 N6): the paired interval covers the item sampling, not the
    run-to-run noise.
12. The KV memory headroom is 5.9 GiB against 6.3 GiB in E-56 (R-82 F7). FACT (the E-56 telemetry): the peak use was
    77,505 MiB. With 79.25 GiB usable (the E-56 vLLM log), about 3.1 GiB stay free with the 0.42 GiB of extra KV. No
    change.

## 11. Files
| File | Role |
|---|---|
| `compact_layout.py` | the row layout, the transforms, the B3 block and the block arithmetic |
| `compact_engine.py` | the vLLM patches, the hooks, the engine records |
| `run_e57.py` | one engine process (`--layout stock` or `compact`; `--stub` on the Mac) |
| `analyze_e57.py` | the frozen analysis |
| `session_e57.sh` | the Pod session (`STUB=1` on the Mac, `REH=1` for the rehearsal) |
| `test_compact_layout.py`, `test_compact_engine.py`, `test_e57.py` | the S3 tests |
| reused, unchanged | `../gemma4_kv_tasks/analyze_e56.py`, `../gemma4_kv_tasks/data_e56.json`, `../gemma4_kv_model/splice_e55.py`, `run_e55.py`, `analyze_e55.py` |

## 12. Verification record (stage S3)
- Tests: `uv run --no-project --with numpy --with torch --with pytest python -B -m pytest -q -p no:cacheprovider
  experiments/gemma4_kv_engine/` gives 37 passed (2026-11-04, after A1; 28 before A1).
  - The layout (7): the float64 exactness, the two-view write, the o_proj fold, the strides, the BF16 match with the
    E-56 emulation scores, the B3 block and the memory table.
  - The engine patch (13): the B3 spec for the 31B (48), the 26B-A4B (48) and the 12B (96); other specs unchanged;
    the V-view fix only on compact rows; a paged-cache emulation in which COMPACT (block 48, padded page) equals EMULV
    (block 32, stock row) bit for bit, both within BF16 error of a float64 reference; GROSS and NAIVE change the output;
    the device-index transforms equal the layout functions; the hooks on a stub module (gaps, counts, removal); the
    capacity arithmetic reproduces the E-56 log line (3.10x); A1: the hook transform uses no Python-list index and
    equals the layout functions (B4); `register` reports a KV-shared global layer (F8); the schedule digest (F4).
  - The pipeline (17, the stub engines with the real hooks on stub modules for the pre-flight, L-26, and the capacity
    phase on `StubEngine`, the gate model): a clean run gives BUILD-GO with EQ EXACT, GO1, GO-CAP (R 1.338 and 1.404),
    TPUT-GAIN, OVH-OK, matching schedule fingerprints and the batches 3 -> 4 (32K) and 2 -> 2 (64K), each limit shown
    by a refused probe; planted faults give BUILD-FAIL (a compact that differs), EQ NEAR with BUILD-GO (a tiny float
    difference, F1), INVALID (no compact spec; no V-view fix, exit 3; an unhealthy baseline, exit 3; a probe too short
    for the gap check, exit 3), BUILD-PIVOT with TPUT-NONE and NA-OVH (a preempting compact engine holds 1 request), a
    reported NAIVE failure that does not stop the run, OVH-HIGH (a slow compact path) and the reduced rehearsal
    workload; unit tests of C1 (the prefill transient, B1), the window (the skip and the trim, B3), the batch search
    and the start batch (B2), and the stub scheduler (a refused admission ends a row; the pool drains to idle).
- The session script in stub mode on the Mac (`STUB=1`, A1 with a boot watchdog log 3,000 s old): the deadline line
  names the boot watchdog end (531 s left); both engines and the analysis end with the expected files and the line
  "E-57: BUILD-GO | EQ EXACT | GO1 ... | GO-CAP R32K 1.338 R64K 1.404 | TPUT-GAIN T32K 1.341 (3->4) T64K 1.018 (2->2)
  | OVH-OK O32K 0.987 O64K 0.983 | failed checks: none".

## 13. Results (S6, 2026-11-05)

Run: D-101; one A100 SXM 80GB (US-KS-2, Pod <pod-id>, $1.78; the A40 rehearsal $0.29; E-57 total $2.07);
design commit 8752f58; results in `results/e57-a100-20261105/`. Both engines exit 0. GROSS, NAIVE and the 64K EMULV row
were cut for time (Section 9 order), because engine B read the weights from the Pod's FUSE volume in 558 s (engine A:
49.5 s). The frozen analysis prints:

`E-57: BUILD-PIVOT | EQ EXACT | GO1 change +0.0046 [-0.0032, +0.0124] | GO-CAP R32K 1.338 R64K 1.404 | TPUT-PARTIAL
T32K 1.243 (3->4) T64K 0.959 (2->2) | OVH-HIGH O32K 1.087 O64K 1.043 | failed checks: none | V9 recorded: same`

| Metric | 32K tokens | 64K tokens |
|---|---|---|
| Steady footprint, stock / compact (blocks) | 1,349 / 1,008 (R 1.338) | 2,373 / 1,690 (R 1.404) |
| Prefill peak, stock / compact (blocks) | 2,602 / 2,262 (1.150) | 3,626 / 2,944 (1.232) |
| KV per request saved | 0.83 GiB | 1.67 GiB |
| Accepted batch, stock / compact | 3 / 4 (each limit: a refused probe) | 2 / 2 |
| Blocks held in the window | 4,033 (3 stock) / 4,019 (4 compact) | 4,737 / 3,375 |
| Decode rate, stock / compact (tokens/s) | 56.0 / 69.7 (T 1.243) | 36.3 / 34.8 (T 0.959) |
| Equal-batch step ratio O, mean / median | 1.087 / 1.079 | 1.043 / 1.021 |
| Hook cost H (EMULV row / STOCK), cache cost K (C3 / EMULV row) | 1.048, 1.037 | not split (row cut) |
| A/A spread (back to back, one process) | -0.06% | -0.002% |

Other FACTs: EQ EXACT (ids 432 of 432, arrays 1,264 of 1,264, the probe, the schedule fingerprints of 3 phases);
accuracy STOCK 0.9722, COMPACT and EMULV 0.9769 (2 final answers change, both toward correct); the startup log lines
1.67x and 2.06x at 66,048 tokens (ratio 1.23); V9: DET equals STOCK; the GPU peak memory 77,783 MiB.

## 14. Post-run review (R-83, 2026-11-05) and the corrected reading

R-83 (`dossier/REVIEW_E-57_POSTRUN_2026-11-05.md`): CONFIRM_WITH_CORRECTIONS. It reran the frozen
analysis (the same line), checked the code hashes, compared the outputs byte by byte and recomputed the ratios.
Corrections (applied here, in the journal and in RESEARCH_OUTPUTS.md):
1. The A/A spread is a back-to-back figure in one process. The row-to-row noise is about 1% to 2% (INTERPRETATION),
   and no row measures the noise between the two engine processes.
2. OVH by mean and median: O32K 1.087 and 1.079 (above the 1.05 line by both and by a 2% shift); O64K 1.043 and 1.021
   (inside the noise). OVH-HIGH comes from 32K only.
3. The step times are bimodal: 20% to 32% of the window steps are about 5 ms slower than the median (5% in the stock
   64K repeat); the compact rows have a slightly heavier tail. This is the mean-median gap.
4. The integer sensitivity: b requests decode together if (b - 1) steady footprints and one prefill peak fit. The fifth
   compact request at 32K lacks 151 blocks and the third at 64K lacks 181 blocks (about 0.44 GiB). So T is a property
   of this 15 GiB pool; a pool of about 15.45 GiB would likely give batches 5 and 3 (HYPOTHESIS).
5. GO1 and the batch shape: E-56 BF16 (batch cap 64) scored 0.9769 and the E-57 STOCK (batch cap 16) 0.9722 with the
   same arithmetic, so a swing of 2 items comes from the batch shape. GO1 means no accuracy loss beyond +/-3 points; it
   does not mean that the compact cache is more accurate.
6. Code note (no effect on any result): in `analyze_e57.py` the V4 check of the VIEWS separation sits after the mode
   loop and reads the last mode; for both engines that mode is VIEWS, so the check is right. The frozen code stays.
7. The prediction record: R-82 predicted the footprints, the peaks (2,602 and 3,626), the log ratio and EQ EXACT; it
   predicted the batches 2 / 3 and 1 / 2, one low in three cases. The A1 gate model predicted all four batches.

The corrected reading:
- FACT: the build is exact (EQ EXACT against its bit-for-bit twin) and keeps the accuracy within +/-3 points. The
  steady decode footprint falls 1.338 times (32K) and 1.404 times (64K), equal to the arithmetic. The global KV bytes
  per token fall 1.5 times (8,192 B to 5,461 B with the 6.25% padding), not 1.6 times.
- INTERPRETATION: the saving is a decode-state saving. The admission gain is smaller (the peak ratios 1.15 and 1.23,
  the log ratio 1.23). The throughput gain at 32K comes from one more request in the batch, not from a faster step.
  BUILD-PIVOT means "the saving is real, and a step cost of 4% to 9% remains in this run-time build".
- HYPOTHESIS: the cause of the 3.7% cache cost (the 48-token block, the two overlapping views, the CPU work of the
  view wrappers, or a process effect); the cause of the 4.8% hook cost (about 100 small eager kernels per step);
  the effect of a larger pool; the effect of the load-time folds.
- Do not claim: "37.5% smaller KV cache" or "1.6x"; "1.34x to 1.40x more requests admitted"; "the compact cache is
  24% faster"; "no step overhead at 64K"; "the block size causes the overhead"; "EXACT equals correct" (the accuracy
  reference is E-56).
- Note after R-84 (2026-11-05, the E-58 pre-run review, F5): FACT (the source at 3966924d5): the Triton metadata
  builder sizes the 3D decode scratch with `model_config.get_head_size()` (256 for Gemma 4 31B), while the kernel
  indexes it with the layer head size (512 at the global layers). So decode batches of 5 to 8 requests in the 3D path
  write up to 4 MiB past that buffer at the global layers. The output phase of E-57 had such batches (the needle groups
  of 6) in every arm, so EQ, H1 and the health gate compared runs of the same code; what the overflow overwrote is
  unknown. The capacity rows (batches of 2 to 4) are not affected. vLLM main fixed the sizing on 2026-11-03 (PR
  #56996). The E-58 check G tests the overflow at runtime.
  RETRACTED (2026-11-05, E-59 README Section 13.6; R-87 C9; journal Section 8.1 row 77): the pinned engine sizes the 3D
  scratch for head 512 with Gemma 4 (every builder of the 31B and 12B engines reports `headdim` 512 and a 512-wide
  scratch before any change), because `get_head_size()` gives the largest head size of the per-layer architecture
  config. The overflow does not happen in the real engine. E-57 recorded no scratch, so its absence in E-57 is inferred
  from the same build; the effect of PR #56996 on Gemma 4 is not known.
- The next step on the PIVOT path (R-83 Section 8, a recommendation, not a decision): first a weight-free kernel
  microbenchmark on one A100 (about 15 min, under $0.60: block 32 against 48, the views against the stock row, the
  wrapper cost, the hook transform in eager, CUDA graph and fused form); then the load-time folds with a 32K-only
  engine run, an interleaved repeat for the process noise, a pool row of about 15.5 GiB, and
  `--safetensors-load-strategy=prefetch`.

## Amendments

### A1 (2026-11-04, before any GPU run; R-82 AMEND_THEN_RUN)
R-82 (`dossier/REVIEW_E-57_PRERUN_2026-11-04.md`) found four blockers in the capacity phase and ten
findings. No data existed. R-82 predicted that the design as frozen in 9c1de38 would print a false BUILD-PARTIAL. The
author checked B1 to B4 against the vLLM source at 3966924d5 and confirmed them. The changes:
- B1 (C1): the steady footprint is the median over the last 32 decode steps (before: the maximum over all decode
  steps, which holds the last prefill chunk's sliding blocks; R-82 predicted R 1.150 and 1.232 from it). The prefill
  peak is reported.
- B2 (C2): the search starts at the gate-model batch, goes down to a clean row and up while a probe request is
  admitted (before: free / footprint, down only). A clean row allows a deferred admission (before: no failed
  allocation). A row ends early and aborts its requests. FACT (Section 2): `scheduler_reserve_full_isl` admits a
  request only if its full prefill reservation fits.
- B3: the window starts at the fourth full step and drops the steps above 5 times its median (before: the second full
  step, no trim).
- B4: the hook transform uses the w slices as device tensors made once per layer (before: a Python-list index on each
  call, two stream syncs per global layer per step in EMULV and COMPACT only).
- F1: EQ NEAR uses the E-56 margins: at most 3 final answers, the NLL interval inside +/-0.002 (before: 2 answers,
  |mean| at most 5e-4).
- F2: reported ratios: the prefill peak, the concurrency, the startup log line.
- F3: an EMULV row in engine A at the accepted batch (the hook cost and the cache cost).
- F4: the schedule fingerprints of each output phase.
- F5: the session deadline ends 240 s before the boot watchdog; engine A gets 52% of the time (before: 45%).
- F6: the time estimates of Section 9.
- F7: no change (Section 10 item 12).
- F8: V1 reports a KV-shared global layer (both configs: `num_kv_shared_layers` 0, read 2026-11-04).
- F9: the decision words BUILD-GO, BUILD-PIVOT and BUILD-KILL, in a fixed order; GO also needs TPUT-GAIN or
  TPUT-PARTIAL and OVH-OK (before: BUILD-GO ignored TPUT and OVH, and PIVOT overlapped GO). H1 keeps the E-56 rule.
- F10: accepted, no change (Section 10 item 10).
- N2, N7: wording. With prefix caching off the engine uses `KVCacheCoordinatorNoPrefixCache`, not the hybrid
  coordinator that the code map names (`dossier/DESIGN_E-57_VLLM_CODE_MAP_2026-11-04.md` Section 4.2 and Section 8
  item 7).
- N8: rehearsal VERIFY item 10. N9: OVH uses clean rows only.
- The predictions of Section 3 (TPUT and the reported ratios) changed before any data; the CAP prediction did not.
- Verification: 37 Mac tests pass (Section 12); the stub session ends with "E-57: BUILD-GO | EQ EXACT | GO1 ...".

### A2 (2026-11-05, after the S5 rehearsal, before the main run)
The S5 rehearsal (D-101; Gemma 4 12B on one A40, EU-SE-1, commit f25b2d4; results in
`results/e57-rehearsal-a40-20261105/`) passed every VERIFY item: both engines exit 0; the compact spec (block 96, 1,280
content bytes, the page padded to 131,072 B) and the stock spec; the V-view fix counters; equal pools (12,288 blocks);
the pre-flight gaps of the 8 layers at most 0.0018; EQ EXACT (ids 56 of 56, arrays 160 of 160, the probe, the schedule
fingerprints of the 3 phases); the steady footprints 453 and 411 (8K) and 581 and 496 (16K), the predicted ratios
1.102 and 1.171; the first decode step held the prefill peak (1,707 blocks against 453 steady at 8K), which confirms
R-82 B1; windows of 128 steps, no drop, the A/A spread 0.0000 and 0.0001; OVH 0.987 and 0.969 (the compact step is not
slower); GROSS breaks the answers; the deadline line names the boot watchdog end; the pool drains to idle after
each row. The rehearsal gives no decision (the 12B at 8K and 16K: the batches are capped at 16 sequences). The
changes, none of which touches a decision rule:
- The capacity phase runs C1 and C2 at both lengths first, then the OVH inputs, then the EMULV rows. Reason: S5 cut
  the 16K EMULV row for time, and the A100 arms take about 6.3 min (Section 9), so the decision rows go first.
- The limit label: a batch at `max_num_seqs` shows "max_num_seqs" (S5 showed "probe refused", because the engine
  refused the probe at the sequence cap, not for the pool).
- V9 (Section 6): the line prints "V9 recorded" and keeps V9 out of "failed checks". S5: ids 30 of 32, arrays 96 of 96
  (the DET subset changes the batch composition).
- Section 9: the time estimates (S5 and the E-56 step times).
- Verification: 37 Mac tests pass; the amended analysis on the copied S5 results prints "E-57: REHEARSAL | EQ EXACT |
  ... | failed checks: none | V9 recorded: differs".
