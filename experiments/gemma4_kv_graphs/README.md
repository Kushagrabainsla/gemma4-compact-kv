# E-59: the compact K=V build with CUDA graphs, and the process effect (preregistration)

Status: PREDECLARED 2026-11-05 (design commit da838ed); amended A1 2026-11-05 (R-86 AMEND_THEN_RUN, before any run)
and A2 2026-11-05 (after S5 rehearsal 1, before the main run; see Amendments). NOT RUN on the A100. S4: D-105 (the
user). S5: rehearsal 1 stopped at the G1 pre-flight (Section 12.1); A2 fixed it; rehearsal 2 passed every VERIFY
item (Section 12.2); A3 (after rehearsal 2): E3 mirrors E1. 11 tests pass. S6 RUN 2026-11-05 on one A100 SXM: INVALID
(V1; Section 13); exploratory: OG -0.41%, inside the 2% band. R-87 CONFIRM_WITH_CORRECTIONS (Section 14).
Idea: I-182. Journal entry: Section 6.1 E-59. Decision: D-104 (we, 2026-11-05: "Plan E-59"). Origin: E-58 README
Section 14 item 8 (R-85 next step) and the author's recommendation to answer both open questions in one run.
Labels: FACT (a source, a log or a computation was read), INTERPRETATION (reasoning), HYPOTHESIS (not tested).
Thresholds are project heuristics unless a source is given.

## 1. Question
In vLLM 0.33.0rc1.dev229+g3966924d5 on one A100 SXM 80GB with the 16-bit Gemma 4 31B, at 3 requests of 32K tokens,
is the decode step of the run-time compact K=V build (the E-57 build) slower than the stock decode step by 5% or more
when both decode with full CUDA graphs? "No" (A1, R-86 F3: OG below the noise band B_g) means that the E-57 step
overhead does not survive CUDA graphs. "Yes" (5% or more) kills the run-time route under CUDA graphs. A cost between
the band and 5% is PIVOT.

Secondary questions (reported, not decided): (a) does the eager build reproduce the E-57 hook cost (4.8%) and cache
cost (3.7%) in this Pod; (b) how large is the process effect (the same stock engine in two processes), against the
E-57 cache cost, which was the only E-57 comparison across two processes (R-85 C5); (c) where the time of one decode
step goes in each arm (a profile); (d) the decode rate of 4 compact requests against 3 stock requests with CUDA
graphs (the capacity gain without the step overhead).

## 2. Rubber duck
- What E-57 found (FACT): built into vLLM in eager mode, the compact cache lets 4 requests of 32K tokens decode where
  3 stock requests decode (a 15 GiB pool), but at the same batch the step is 8.7% slower (the hooks 4.8%, the cache
  path 3.7%).
- What E-58 found (FACT, R-85): the compact attention layer is not slower; the 110 small hook kernels cost 3.1 ms of
  CPU issue time and 0.4 ms of GPU time in a CUDA graph per step; E-58 reproduces about one quarter (1.2 of 4.6 ms) of
  the E-57 overhead. The rest is not located. The first candidate is the process effect.
- Why CUDA graphs (INTERPRETATION): a full decode graph replays the recorded kernels with one launch, so the CPU issue
  time of the hooks (and of the whole eager step) leaves the step. If the rest of the E-57 overhead is also CPU or
  process noise, the build has no step cost that matters. The production default of vLLM uses CUDA graphs, so the
  eager number of E-57 is not the number a user sees.
- Why the process repeat: E-57 measured the cache cost across two engine processes (COMPACT in engine B against
  EMULV in engine A) but measured its A/A spread inside one process (R-85 C5). Two processes of the same stock
  engine measure the process effect directly.
- What would kill it early: the hooks cannot be captured (V1 fails in the rehearsal), or the build stays 5% slower
  with graphs.
- The trap that the design must avoid (FACT, the pinned source `vllm/v1/attention/backends/triton_attn.py`): with
  CUDA graphs on, the Triton builder sets `seq_threshold_3D` to the capture size nearest to 8 and sizes the 3D decode
  scratch with that many rows at head 256. The Gemma 4 global layers index the scratch with head 512, so a batch fits
  only if 2 x batch <= rows. Capture sizes [1, 2, 3, 4] would give 4 rows, and batch 3 would write past the buffer.
  The design captures [1, 2, 3, 4, 8] (8 rows; batches up to 4 fit) and checks the rows at run time (V3).
- The second trap (INTERPRETATION, the E-57 code): the E-57 hooks are Python module hooks that attach after the engine
  starts. A CUDA graph records only what runs during the capture, so hooks that attach later never run in a decode
  step, with no error. E-59 attaches the hooks right after the model loads (before the profile run and the capture),
  in every engine, and V1 checks that the compact engine gives the same tokens as the emulated engine: the compact
  cache gives wrong outputs without the hooks, so equal tokens show that the hooks ran inside the graphs.
- Order of work: the Mac code and tests, the different-model review, a rehearsal on one A40 with Gemma 4 12B, then
  the A100 run. The decisive graph block runs first in the session; the eager block (secondary) runs after it.

## 3. Hypotheses (written before any run)
- H1 (primary): with full decode graphs, the compact build is less than 2% slower than stock at batch 3 and 32K (OG
  below the band): the E-58 graph numbers predict about +0.9% for the hooks and about -0.9% for the cache layer.
- H0: the build stays 5% or more slower with graphs (a GPU-side cost that E-58 did not see, for example in the
  48-token block tables or the KV manager).
- Secondary predictions: the eager block reproduces a hook cost of 3% to 6% (E-57: 4.8%); the eager cache cost is 0%
  to 4%, and the process effect is 1% to 3%; graphs make the stock step 5% to 15% faster; 4 compact requests decode at
  1.28 to 1.36 times the rate of 3 stock requests (E-57 eager: 1.243). Predicted word: GO with HOOK-FIXED (about 60%),
  PIVOT (about 20%), INVALID by V5 (about 12%), KILL (about 8%).

## 4. Setup (frozen)

### 4.1 Hardware
- Main run: 1 x A100 SXM 80GB, Secure Cloud, on-demand, minCudaVersion 13.0 (the E-56, E-57 and E-58 GPU type). A
  container disk of 120 GB holds the weights (L-32: the Pod volume is a network file system, and a second engine read
  the weights from it in 558 s in E-57); a Pod volume of 20 GB holds the code, the caches and the results.
- Rehearsal (S5): 1 x A40 (48 GB), Secure, with Gemma 4 12B (as the E-57 rehearsal), a container disk of 60 GB.

### 4.2 Software
- Image `vllm/vllm-openai@sha256:f255c4fad2f30ff5ee80db14d3a03afe9ae0f10d3470339864ff04745fe44cb3` (vLLM
  0.33.0rc1.dev229+g3966924d5, torch 2.14.0+cu130), the image of E-49 to E-58.
- Model: `google/gemma-4-31B-it` revision 842da3794eaa0b77d5f08bae87a17459d91ff475 (BF16; E-56, E-57). Rehearsal:
  `google/gemma-4-12B-it` revision 43c9d014b231474fa7212dbb48a762566c2b255d.
- The E-57 build code, unchanged: `compact_engine.py` (the B3 spec, the V-view wrappers, the VIEWS hooks),
  `compact_layout.py`; from `run_e57.py` the prompts (`cap_prompt`), the engine view (`VllmEngine`) and the stub
  scheduler (`StubEngine`); from E-55 `splice_e55.py` and `run_e55.py`.

### 4.3 Engine settings (all engines)
- As E-57: TRITON_ATTN, the V2 model runner, in process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`), a BF16 KV cache, no
  prefix caching, `max_model_len` 66,048, `max_num_batched_tokens` 2,048, `max_num_seqs` 16, a fixed KV pool of 15 GiB
  (`kv_cache_memory_bytes`), seed 0, no multimodal inputs.
- The one factor of the graph block: eager engines use `enforce_eager=True` (E-57). Graph engines use
  `enforce_eager=False` with `CompilationConfig(mode=NONE, cudagraph_mode=FULL_DECODE_ONLY, cudagraph_capture_sizes=[1,
  2, 3, 4, 8])`: no torch.compile (FACT, the pinned `vllm/config/vllm.py`: FULL_DECODE_ONLY needs no piecewise
  compile), full graphs for decode-only steps, eager prefill and mixed steps. Batches 3 and 4 have their own graphs
  (no padding).
- A1 (R-86 B1): in every engine, a wrapper of `TritonAttentionMetadataBuilder.__init__` makes the Triton 3D decode
  scratch 512 wide in its last dimension (the stock build makes it 256 wide, the model head size). FACT (the pinned
  `triton_unified_attention.py`): the store and the reduction compute the scratch offsets from the layer's
  HEAD_SIZE_PADDED, so the arithmetic does not change; the writes of the global layers (head 512) stop going past the
  buffer. Without it, the capture warm-up of size 8 runs 8 one-token requests through the 3D path and writes past the
  buffer, before any check can run.
- The hooks attach in every engine right after the model loads (a wrapper of `GPUModelRunner.load_model`), with the
  capture mode of the engine (OFF for stock engines, VIEWS for the emulated and compact engines). Eager engines can
  change the mode between rows, as E-57 did; graph engines cannot (V2 checks that their rows use the capture mode).

### 4.4 Workload (the E-57 C2 rows)
- One row: `batch` requests of 32,768 - 256 prompt tokens each (`run_e57.cap_prompt(32768, i + 1, 256)`: synthetic
  token ids in [1,000, 100,000), distinct per request, the same in every arm), greedy, `ignore_eos`, at most 512
  output tokens. The row steps the engine until all `batch` requests decode, skips 3 full steps (the async lag), times
  a window of 128 full decode steps, then (in profiled rows) runs 6 more steps under the torch profiler, then aborts
  its requests and drains.
- Per engine: a warm-up (one request of 1,024 prompt tokens and 32 output tokens in each mode that the engine uses),
  then the rows.

### 4.5 Engines and rows (each engine is one process; this order)

| # | Engine | Layout | Graphs | Capture mode | Rows (arm and batch) | Profiled rows |
|---|---|---|---|---|---|---|
| G1 | stock-g1 | stock | yes | OFF | STOCK 3, STOCK 3 | 1 |
| G2 | emulv-g | stock | yes | VIEWS | EMULV 3, EMULV 3 | none (A1) |
| G3 | compact-g | compact | yes | VIEWS | COMPACT 3, COMPACT 3, COMPACT 4 | 1 |
| G4 | stock-g2 | stock | yes | OFF | STOCK 3 | none |
| E1 | stock-e1 | stock | no | OFF | STOCK 3, EMULV 3, STOCK 3 | 1 (A1) |
| E2 | compact-e | compact | no | VIEWS | COMPACT 3, COMPACT 3 | none (A1) |
| E3 | stock-e2 | stock | no | OFF | STOCK 3, EMULV 3, STOCK 3 (A3) | none |

- The graph block (G1 to G4) decides; STOCK runs at both ends of it (drift). The eager block is secondary and runs
  last, so the time cap cuts it first.
- Rehearsal: the same engines and rows with Gemma 4 12B at 16,384 tokens, a 12 GiB pool, `max_model_len` 16,896 and a
  window of 32 steps.

## 5. Metrics

### 5.1 Primary
- s(row): the median decode step time of the row's window (ms), from `time.perf_counter` around `LLMEngine.step()` (as
  E-57). S_g = the median of the three STOCK-g rows (G1 twice, G4 once); E_g = the median of the G2 rows; C_g = the
  median of the two COMPACT 3 rows of G3.
- **OG = C_g / S_g - 1** (lower is better): the step cost of the compact build with CUDA graphs.
- MDE: 2% (MEASUREMENT.md default for speed).

### 5.2 Noise
- a_g: the largest in-process spread of the graph block, |row 2 / row 1 - 1| in G1, G2 and G3 (the batch-3 rows).
- p_g: the graph process and drift effect (A1, R-86 F4: G4 runs 3 engines after G1, so p_g also holds the drift),
  |s(G4) / median(G1 rows) - 1|. p_g and p_e are single pairs.
- **B_g = max(0.02, a_g, p_g)**: the band of the decision.
- Eager: a_e = the largest of |s(E1 row 3) / s(E1 row 1) - 1| and |s(E2 row 2) / s(E2 row 1) - 1|; p_e = |median(E3
  STOCK) / median(E1 STOCK rows) - 1|; B_e = max(0.02, a_e, p_e).

### 5.3 Secondary (reported; the words below do not change the decision)
- HG = E_g / S_g - 1; KG = C_g / E_g - 1 (the graph hook and cache costs).
- HE = the mean over E1 and E3 of s(EMULV) / s(STOCK) in the same process, minus 1 (the E-57 H: 4.8%).
- KE = median(E2 rows) / mean(EMULV of E1, EMULV of E3) - 1 (the E-57 K: 3.7%, across processes).
- GS = S_g / mean(STOCK of E1, STOCK of E3) - 1 (the graph gain of the stock step).
- TG = (4 / s(G3 COMPACT 4)) / (3 / S_g) (the decode rate of 4 compact requests against 3 stock requests).
- Per row: the mean step, the share of steps above 1.05 times the median (the bimodality of R-83), the median CPU
  time of the process per step (A1, R-86 F2: `time.process_time`, every thread), the CUDA graph replays of the window
  (a sum; the async worker can move one replay to the next step), the held blocks.
- Per engine (A1, R-86 F1): the free GPU memory after the start and the largest reserved memory.
- Per profiled row (6 steps): the kernel time, the GPU busy time (the union of kernel intervals), the GPU gaps, the
  launch calls and graph launches, and the top kernels by time, per step.
- Hook word: NA if the eager rows are missing; HOOK-NOREPRO if HE < B_e; else HOOK-FIXED if HG < B_g; else
  HOOK-REDUCED if HG <= HE / 2; else HOOK-STAYS.
- Cache word: NA if missing; CACHE-NOREPRO if KE < B_e (the eager cache cost is inside the eager band, which holds the
  process effect: "inside two pairs of noise", A1, R-86 F4); else CACHE-EAGER if KG < B_g (it exists eager and leaves
  with graphs); else CACHE-REAL.
- Report flags (A1, R-86 F3): FASTER if OG <= -B_g; CACHE-GAIN if KG <= -B_g.
- TG comes from one row; a batch-4 row that is not clean gives TG NA (A1, R-86 B2).
- Flag "process effect": p_e >= 0.02 or p_g >= 0.02.
- The repeats: 3 STOCK-g rows, 2 EMULV-g rows, 2 COMPACT-g rows, each a window of 128 steps. INTERPRETATION: the
  decision unit is the row median, and the noise rule (V5) replaces the 5-repeat default of MEASUREMENT.md, which
  this budget cannot buy for a 31B model.

## 6. Validity checks (a failed check makes the run INVALID)
- V1 (hooks inside the graphs): for each of the 3 prompts, the tokens of the first COMPACT 3 row of G3 equal the tokens
  of the first EMULV row of G2 over their common part, and the common part is at least 96 tokens (A1, R-86 B2: the
  rows hold about 134 tokens; 24 in the rehearsal). If the eager block
  ran: the same for E2 row 1 against the EMULV row of E1.
- V2 (setup): an A100 and the pinned vLLM; the 31B revision; every graph engine reports compilation mode NONE,
  cudagraph mode FULL_DECODE_ONLY and capture sizes that hold 3, 4 and 8, and its rows use its capture mode; every
  eager engine reports cudagraph mode NONE; the replays of every graph row's window are at least 0.9 times its steps
  (A1, R-86 B2), and every eager row has none. A1 (R-86 B3): a graph engine with no replay in its warm-up stops
  before its rows (exit 3), because the config can fall back to no graphs while it still reports FULL_DECODE_ONLY.
- V3 (scratch; A1, R-86 B1): every Triton metadata builder of every engine has a scratch that is at least 512 wide in
  its last dimension and has at least as many rows as the largest batch of the engine. An engine that fails V3 stops
  before its rows.
- V4 (rows): G1, G2, G3 and G4 ran all their batch-3 rows (A1, R-86 B2: the batch-4 row feeds only TG); every
  decision row is clean (no preemption, the planned batch, a
  window of at least 100 steps); the hooks attached to the 10 global layers with no problem; the compact spec applied
  in G3 (and E2).
- V5 (noise): B_g <= 0.03.
- Report only: the telemetry (clock and throttle reasons); the token equality of STOCK-g and STOCK-e and of the two
  processes of each stock engine (determinism).

## 7. Decision rule (frozen; project heuristics)
The first rule that applies:
- REHEARSAL: a rehearsal or a stub run.
- INVALID: a check of V1 to V5 fails (the line names it).
- GO: OG < B_g. With CUDA graphs, the run-time compact build has no step cost larger than the noise band at batch 3
  and 32K. Next: the thesis states the capacity gain with the graph decode rate (TG) as the measured throughput.
- KILL (of the run-time route with CUDA graphs): OG >= 0.05. The build costs 5% or more of a graph decode step. Next:
  the profile names the cost; the fix needs the load-time folds or a kernel change.
- PIVOT: every other case (B_g <= OG < 0.05). A cost remains; the profile and the secondary words name the next fix.
  A1 (R-86 F3): the word is PIVOT-WEAK if the fastest COMPACT row is faster than the slowest STOCK row (the rows
  overlap).
- A1: GO carries the flag FASTER if OG <= -B_g.

`analyze_e59.py` prints one line: `E-59: <word> | OG ... (band B_g ...) | HG KG | eager: HE KE p_e | hook word | cache
word | GS | TG | failed checks`.

## 8. Controls and baselines
- Baseline: STOCK-g (G1 and G4), the stock engine with the same graph settings.
- A/A: the second row of G1, G2, G3 and E1 or E2 (in process); G4 against G1 and E3 against E1 (across processes).
- Positive control: the eager block must reproduce the E-57 hook cost (HE >= B_e), or the hook word says NOREPRO; the
  graph engines must replay graphs in every window step (V2).
- Negative control: the compact engine and the emulated engine give the same tokens (V1), with graphs and without.

## 9. Budget
- Main run: 1 x A100 SXM 80GB at the live price (E-58: $1.79/h). Boot watchdog 3,600 s (`pod_cmd.py --cap 3600`),
  session watchdog 3,300 s, AUTO_STOP 240 s after the session, an SSH limit of 600 s. Cost cap $1.90 (60 min and the
  disks). Plan (INTERPRETATION, from E-57 and E-58): create to SSH 4 to 9 min; the download 4 to 9 min (62.6 GB; E-56:
  220 s); each engine start 60 to 90 s (the weights from the container disk and the page cache) plus about 30 s of
  graph capture; each row about 65 s (E-57: 64 s at 32K); 15 rows and 7 engines: about 26 min. The Pod about 45 min,
  about $1.35.
- What the time cap cuts first: the eager block (E3, then E2, then E1); a process does not start if less than 300 s
  are left (A1, R-86 F8; 150 s before). The graph block (G1 to G4) runs first. A1 (R-86 B3): if G1 does not end ok,
  the session skips every other engine and stops (no decision can exist). R-86 estimate: about 49 min, about $1.46,
  with a margin of about 7 min to the deadline.
- S5 rehearsal: 1 x A40 at the live price (E-58: $0.59/h), REH=1, Gemma 4 12B at 16,384 tokens. Boot watchdog 2,400
  s; cost cap $0.40. VERIFY: (1) the graph engines build with FULL_DECODE_ONLY and mode NONE, and capture; (2) the
  hooks attach at load in every engine (10 or the 12B count of global layers, no problem); (3) V1 holds with graphs
  and eager; (4) V3 holds (the scratch rows); (5) the replay counts (V2); (6) the profile summary of each profiled
  row; (7) the time of each engine and row, scaled to the A100 plan; (8) the session files, the code hashes, the
  REHEARSAL line, AUTO_STOP; (9) the download to the container disk.
- Balance floor: the protocol rule is the cap plus $35. If the live balance is lower, we decide (as D-95, D-98,
  D-101, D-103).

## 10. Threats to validity
1. Graphs without torch.compile. vLLM's default is FULL_AND_PIECEWISE with torch.compile; E-59 uses FULL_DECODE_ONLY
   with no compile, because the E-57 hooks are Python with side effects. The result covers full decode graphs; a
   production build would fold the transforms into the model code.
2. Hooks missing in the graphs: V1 (the compact cache gives wrong tokens without the hooks).
3. The scratch overflow of the pinned build (R-84, E-58 check G): batches up to 4 with 8 rows stay inside the buffer;
   V3 checks it per engine. The prefill and mixed steps use the 2D kernel (no scratch).
4. The process effect is estimated from one repeat per block (two pairs). It is a small sample; the band takes the
   largest spread.
5. A different A100 host than E-57: all comparisons are inside this Pod; the E-57 numbers are context only.
6. Bimodal step times (R-83): the median is the metric; the mean and the share of slow steps are reported.
7. GIL contention and the async scheduler (in process): the same in every arm, as E-57.
8. The profiler perturbs the steps: it runs after the timed window, in its own steps, and its numbers are report-only.
9. Drift over the session: STOCK at both ends of the graph block; telemetry for the whole session.
10. The token check compares greedy tokens of synthetic prompts; it shows that the arithmetic is the same, not that
    the text is useful (E-56 and E-57 tested answers).
11. TG compares different batches by design (stock does not fit 4 requests at 32K in the 15 GiB pool, E-57); it is
    the capacity gain, not a speed gain of the step.
12. A1 (R-86 F5): graph engines enable the vLLM JIT warm-up and eager engines disable it; the warm-up request runs in
    both. The eager STOCK rows run with the hooks attached in mode OFF (as E-57), which costs a little Python per
    global layer, so HE is understated by about 0.05 ms per step (R-86).
13. A1 (R-86 F1): memory. A fixed 15 GiB pool skips the graph memory profile. R-86 puts the headroom over the E-57
    peak at about 3.2 to 3.8 GiB and the risk of an out-of-memory error at about 5% (HYPOTHESIS). The pool stays (batch
    4 needs it); the engines record the free and the reserved memory.
14. A1 (R-86 F9): the A40 rehearsal (12B) does not test the 31B memory, the 31B head and KV-head numbers, or the
    balance of a GPU-bound step. R-86 also asked for a canary that shows the scratch fault without the fix. Answer: E-58
    check G measured that fault on the A100 and the A40 (4 MiB past the buffer at 8 requests); the widened scratch
    removes its cause, and V3 checks the width in every engine.

## 11. Files
- `run_e59.py`: one engine process (the hooks at load, the graph or eager engine, the records, the warm-up, the rows,
  the profile). `--stub` runs the same flow on a Mac with the E-57 stub scheduler and a step-time model.
- `analyze_e59.py`: the frozen analysis (V1 to V5, the numbers, the words, the line).
- `session_e59.sh`: the Pod session (the record, telemetry, the code hashes, the download to the container disk, the
  seven engines in order with the time rules, the analysis, AUTO_STOP).
- `make_manifest_e59.py`: the SHA-256 of each file that runs, at the design commit (A1, R-86 F10: with
  `analyze_e55.py`, which `run_e55.py` imports).
- `test_e59.py`: the Mac tests (the row and window logic, the stub session, the words and the checks on synthetic
  records).

## 12. Verification record (stage S3 and S5)
- Tests (2026-11-05, after A1): `uv run --no-project --with numpy --with torch --with pytest python -B -m pytest -q -p
  no:cacheprovider experiments/gemma4_kv_graphs/` gives 10 passed (A1 added the session stop after a failed G1 and the
  amended words and checks): the plan rules (a graph row in another mode, the
  compact layout without VIEWS, a batch of 5); the scratch rule (8 rows at head 256 hold batch 4; 4 rows fail batch 3;
  a fixed build with head 512); the window (the skipped steps, the profiled steps outside it, the trim, the slow share,
  the replays); the trace summary (overlapping kernels, gaps, categories, graph launches); the token check; the stub
  session; the planted failures; the decision words on scaled copies of the stub records (GO, PIVOT, KILL, INVALID by
  V5 and by V2, the hook and cache words).
- The stub session (`STUB=1 sh session_e59.sh` on the Mac, 83 s for the seven engines): every engine ok, the hooks
  attached at load, a REHEARSAL line with no failed check, V1 equal with graphs and eager, and the planted step-time
  model recovered (OG -0.94%, HE +5.13%, GS -10.0%, TG 1.334, HOOK-FIXED, CACHE-NOREPRO).
- Planted failures (direct engine runs): `scratch_small` stops the engine before its rows (exit 3, a scratch line in
  the pre-flight record); `graph_drops_hooks` fails V1; `no_replay` fails V2; a compact engine with mode OFF stops
  (exit 3).
- Not testable on the Mac (VERIFY in S5): the vLLM graph settings, the capture of the hooks, the replay counter, the
  scratch record of the real builders, the profiler trace format, and the times.

### 12.1 S5 rehearsal 1 (2026-11-05, one A40, Gemma 4 12B; D-105)
- Pod `<pod-id>` (`msthesis-e-59-rehearsal-20261105`, CA-MTL-1, Secure, $0.59/h, CUDA floor 13.2). Created
  19:56:26 UTC, SSH 423 s later, the session 20:03:44 to 20:05:46, AUTO_STOP at about 20:09:50, deleted (get-pod 404).
  Cost about $0.14 (the Pod times; billing not posted). Code hashes 9 of 9 at 0e132f2. Results:
  `results/e59-rehearsal-a40-20261105/` (27 files, checksums match).
- Passed: the 12B (23 GB) downloaded to the container disk in about 35 s (VERIFY 9); G1 built with compilation mode
  NONE and FULL_DECODE_ONLY and captured 5 full decode graphs (sizes 1, 2, 3, 4, 8); the hooks attached at load to the
  8 global layers of the 12B with no problem; the free GPU memory after the start was 9.1 GiB of 44.4 GiB; the session
  skipped G2 to E3 after the failed G1 and stopped (A1, R-86 B3); AUTO_STOP stopped the Pod.
- Failed: the G1 pre-flight stopped with "no Triton metadata builder found", and the widening wrapper at the
  builder's `__init__` counted 0. FACT (the pinned source): vLLM freezes the gc heap after the start
  (`freeze_gc_heap` in `v1/engine/core.py` line 250 and `gpu_worker.py` line 1117), and `gc.get_objects()` skips
  frozen objects, so the scan of V3 could not find any builder in any engine. Why the `__init__` wrapper did not count
  is not known (FACT: the pinned builder allocates the scratch in `__init__` with `model_config.get_head_size()`,
  which is `head_dim` = 256 for Gemma 4, so it should have widened); A2 adds a diagnosis line.
- Not reached: VERIFY 3 to 7 (V1, V3 on real builders, the replays in rows, the profiles, the times). A second
  rehearsal is needed.
- Correction (after rehearsal 2): the `__init__` wrapper did run in rehearsal 1. Rehearsal 2 shows 6 builder
  `__init__` calls per engine with a scratch that is already 512 wide (Section 12.2), so the wrapper had nothing to
  widen; the count of 0 did not mean that the wrapper did not run.

### 12.2 S5 rehearsal 2 (2026-11-05, one A40, Gemma 4 12B; D-106)
- Pod `<pod-id>` (`msthesis-e-59-rehearsal2-20261105`, CA-MTL-1, Secure, $0.59/h). Created 20:30:01 UTC, SSH
  471 s later, the session 20:38:07 to 20:48:39 (632 s: setup 41 s, seven engines), AUTO_STOP about 20:52:40.
  Code hashes 9 of 9 at 79f66a1. Results: `results/e59-rehearsal2-a40-20261105/` (42 files, checksums match).
- VERIFY (Section 9), all pass: (1) the four graph engines built with mode NONE and FULL_DECODE_ONLY and captured 5
  graphs each; (2) the hooks attached at load in every engine (8 global layers of the 12B, no problem, no hook error);
  (3) V1 holds with graphs and in eager mode (the decision line has no failed check); (4) V3: 6 builders per engine,
  each scratch [8 or 16 rows, 16 heads, 16 segments, 512]; (5) 32 graph replays in each 32-step graph window, 31 in
  each graph warm-up, none in eager rows; (6) a profile summary in G1, G3 and E1 row 1 (graph rows: 1 graph launch
  per step, 986 kernels per step for STOCK and 1,074 for COMPACT, gaps 0.4 to 1.1 ms per step); (7) engine starts 19
  to 46 s, rows 18.5 to 24.3 s, the whole session 632 s of a 1,491 s window; (8) the session files, the code hashes,
  the REHEARSAL line, AUTO_STOP; (9) the download to the container disk (41 s). Free GPU memory after each start:
  8.8 to 9.1 GiB of 44.4 GiB.
- FACT (the builder records): every builder of the 12B engines reports `headdim` 512 and a 512-wide scratch before any
  widening (A1 `__init__` wrapper: 6 calls per engine, 0 widened; A2 pass: 0 widened). INTERPRETATION: in this build
  the Gemma 4 architecture config is per layer (the pinned `Gemma4Config.verify_and_update_config` reads
  `arch_config[i].head_size` per layer), and `model_config.get_head_size()` gives the largest head size, 512, not
  `head_dim` = 256 as R-84 read. If the 31B record of the main run shows the same, the 3D scratch overflow of R-84
  F5 and E-58 check G does not happen in the real engine (E-58 sized its own scratch with 256). The main run decides.
- Rehearsal numbers (report only; 12B, 16K, window 32): OG +0.65% (band 2%), HG +0.50%, KG +0.16%, p_g 0.16%, HE
  -1.69%, KE +0.16%, p_e 3.92%, GS -4.35%, TG 1.312. The eager first-row effect led to A3.

## 13. Results (S6, 2026-11-05, one A100 SXM 80GB; D-105)

### 13.1 Run record
- Attempt 1: Pod `<pod-id>` (data center not reported). The local Pod volume `/workspace` (xfs with project
  quota) reported 20 GB used and 0 free while it was empty, so the boot watchdog could not write its log and the run
  script stopped at its first upload ("No space left on device"). Deleted after about 4 min (about $0.12). A host
  fault; no session ran.
- Attempt 2: Pod `<pod-id>` (US-MD-1, $1.79/h, GPU-beee1bcf, AMD EPYC 7742 host, 16 vCPU, 250 GB RAM). Created
  21:01:51 UTC, SSH 270 s later, boot watchdog 3,000 s (A100 cap $1.90 with attempt 1), the 31B download to the
  container disk 215 s, the session 21:06:35 to 21:33:47 (1,632 s), deleted 21:34:48 after the copy. Cost about $0.98
  (the A100 together about $1.10; with the rehearsals E-59 about $1.46; billing not posted). Code hashes 9 of 9 at
  a79c190. Results: `results/e59-a100-20261105/` (42 files, checksums match).
- Every engine ended ok with every planned row (15 rows), no preemption, windows of 128 steps, 128 graph replays in
  each graph window and none in eager rows, 31 replays in each graph warm-up, the hooks on the 10 global layers, the
  compact spec in G3 and E2. Free GPU memory after each start: 5.0 to 5.7 GiB of 79.3 GiB.
- Telemetry: the SM clock ran under the power cap (reason 0x4 in 947 of 1,093 busy samples). The median clock per
  engine was 1,365 MHz, except the compact engines G3 and E2 at 1,380 MHz (their power medians were 345 W and 333 W,
  against 342 W to 358 W). The 947 of 1,093 busy samples count the whole session; inside the engine windows R-87
  counts 947 of 1,071.

### 13.2 The frozen decision
`E-59: INVALID | OG -0.41% (band +2.00%; S_g 50.7333 ms, C_g 50.52495 ms) | graph HG +0.65% KG -1.06% process and drift
p_g +0.13% | eager HE +9.06% KE -7.21% p_e +6.67% | HOOK-NOREPRO CACHE-NOREPRO (process effect >= 2%) | GS -6.00% | TG
1.249 | failed checks: V1`

V1 failed: for prompt 2 (the third request), the COMPACT tokens differ from the EMULV tokens from position 3, with
graphs and in eager mode. V2 to V5 passed. Under the frozen rule (Section 7), the word is INVALID. Everything below is
exploratory.

### 13.3 Why V1 failed (FACT from the records; the interpretation is labelled)
- Prompts 0 and 1: the COMPACT tokens equal the EMULV tokens over all 162 and 146 common tokens, with graphs and eager.
- Within each layout, graphs and eager give the same tokens for every prompt, prompt 2 included: G3 (COMPACT, graphs)
  equals E2 (COMPACT, eager); G2 (EMULV, graphs) equals E1 and E3 (EMULV, eager); G1 (STOCK, graphs) equals G4 and
  E3 (STOCK).
- Every row with the stock cache has one deferred admission (`alloc_fail` 1: the third request waited for free
  blocks); every COMPACT batch-3 row has none (`alloc_fail` 0). The compact cache leaves more blocks free (E-57).
- INTERPRETATION: the deferred admission moves the third request's prefill chunks to other steps (the 2,048-token
  budget is shared with the decode tokens of the running requests), so its prompt is split at other positions and its
  arithmetic rounds differently. On a synthetic random-token prompt the next-token distribution is flat, so greedy
  decoding flips at position 3. V1 assumed equal schedules in the two engines, and the capacity effect itself breaks
  that assumption at 32K in this pool. The rehearsal (16K, a 12 GiB pool) had no deferred admission.
- What V1 was for (Section 2): to show that the hooks run inside the CUDA graphs. The equality of graphs and eager
  within each layout, for all prompts and about 130 to 160 decode tokens each, shows that directly.

### 13.4 Numbers (ms per decode step, medians of 128-step windows; batch 3 at 32K unless stated)

| Engine | Rows |
|---|---|
| G1 STOCK, graphs | 50.733, 50.689 |
| G2 EMULV, graphs | 51.061, 51.068 |
| G3 COMPACT, graphs | 50.513, 50.537; batch 4: 54.172 |
| G4 STOCK, graphs (second process) | 50.778 |
| E1 eager: STOCK, EMULV, STOCK | 52.581, 63.245, 59.078 |
| E2 eager: COMPACT, COMPACT | 54.779, 54.593 |
| E3 eager: STOCK, EMULV, STOCK (second process) | 52.000, 54.625, 52.217 |

- Graph block (the frozen numbers): S_g 50.733, C_g 50.525, E_g 51.064; OG -0.41%, HG +0.65%, KG -1.06%; the largest
  in-process spread 0.09%, the process and drift effect 0.13%, so B_g = 2% (the MDE). TG = 1.249.
- Graph profiles (6 steps after the window, report only): STOCK 1,225 kernels per step, COMPACT 1,335 (+110, the hook
  kernels); one graph launch per step; GPU gaps 0.37 and 0.38 ms per step. Kernel time by category, STOCK against
  COMPACT: attention 10.67 against 10.18 ms (-0.49 ms; E-58 measured -0.49 ms for the read); index and elementwise
  kernels 0.98 against 1.43 ms (+0.46 ms, the hook transform; E-58: +0.40 ms in a graph); GEMM 37.36 against 37.33 ms.
- The eager block, frozen numbers: HE +9.06%, KE -7.21%, p_e 6.67%, GS -6.00%. These are contaminated (below).

### 13.5 The profiler residue in E1 (FACT; interpretation labelled)
- E1 row 1 (STOCK) ended with 6 profiled steps. The rows after it ran slower in every mode: row 3 (STOCK) 59.078 ms
  against row 1 52.581 ms (+12.4%); row 2 (EMULV) 63.245 ms against the unprofiled E3 EMULV row 54.625 ms (+15.8%).
- E3 (no profile) shows no drift: rows 1 and 3 differ by 0.4%. In the graph engines, the rows after a profiled row
  match it (G1 0.09%, G3 0.05%).
- INTERPRETATION: the torch profiler leaves a CPU cost on each kernel launch after it stops. An eager step launches
  about 1,225 kernels and is bound by the CPU on this A100, so the cost shows (about 6.5 ms per step); a graph step
  launches one graph. The rehearsal did not show it, because the 12B step on the A40 is bound by the GPU.
- Clean eager readings (exploratory; E2 and E3 only, no profiled row): the hook cost HE = 54.625 / 52.108 - 1 =
  +4.83% (E-57: +4.8%); the eager cache cost KE = 54.686 / 54.625 - 1 = +0.11% (E-57: +3.7%); the eager process
  effect from the clean first rows, 52.000 against 52.581, is -1.1%; the graph gain of the stock step GS = 50.733 /
  52.345 - 1 = -3.1%.

### 13.6 The Triton 3D scratch (FACT; a correction)
- Every Triton builder of every 31B engine reports `headdim` 512 and a scratch of [8, 32, 16, 512] before any widening
  (6 `__init__` calls per engine, 0 widened by A1 or A2), as the 12B engines did in rehearsal 2.
- So the pinned engine sizes the 3D scratch for head 512 with Gemma 4: `model_config.get_head_size()` gives the
  largest head size of the per-layer architecture config, not `head_dim` = 256. The scratch overflow that R-84 F5
  read from the source, and that E-58 check G measured on a scratch that E-58 itself sized with 256, does not happen
  in the real engine. It did not touch E-56, E-57 or E-59. The widening of A1 and A2 changed nothing in this run.

### 13.7 Reading (INTERPRETATION, before the post-run review; Section 14 corrects it)
1. With full decode CUDA graphs, the run-time compact build is not slower than the stock engine at 3 requests of 32K
   tokens: 50.52 against 50.73 ms (-0.41%; the noise is about 0.1%). The frozen word is INVALID because V1 assumed
   equal schedules; the timing and the profiles are exploratory.
2. Under graphs, the compact cache saves about 0.5 ms of attention per step, and the hook transform costs about
   0.46 ms; they cancel. The E-57 hook cost (4.8%) is CPU launch time of an eager step: it is reproduced in eager mode
   (+4.83%) and falls to +0.65% with graphs.
3. The E-57 cache cost (+3.7%) is not reproduced: the clean eager compact engine is +0.11% against the clean eager
   emulated engine. The process effect here is about 0.1% with graphs and about 1% in eager mode.
4. The throughput of 4 compact requests against 3 stock requests with graphs is 1.249 times (E-57 eager: 1.243). The
   4th request adds 7.2% to the step at 32K.
5. Lessons: an output-identity check must hold the schedule fixed or compare only what the schedule cannot change; a
   profiler must not run inside a process that measures CPU-bound steps later; a host volume can fail on create (the
   run script stopped at once).

## 14. The corrected reading (R-87, 2026-11-05, CONFIRM_WITH_CORRECTIONS)

R-87 (`dossier/REVIEW_E-59_POSTRUN_2026-11-05.md`, reviewer) reran the frozen analysis and
recomputed every number in Sections 13.2 and 13.4 from the raw rows: all match. The word INVALID stands. This section
replaces Section 13.7 and corrects Sections 13.3, 13.5 and 13.6.
1. The frozen word (C1). INVALID stands. V1 was a design flaw: it assumed equal schedules in the two engines. New FACT
   (the `running` trace): all three requests first run at step 32 in every stock-cache row and at step 31 in every
   COMPACT batch-3 row; the `decoding` counts are equal. FACT: STOCK and EMULV also differ on prompt 2 (from position
   4), with the same layout and schedule, so prompt 2 flips for any small arithmetic change. INTERPRETATION: the
   schedule difference is sufficient to explain the V1 failure; it is not shown to be necessary (a block-boundary
   effect of the 48-token blocks is not excluded, but prompts 0 and 1 stay equal over 308 tokens). HYPOTHESIS: the
   next-token distribution of the random-token prompt is flat (no logit margins were recorded). The outputs are loops
   of single tokens, so V1 had low power on these prompts.
2. The hooks inside the graphs (C2). Three facts together show it: graphs equal eager within each layout for every
   prompt; EMULV differs from STOCK with graphs as it does in eager mode (the hooks change the arithmetic inside the
   graph); the COMPACT graph step runs 1,335 kernels against 1,225 for STOCK in one graph launch. This is a reading of
   the records, not a passed check.
3. The profiler residue (C3). INTERPRETATION, not a measured cause: host CPU noise is not excluded, and E1 rows 2 and
   3 are not independent of each other.
4. The clean eager readings (C4, C7). Sources: HE uses E3 only (+4.83%; +4.6% to +5.1% with one stock row or the
   other); KE uses E2 against E3, two processes (+0.11%, with about 1% of process uncertainty); the eager total is
   E2 against E3 STOCK, 54.686 / 52.108 - 1 = +4.95% (E-57: +8.7%, not reproduced); p_e (-1.1%) and GS (-3.1%) also
   use E1 row 1; E3 alone gives GS -2.6%.
5. The frozen eager words (C5). HOOK-NOREPRO, CACHE-NOREPRO, the process flag and HE +9.06%, KE -7.21%, p_e +6.67%, GS
   -6.00% are contaminated by the E1 rows after the profile. Do not quote them.
6. OG (C6). -0.41% lies inside the 2% band (the MDE). Write "no step cost larger than about 2% was seen", not
   "faster". The power cap does not drive OG: the GEMM time is equal (37.36 and 37.33 ms), and the lower power of the
   compact engines is part of the treatment.
7. TG (C8). 1.249 comes from one row, which had a deferred admission (`alloc_fail` 1).
8. The scratch (C9). R-84 F5, R-86 B1 and the E-58 check G inference are retracted. The check G measurement stays true
   for the 256-wide buffer that E-58 made itself. E-56 and E-57 ran the same build, so the overflow very likely did
   not happen there either (inferred: they recorded no scratch). The effect of PR #56996 on Gemma 4 is not known; do
   not say that it fixed an overflow. Section 4.3 (the A1 text "the stock build makes it 256 wide"), Section 10 items
   3 and 14, the A1 item B1 and the comments in `run_e59.py` (lines 98 and 122) are superseded by Section 13.6. The
   code stays as it ran (its hashes are the record).
9. Safe wording (R-87): "In E-59 (one A100, vLLM 3966924d5, Gemma 4 31B in 16 bit, full decode CUDA graphs without
   torch.compile) the pre-set validity check V1 failed, so the frozen word is INVALID. The two engines had different
   admission schedules for one prompt, and its greedy tokens differed. The timing data are exploratory. At 3 requests
   of 32K tokens the median decode step was 50.52 ms for the compact build and 50.73 ms for stock. The pre-set band was
   2%. We saw no step cost larger than about 2% in this run. We do not claim zero cost." Do not claim: no step cost
   with CUDA graphs, faster, output equality under graphs as a checked fact, the E-57 cache cost disproved, the
   frozen eager words, TG as validated, or any scratch overflow.
10. Next step (R-87; a recommendation, not a decision): either an E-60 repair run on one A100 (about $1.0 to $1.5:
    G2 and G3 and one stock graph engine; admission made equal by the pool size; V1 frozen on a chunk fingerprint;
    tokens checked on real text), or stop the GPU work for I-182 and write up the exploratory numbers with the INVALID
    label.

## Amendments

### A3 (2026-11-05, after S5 rehearsal 2, before the main run)
Rehearsal 2 (Section 12.2) showed a first-row effect in the eager engines: the first timed row ran slower than the
later rows of the same process (E1: 52.84 ms against 52.45 ms; E3: 54.71 ms, against 52.76 ms for its EMULV row 2).
E3 had one STOCK row, its first, so p_e mixed that effect with the process effect (3.92% in the rehearsal). The fix:
E3 runs STOCK, EMULV, STOCK, the row positions of E1, so p_e compares the medians of rows 1 and 3 in both processes.
Cost: one row (about 65 s on the A100). Unchanged: every rule and threshold, the decision rows, the graph block.

### A2 (2026-11-05, after S5 rehearsal 1, before the main run)
Rehearsal 1 (Section 12.1) stopped at the G1 pre-flight. The fixes (no rule changes):
- The scratch is widened on the builders that the model runner holds: a wrapper of `GPUModelRunner.initialize_kv_cache`
  widens every builder of `runner.attn_groups` right after the KV cache and the builders exist, before the warm-up runs
  and the graph capture (FACT, the pinned `gpu_worker.py`: `initialize_kv_cache`, then `compile_or_warm_up_model`, then
  `capture_model`). It works whatever the builder class is. The `__init__` wrapper of A1 stays and now prints a
  diagnosis line per builder.
- The V3 record reads the builders from `runner.attn_groups` (with their class names), not from a gc scan.
- A pre-flight exit now also writes the final record fields (the end time, the hook counts).
- Verification: 11 Mac tests pass (a new test widens fake runner builders and checks the record).
- Unchanged: every rule, threshold, engine, row and order.

### A1 (2026-11-05, before any run; R-86 AMEND_THEN_RUN)
R-86 (`dossier/REVIEW_E-59_PRERUN_2026-11-05.md`, reviewer; it read the pinned vLLM files and four
more at 3966924d5) confirmed that vLLM accepts the graph settings, that `self.model` in `load_model` is the captured
module, that the hook operations are safe to capture, and that the V-view wrappers only build views. It found three
blockers and ten findings. No data existed. The changes:
- B1: the capture warm-up of size 8 writes past the Triton 3D scratch (8 rows at head 256, indexed at 512), before V3
  runs. Fix: the scratch is widened to 512 in every engine (Section 4.3); V3 checks the width and the rows.
- B2: three rules could give a false INVALID: replays at least 0.9 times the window (not 1.0); V4 only for the batch-3
  rows (a bad batch-4 row gives TG NA); V1 at 96 common tokens (not 128).
- B3: a graph engine with no replay in its warm-up stops (exit 3); the session stops if G1 does not end ok.
- F1: the engines record the free and the reserved GPU memory. F2: the CPU time of every thread. F3: "No" means OG
  below the band; the flags FASTER and CACHE-GAIN; PIVOT-WEAK. F4: p_g is the process and drift effect; the meaning of
  CACHE-NOREPRO. F5: Section 10 items 12. F6, F7: no change (confirmed). F8: MIN_START 300 s; profiles only in G1, G3
  and E1 row 1. F9: Section 10 item 14 (answered). F10: `analyze_e55.py` in the hash list and the manifest.
- Unchanged: the question, OG, B_g, the GO and KILL thresholds, V5, the engines, the rows and the order.
- R-86 predictions (before data): GO about 65%; OG -1% (-3% to +1.5%); HE +4%; KE +1.5%; p_e 1.0%; p_g 0.7%; GS -14%;
  TG 1.33.
- Verification: 10 Mac tests pass (the new rules, the warm-up replay stop, the session stop after G1).
