# E-58: where the step cost of the E-57 compact K=V cache comes from (a weight-free kernel and hook microbenchmark; preregistration)

Status: RUN 2026-11-05 on one A100 SXM (D-103): GO (CACHE-NONE, HOOK-CPU; the E-57 cache cost stays open); R-85
CONFIRM_WITH_CORRECTIONS (Section 14). PREDECLARED 2026-11-05; amended A1 2026-11-05 (R-84, before any run; see
Amendments). S2: R-84 AMEND_THEN_RUN, applied as A1. S3: 6 tests pass and the stub session runs (Section 12). S5: the
A40 rehearsal passed (Section 12.1). Results: Section 13.
Idea: I-182. Decision: D-102 (we, 2026-11-05: "Plan the benchmark (about $0.60): a short GPU test to find why the
cache part is slower. And then add to the write ups clearly."). Origin: R-83 Section 8, Step 1.
Labels: FACT (a source, a log or a computation was read), INTERPRETATION (reasoning), HYPOTHESIS (not tested).
All thresholds are project heuristics unless a source is given.

## 1. Question

E-57 (one A100, vLLM 0.33.0rc1.dev229+g3966924d5, TRITON_ATTN, eager) measured that the compact build makes a decode
step slower at the same batch: at 32K tokens and batch 3, the run-time hooks add 2.55 ms (4.8%, EMULV against STOCK) and
the compact cache path adds 2.09 ms (3.7%, C3 against EMULV) to a 53.5 ms step. Which part costs this time, in
isolation at the E-57 decode shapes:
- the attention read of the Triton kernel with the compact layout (the 48-token block, or the two overlapping views),
- the cache write,
- the CPU time of the E-57 view wrappers,
- the CPU or GPU time of the hook transforms,
and can CUDA graphs or one fused transform remove the hook cost without a weight change?

A "no cause found" answer: no component reaches its threshold (Section 7), so the E-57 costs are a step-level effect.

## 2. Rubber duck

- Why a microbenchmark. A full engine rerun needs the 62.6 GB checkpoint, two engine loads (E-57: 558 s for one from
  the network volume) and has 1% to 2% row noise (R-83). The question is about four small components, and each can be
  timed directly with synthetic caches. No model weights are needed.
- Why these shapes. D32 is the E-57 32K window (3 requests of 32,768 tokens, where the split exists). D64 is the 64K
  window (2 requests of 65,536 tokens; E-57 has the total only: +2.34 ms by mean, +1.17 ms by median).
- Why the kernels can be called directly. FACT (`triton_attn.py` 644 to 735 and 805 to 843 at 3966924d5): the backend
  splits the cache row at the head size and calls `unified_attention` and `triton_reshape_and_cache_flash` with the
  arguments that `read_kwargs` copies (BF16, no window, no softcap, no sinks, scale tensors of 1.0, the 3D scratch).
  FACT (`triton_unified_attention.py` 976 to 1094): `block_size = v.shape[1]`; with 8 queries per KV head, BLOCK_M is
  16 and BLOCK_Q is 2; the decode tile is 16 tokens for head 512 in BF16 (`_get_tile_size`); the 3D path runs when
  the requests times cdiv(1, BLOCK_Q) are at most `seq_threshold_3D` = 128 // 16 = 8 (`triton_attn.py` 136, the model's
  16 KV heads). So 3 and 2 requests use the 3D path, with 16 segments, as in E-57.
- The tile and the block. INTERPRETATION: the 16-token decode tile divides both 32 and 48, so no tile reads two
  blocks. The block index needs one integer division per token. So the block size should cost little at decode.
- The views and the bytes. A stock row keeps K and V in separate 1,024-byte halves. A compact row keeps 640 values,
  and the V view reads 384 values that the K view of the same tile read a moment before. INTERPRETATION: if the L2
  cache serves that overlap, the compact read moves about 37.5% fewer DRAM bytes than the stock read; if the 3D kernel
  is bound by latency and not by bandwidth, both reads take about the same time.
- The same logical data. Every layout is filled from the same 640-value rows x: a compact row holds x; a stock row holds
  K = x[..., :512] and V = x[..., 128:], the two views of x. So all four layouts must give the same output (V1), and
  E-57 showed bit-identity between block 32 and block 48 (EQ EXACT).
- CPU and GPU time. The E-57 step was eager. A cost on the CPU adds to the step only when the CPU limits the step, and a
  cost on the GPU adds when the GPU limits it. So each arm reports both: the CPU issue time (the Python and launch time
  of a step, measured while a GPU sleep keeps the GPU busy, so the loop never waits) and the GPU time (CUDA events),
  plus the GPU time of a CUDA graph replay (no CPU at all).
- The closure check. The X arms run the 10-layer global step as the E-57 engines ran it (STOCK, EMULV, COMPACT, with the
  real wrappers and hooks). Their differences are the microbenchmark's form of the E-57 hook cost and cache cost.
- What E-58 does not test: the rest of the decode step (the GEMMs, the sliding layers, the scheduler), the slow-step mode
  of E-57 (R-83 Section 4), the effect between two engine processes, FP8, the FA4 path, the load-time folds.

## 3. Hypotheses (HYPOTHESIS; written before any run)

- H-VIEWS: C32 minus S32 is between -1.5 and +0.3 ms per step at D32 (the compact read is not slower; faster if the L2
  cache serves the view overlap and the kernel is bound by bandwidth).
- H-BLOCK: C48 minus C32 is within +/-0.3 ms per step (the tile divides both blocks).
- H-WRAPPER: the wrapper cost (C48 wrapped minus S32 wrapped, CPU issue time) is below 0.3 ms per step (10 as_strided
  calls and 20 checks).
- H-HOOK: the hook CPU issue time (VIEWS minus no hooks) is 1.5 to 3 ms per step, and its GPU time in a CUDA graph is
  0.1 to 0.5 ms (about 10 small kernels per layer). Predicted word: HOOK-CPU.
- Closure: the X hook row (EMULV minus STOCK) reproduces most of the E-57 +2.55 ms in CPU issue time; the X cache row
  (COMPACT minus EMULV) is small or negative in GPU time. If so, the E-57 cache cost is not in these components.
- Predicted decision: GO with CACHE-NONE (or "compact kernels faster") and HOOK-CPU.

## 4. Setup (frozen)

### 4.1 Hardware
- Main run (S6): 1 x A100 SXM 80GB (the E-57 GPU type), Secure, on demand, from a live catalog read at S4 (E-57:
  $1.79/h).
- Rehearsal (S5): 1 x A40 48GB (the E-57 rehearsal GPU type), Secure, on demand, from a live read at S4 (E-57: $0.59/h).

### 4.2 Software
- Image `vllm/vllm-openai@sha256:f255c4fa...` (the E-56 and E-57 image; vLLM 0.33.0rc1.dev229+g3966924d5, Triton 3.8.0,
  torch 2.14.0+cu130). No model download.
- The E-57 code, unchanged: `compact_engine.py` (`install`, `fix_value_cache`, `ViewsSplice`, `Indexer`),
  `compact_layout.py`, `splice_e55.py`.

### 4.3 Data and shapes
- D32: 3 requests of 32,768 tokens; D64: 2 requests of 65,536 tokens; one decode token per request; 10 layers; 32 query
  heads, 4 KV heads, head 512 (the Gemma 4 31B global layers). Random BF16 rows (seeded per shape and layer); the
  kernel time does not depend on the values.
- Layouts (each layer has its own cache, so the L2 cache cannot keep a layer between calls): S32 (stock row 1,024,
  block 32, page 262,144 B), S48 (stock row, block 48, page 393,216 B), C32 (compact row 640, block 32, page padded to
  262,144 B), C48 (compact row, block 48, page padded to 262,144 B). The LBNHC strides of the engine (the token index
  outside the head index). Each request uses a contiguous range of blocks, with one more slot for the decode write.

### 4.4 Arms (`bench_e58.py`, one process; `compact_engine.install("compact")` wraps the backend's two kernel entry points
as in E-57)
| Arm | What runs (10 layers per step) | Measures |
|---|---|---|
| R S32, S48, C32, C48 | the raw `unified_attention` read | GPU time (CUDA events), 3 interleaved rounds |
| W S32, C48 | the raw cache write of the decode tokens | GPU time |
| P S32, C48 raw | a backend that slices the two views itself, then the raw write and read | CPU issue time |
| P S32, C48 wrapped | the backend split, then the E-57 wrappers, the write and the read | CPU issue time |
| H NONE, OFF, VIEWS | modules that do no attention, without hooks and with `ViewsSplice` (mode OFF, mode VIEWS) | CPU issue time, GPU time, GPU time of a CUDA graph, kernel count |
| H EAGER_FN, COMPILED_FN | the VIEWS transform without module hooks, eager and with torch.compile | the same |
| X STOCK, EMULV, COMPACT | 10 layer modules (the write and the read; EMULV and COMPACT with the hooks in mode VIEWS; COMPACT on the C48 cache through the wrappers) | CPU issue time, GPU time, wall time with a sync per step, GPU time of a CUDA graph, kernel count |
| G (A1, report only) | one global-layer decode call with 8 requests (the 3D path) on a scratch of the size that vLLM 3966924d5 allocates (head 256), inside a guard buffer | the guard values that the kernel changed |

### 4.5 Order and counts
- Per shape (D32, then D64): build and fill the 40 caches; V1; the R rounds; W; P; H; X.
- GPU time: 3 warm-up steps, then 50 steps between two CUDA events; before the R rounds, one warm round of each layout
  (A1). CPU issue time: the median of 9 loops of 3 steps (A1; 5 before), each loop behind `torch.cuda._sleep` (about
  70 ms of GPU time, measured once per shape), with checks that the GPU was still busy at the end of the loop and that
  the loop took at most half of the sleep (V3). Wall time: 50 steps with a sync after each. Graph: the step captured
  once, then 50 replays; the H NONE arm has no GPU work, so its graph time is 0 and no empty graph is captured (A1).
- After both shapes: the scratch check G (A1, report only).

## 5. Metrics

### 5.1 Per step of 10 layers (ms), at each shape
- views = R(C32) - R(S32); block = R(C48) - R(C32); block (stock) = R(S48) - R(S32); kernels = R(C48) + W(C48) -
  R(S32) - W(S32).
- wrapper (the E-57 engine B against engine A) = P(C48 wrapped) - P(S32 wrapped); reported: P(wrapped) - P(raw) per
  layout.
- hook CPU = H(VIEWS).cpu - H(NONE).cpu; hook GPU (graph) = H(VIEWS).graph - H(NONE).graph; reported: the eager event
  time (A1, R-84 F2: the hook loop is bound by the CPU, so this number is not a GPU time), the hook dispatch cost
  H(VIEWS) - H(EAGER_FN), and the compiled transform.
- Closure (reported): X(EMULV) - X(STOCK) and X(COMPACT) - X(EMULV), in CPU issue, GPU, wall and graph time, against
  the E-57 +2.55 and +2.09 ms. A1 (R-84 F1): the closure is judged on the X wall row; the CPU row is an upper bound,
  because a CPU cost adds to a step only when the CPU limits the step.
- The scratch check G (report only): the count of guard values past the engine-size scratch that changed, and whether
  the output equals the output with a scratch of the right size.
- Reported: the achieved bandwidth of each read (distinct bytes / time), the kernel counts.

## 6. Validity checks (a failed check makes the run INVALID)
- V1: at each shape, the outputs of S48, C32, C48, C48 through the wrappers, and the X COMPACT layer against X EMULV,
  are bit-identical to the S32 output (or to each other), or differ by at most 0.002 (A1; 0.02 before: R-84 B3, the
  output maximum is about 0.09); at D32 all 6 checks exist.
- V2 (main run): the GPU is an A100 and the vLLM version is 0.33.0rc1.dev229+g3966924d5.
- V3: in every CPU issue measurement, the GPU was still busy when the issue loop ended, and no loop took more than half
  of the measured sleep (A1, R-84 F3).
- V4: at D32, the round spread (max / min - 1) of each R arm is at most 0.03 (A1; 0.10 before: R-84 B3, 10% of a
  read is above the 0.5 ms line).
- V5: at D32, V1, the R, W, P (wrapped), H (NONE, VIEWS) and X arms ran without an error, and the run ended ok.

## 7. Decision rule (frozen; project heuristics; at D32)
- Cache words (several can hold): CACHE-VIEWS if views >= 0.5 ms; CACHE-BLOCK if block >= 0.5 ms; CACHE-WRAPPER if
  wrapper >= 0.5 ms; else CACHE-NONE. Reported flag: "compact kernels faster" if kernels <= -0.5 ms. (0.5 ms is 25% of
  the E-57 cache cost, R-83 Section 8.)
- Hook word (A1, R-84 B1: four words that exclude each other; C = hook CPU, G = hook GPU in a graph): HOOK-CPU if C >=
  1.0 ms and G < 0.5 ms; HOOK-GPU if G >= 0.5 ms and C < 1.0 ms; HOOK-MIXED if C >= 1.0 ms and G >= 0.5 ms; HOOK-SMALL
  if C < 1.0 ms and G < 0.5 ms; HOOK-NA if a needed number is missing. A separate flag "explains E-57" is set if C >= 2.0
  ms; it does not choose the word. HOOK-CPU says that the hook has a CPU cost; it does not say that the cost is on the
  critical path of an engine step (R-84 F1).
- The decision (the first rule that applies):
  - REHEARSAL: a rehearsal or a stub run.
  - INVALID: a check of V1 to V5 fails, or a word is NA.
  - KILL (of the stock-kernel route of the compact read): views >= 1.5 ms at both D32 and D64. The compact read in the
    stock kernel costs more GPU time than most of the E-57 cache cost, although it moves fewer bytes; a custom kernel
    would be needed. (A1, R-84 F4: "views" includes the byte saving, so a negative value shows that the compact read is
    not slower, not that the two views cost nothing.)
  - PIVOT: CACHE-NONE and HOOK-SMALL. The microbenchmark does not reproduce the E-57 costs; the next step is a profile
    of one engine decode step (it needs the model).
  - GO: every other case. A cause is found, and the word names the next fix. A1 (R-84 F7): GO with CACHE-NONE explains
    at most the hook cost; the E-57 cache cost then stays open (the line says so), and the next step includes a profile
    of one engine decode step.

| Word | Meaning | Next |
|---|---|---|
| HOOK-CPU | the hook cost is CPU time | CUDA graphs for decode, or one fused transform; no weight change |
| HOOK-GPU | the hook kernels cost GPU time | the load-time folds (R-83 Step 2) |
| HOOK-MIXED | both parts matter | the load-time folds (they remove both parts) |
| HOOK-SMALL | the hooks cost little in isolation | the E-57 hook cost is a step-level effect |
| CACHE-VIEWS | the two-view read costs GPU time | a reordered row, separate K and V rows, or a kernel change |
| CACHE-BLOCK | the 48-token block costs GPU time | a 32- or 64-token block (a P4 page plan) |
| CACHE-WRAPPER | the V-view fix costs CPU time | make the V view in the backend, not in a Python wrapper |
| CACHE-NONE | the kernels and the wrappers do not explain the E-57 cache cost | the E-57 cache cost is a step-level effect |

- `analyze_e58.py` prints one line: `E-58: <word> | D32 cache | hooks | X hook | X cache | D64 | scratch guard |
  failed checks`.

## 8. Controls and baselines
- The baseline: S32, the E-57 stock layout, raw kernel calls.
- One factor per comparison: views (C32 against S32: the row only; the same block and page stride), block (C48 against
  C32: the block only; the same row and page stride), wrapper (the same kernels, with and without the Python layer).
- A/A: the 3 interleaved rounds of each R arm (V4).
- Bit-identity (V1): the four layouts compute the same result from the same logical K and V.

## 9. Budget
- Main run: 1 x A100 SXM 80GB at the live price (E-57: $1.79/h). Boot watchdog 900 s (`pod_cmd.py --cap 900`), session
  watchdog 900 s, AUTO_STOP 240 s after the session, an SSH limit of 600 s. Worst case: 10 min to SSH plus the boot
  watchdog: 25 min, $0.75. Cost cap $0.75. Plan (INTERPRETATION): create to SSH 4 to 6 min; setup 1 min; the benchmark
  about 6 min (Triton compiles about 2 min, torch.compile about 1 min, the timed arms about 2 min); copy 1 min. Pod
  about 15 min, about $0.45.
- S5 rehearsal: 1 x A40 at the live price (E-57: $0.59/h), REH=1, the same script and shapes. Boot watchdog 900 s; worst
  case 25 min, $0.25. Cost cap $0.25. VERIFY: (1) both shapes run with no arm error (if D64 does not fit the 48 GB,
  the record says so and D32 is complete); (2) V1 on the GPU; (3) V3; (4) the graph and compile arms run or record
  their error; (5) the kernel counts (the profiler works); (6) the session files, the code hashes and the REHEARSAL
  line; (7) the time of each part, against the 900 s of the A100 session (A1, R-84 F6: if D32 takes more than 250 s
  on the A40, raise the Pod boot margin or drop COMPILED_FN at D64 with an amendment); (8, A1) V3 holds with the loop
  bound (the X COMPACT loop well inside half of the sleep); (9, A1) the scratch check G runs.
- Balance floor: the protocol rule is the cap plus $35. If the live balance is lower, we decide (as D-95, D-98,
  D-101).

## 10. Threats to validity
1. Isolation. The arms run the global-layer operations without the rest of the model, so the CPU and GPU overlap of a
   real step is not reproduced. Each arm reports the CPU and the GPU time separately, and the X rows show the sum.
2. Eager mode. E-57 ran eager; vLLM decodes with CUDA graphs by default. The graph rows show what a graph removes.
3. Synthetic data and contiguous blocks. The kernel time does not depend on the values. An engine pool with requests in
   interleaved blocks can change the L2 and TLB behavior a little.
4. One process. The effect between the two E-57 engine processes is not measured.
5. The 3D decode path is assumed from the source (3 and 2 requests, threshold 8); the record keeps the metadata.
6. The slow-step mode of E-57 (R-83 Section 4) is a step-level effect that a microbenchmark cannot show.
7. Fidelity differences (A1, R-84 F5; they do not change the plan): the decode write goes to a slot that the read does
   not cover (in the engine the new token is read); the P and W writes put random values into the overlap of a compact
   row (the slot is never read); the engine calls each layer through custom ops, which add the same CPU time to both
   engines (they cancel in the differences, not in the absolute CPU numbers); the benchmark loops keep the Python code
   hot in the CPU cache, so an engine may pay 1.3 to 2 times the measured CPU cost per hook (INTERPRETATION, R-84); the
   host CPU is recorded (`lscpu.txt`) for a comparison with the E-57 Pod.
8. The 3D scratch of the pinned build (A1, R-84 F5). FACT (the source at 3966924d5: `triton_attn.py` 117 and 163,
   `model_arch_config_convertor.py` 91 to 117, `triton_unified_attention.py` 631 to 642): the Triton metadata builder
   sizes `softmax_segm_output` with `model_config.get_head_size()`, which is `head_dim` = 256 for Gemma 4 31B, while the
   kernel indexes it with the layer head size, 512 for the global layers. So a decode batch of 5 to 8 requests in the
   3D path (eager) writes up to 4 MiB past the buffer at the global layers. FACT (vLLM main, commit a0d909f53, PR
   #56996, 2026-11-03): the builder now takes the head size from the KV cache spec, which fixes this. E-56 and E-57
   ran the pinned build: E-57 had decode batches of up to 6 in its output phase (the needle groups) in every arm, so
   the compared arms ran the same code, and the baselines were healthy; what the overflow overwrote is unknown. The
   check G shows whether the overflow happens at runtime. The benchmark arms use a scratch of the right size (head 512)
   and batches of 3 and 2.
   RETRACTED (2026-11-05, E-59 README Section 13.6; R-87 C9; journal Section 8.1 row 77): the pinned engine sizes the 3D
   scratch for head 512 with Gemma 4 (every builder of the 31B and 12B engines reports `headdim` 512 and a 512-wide
   scratch before any change), because `get_head_size()` gives the largest head size of the per-layer architecture
   config. The overflow does not happen in the real engine. Check G stays true for the 256-wide buffer that E-58 made
   itself; the `bench_e58.py` comment on the engine head size (line 59) is superseded, and the code stays as it ran.

## 11. Files
| File | Role |
|---|---|
| `bench_e58.py` | the benchmark (one process; `--stub` on the Mac) |
| `analyze_e58.py` | the frozen analysis |
| `session_e58.sh` | the Pod session (`STUB=1` on the Mac, `REH=1` for the rehearsal) |
| `make_manifest_e58.py` | the SHA-256 manifest of the files that run, at the design commit |
| `test_e58.py` | the S3 tests |
| reused, unchanged | `../gemma4_kv_engine/compact_engine.py`, `compact_layout.py`, `../gemma4_kv_model/splice_e55.py` |

## 12. Verification record (stage S3)
- Tests: `uv run --no-project --with numpy --with torch --with pytest python -B -m pytest -q -p no:cacheprovider
  experiments/gemma4_kv_microbench/` gives 6 passed (2026-11-05, after A1): the pages, strides and views of the four layouts; the
  same logical K and V in every layout; the reference read against a float32 attention and its bit-identity across the
  layouts; the stub benchmark (every arm, V1 bit-identical in all 6 checks) and its analysis line; the decision words on
  synthetic records (GO, PIVOT, KILL and each cache and hook word); the validity checks (V1, V2, V3, V4, V5,
  REHEARSAL).
- The session script in stub mode on the Mac (`STUB=1`, a boot watchdog log 300 s old): the deadline line names the
  boot watchdog end; the benchmark, the analysis and the code hashes end with the expected files and an "E-58:
  REHEARSAL" line.

### 12.1 S5 rehearsal record (2026-11-05, one A40, D-103)
- Pod `<pod-id>` (`msthesis-e-58-rehearsal-20261105`, CA-MTL-1, Secure, $0.59/h). Created 16:43:14 UTC, SSH
  512 s later, the session 16:52:01 to 16:53:32 (setup 11 s, the benchmark 91 s), AUTO_STOP at 16:57:3x, deleted
  (get-pod 404). Cost about $0.14 (cap $0.25). Code hashes 6 of 6 at 6c347eb. Results:
  `results/e58-rehearsal-a40-20261105/` (26 files, checksums match).
- VERIFY (Section 9), all pass: (1) both shapes, no arm error; (2) V1 bit-identical in all 12 output checks; (3) every
  issue loop kept the GPU busy; (4) the graph and compile arms ran (no graph error); (5) kernel counts 110 (VIEWS and
  EAGER_FN), 40 (COMPILED_FN), 30 (X STOCK), 140 (X EMULV and X COMPACT); (6) the session files, the code hashes and
  the REHEARSAL line; (7) D32 took about 49 s (limit 250 s), the whole benchmark 91 s of a 727 s window; (8) the
  longest loop 17.5 ms against a bound of 29.2 ms (half of the 58.3 ms sleep); (9) check G ran.
- Report-only numbers on the A40 (not the decision GPU): the read round spread 0.0% to 0.6%; check G changed
  1,048,576 guard values (the whole second half of the kernel's 2,097,152-float need; the outputs stayed identical).
  No amendment is needed.

## 13. Results (S6, 2026-11-05, one A100 SXM 80GB; D-103)

Pod `<pod-id>` (US-MD-1, $1.79/h, GPU-669671f1, AMD EPYC 7742 host: the same CPU model as E-57, a different
GPU). Created 17:05:50 UTC, the session 17:10:14 to 17:11:26 (the benchmark 72 s), deleted 17:11:54 after the copy.
Cost about $0.18 (cap $0.75); E-58 total about $0.32 with the rehearsal (caps $0.25 and $0.75). Two creates with minCudaVersion 13.0 and a 40 GB disk and volume
failed for capacity; the third (13.2, 30 GB and 20 GB) worked. No model is loaded, so the disk sizes do not change the
measurement. Results: `results/e58-a100-20261105/` (26 files, checksums match). The Mac analysis gives the same line.

Decision line (frozen rule, Section 7):

`E-58: GO (the E-57 cache cost stays open) | D32 cache: CACHE-NONE (compact kernels faster) (views -0.33 ms, block
-0.16, wrapper +0.17, kernels -0.50) | hooks: HOOK-CPU (explains E-57) (CPU +3.14 ms, GPU graph +0.40) | X hook: wall
+1.04 CPU +3.51 graph +0.46; X cache: wall +0.16 GPU +0.27 CPU +0.16 graph -0.45 (E-57: hook +2.55, cache +2.09) | D64:
views -0.29, block +0.07, kernels -0.23 | scratch guard: 1048576 values past the engine-size scratch changed | failed
checks: none`

### 13.1 Numbers (FACT; ms per 10-layer step; medians of 3 rounds for the reads)

| Quantity | A100 D32 | A100 D64 | A40 D32 (rehearsal, report only) |
|---|---|---|---|
| Read S32 / S48 / C32 / C48 | 5.076 / 4.994 / 4.741 / 4.582 | 8.411 / 8.037 / 8.120 / 8.186 | 14.886 / 14.623 / 10.739 / 10.909 |
| views (C32 - S32) | -0.335 | -0.291 | -4.146 |
| block, compact (C48 - C32) / stock (S48 - S32) | -0.158 / -0.081 | +0.066 / -0.375 | +0.170 / -0.263 |
| kernels (read and write, C48 - S32) | -0.502 | -0.227 | -3.802 |
| wrapper (CPU, C48 wrapped - S32 wrapped) | +0.175 | +0.185 | +0.142 |
| hook CPU (VIEWS - NONE) / hook GPU in a graph | +3.142 / +0.396 | +3.142 / +0.394 | +2.452 / +0.306 |
| hook dispatch CPU (VIEWS - EAGER_FN) | +0.590 | +0.594 | +0.481 |
| compiled transform: CPU / graph (kernels) | 1.028 / 0.076 (40) | 1.036 / 0.071 (40) | 1.028 / 0.078 (40) |
| X hook (EMULV - STOCK): wall / CPU / GPU / graph | +1.042 / +3.507 / +0.828 / +0.459 | +0.889 / +3.528 / +0.654 / +0.496 | +0.655 / +2.679 / +0.479 / +0.385 |
| X cache (COMPACT - EMULV): wall / CPU / GPU / graph | +0.158 / +0.160 / +0.269 / -0.448 | -0.205 / +0.184 / -0.222 / -0.213 | -3.999 / +0.123 / -4.022 / -4.001 |
| Read ratio C32 / S32 | 0.934 | 0.965 | 0.721 (D64: 0.634) |
| DRAM rate, distinct bytes: S32 / C32 (GB/s) | 1,587 / 1,062 | 1,277 / 826 | 541 / 469 |

- Validity: V1 bit-identical in all 12 output checks; V2 A100 SXM and the pinned vLLM; V3 every loop kept the GPU busy,
  the longest loop 18.9 ms against a bound of 35.6 ms; V4 the largest round spread at D32 0.13% (S48; R-85 C9); V5
  no arm error. The SM clock was 1,410 MHz in 77 of 79 telemetry samples with a busy GPU and 1,155 MHz in 2 (R-85 C9);
  no throttle reason was set in the busy samples.
- Peak DRAM rates for the rate rows (data sheet values): A100 SXM 80GB 2,039 GB/s; A40 696 GB/s (R-85 C7).
- Kernel counts: 110 hook kernels per 10 layers (VIEWS, EAGER_FN), 40 compiled, 30 for the stock layer, 140 for the
  EMULV and COMPACT layers.
- Check G (report only): with 8 decode requests of 2,048 tokens, the 3D decode path of the pinned build changed all
  1,048,576 guard values past a scratch of the engine size (4 MiB of float32); the outputs equal the outputs with a
  scratch of the right size. The A40 gave the same count.
  Correction (2026-11-05, E-59): "the engine size" was an assumption (head 256); the real engine makes the scratch 512
  wide, so check G measured a buffer that the engine never has (journal Section 8.1 row 77).
- Against the hypotheses (Section 3): H-VIEWS, H-BLOCK and H-WRAPPER hold; H-HOOK holds for the graph time (0.40 ms)
  but the CPU time (3.14 ms) is 0.14 ms above the predicted 1.5 to 3 ms; the closure is as predicted; the predicted
  word (GO, CACHE-NONE, HOOK-CPU) is the result. Against R-84: GO and HOOK-CPU as predicted; the compact read is faster
  by less than predicted (0.34 ms for views, 0.49 ms for the read, against 0.7 to 2.5 ms); the hook CPU time is above
  the predicted 1.2 to 2.8 ms.

### 13.2 Reading (INTERPRETATION, before the post-run review; R-85 corrects it, Section 14 replaces it)
1. The attention layer does not hold the E-57 cache cost. In eager mode the compact layer costs +0.16 ms of wall time
   per step, the same as the CPU cost of the V-view wrappers (+0.16 to +0.18 ms); in a CUDA graph it is 0.45 ms faster.
   About 1.9 ms of the E-57 +2.09 ms is outside the layer: a step-level effect. Candidates (not tested): the KV cache
   manager and block tables with the 48-token block and the larger common block, the slot mapping, the scheduler, the
   E-57 instrumentation, and the bimodal step times that R-83 found. The next step is a profile of one engine decode
   step (it needs the model).
2. The hook cost is CPU issue time. The 110 small kernels of the emulation transform cost 3.14 ms of CPU and 0.40 ms of
   GPU in a graph per step. E-57 ran eager; its hook cost (+2.55 ms) lies between the GPU part (0.4 to 0.8 ms) and the
   CPU part (3.1 to 3.5 ms), so the E-57 eager decode step was partly bound by the CPU. A CUDA graph or the compiled
   transform (1.03 ms CPU, 0.08 ms graph, 40 kernels) removes most of it with no weight change; the load-time folds
   remove it all.
3. The compact read is faster on both GPUs, by different amounts. On the A40 the stock read runs at 78% of the DRAM
   peak and the compact read at D64 takes 0.634 of the stock time (the byte ratio is 0.625): the byte saving becomes
   time. On the A100 the stock read also runs at 78% of its peak, but the compact read reaches only 52% in distinct
   bytes and saves 3.5% to 6.6%: another limit sets its time (probably the L2 traffic of the two overlapping views, or
   the issue rate). This repeats the E-52 finding on the H100 (the stock kernels are mostly bound on chip).
   HYPOTHESIS: the byte saving pays more on GPUs with less bandwidth per unit of compute (A40, L40S, consumer cards).
   One A40 rehearsal (report only) supports it; no run tests it.
4. The flag "compact kernels faster" holds by 0.002 ms at D32 (-0.502 against -0.5). The round spread is 0.1%, so the
   sign is solid, but the flag is at its edge.

## 14. The corrected reading (R-85, 2026-11-05, CONFIRM_WITH_CORRECTIONS)

R-85 (`dossier/REVIEW_E-58_POSTRUN_2026-11-05.md`, reviewer) recomputed every number from
`bench.json` and confirmed the decision (GO, CACHE-NONE, HOOK-CPU) and the use of the frozen rule. It found the reading
in Section 13.2 too strong. This section replaces Section 13.2.
1. What E-58 reproduces (C1). The flag "explains E-57" comes from the CPU row (3.14 ms). The wall row, which A1 names
   for the closure, gives +1.04 ms of the E-57 +2.55 ms hook cost (41%). The hook and cache rows together give +1.20
   ms of the 4.64 ms that E-57 measured at 32K (26%). FACT: about three quarters of the E-57 step overhead is not
   located.
2. The cache cost (C2, C5, C6). FACT: the compact attention layer is not slower than the stock layer (eager wall
   +0.16 ms, equal to the wrapper CPU cost; graph -0.45 ms). The part of the E-57 +2.09 ms that lies outside the layer
   is 1.9 ms if the engine step is bound by the CPU and about 2.5 to 2.6 ms if it is bound by the GPU. The first
   candidate (C5) is the process: K is the only E-57 comparison across two engine processes (COMPACT in engine B
   against EMULV in engine A), and the E-57 A/A spread was measured inside one process. The bimodal slow steps give at
   most 0.43 ms (the mean ratio 1.087 against the median 1.079). The E-57 instrumentation is not a candidate (C6): the
   `allocate_slots`, `_preempt_request` and `schedule` wrappers run in both engines, `STATE["sched"]` is None in the
   capacity rows, and the page size (262,144 B) and the pool (6,144 blocks) are equal.
3. The hook cost (C3). FACT: the 110 small kernels of the run-time transform cost 3.14 ms of CPU issue time and
   0.40 ms of GPU time in a CUDA graph per step; the compiled transform costs 1.03 ms of CPU and 0.08 ms in a graph.
   HYPOTHESIS (not a finding): the E-57 eager step was partly bound by the CPU. E-57 has no CPU step time; the
   microbenchmark X rows are themselves CPU-bound for EMULV and COMPACT and GPU-bound for STOCK. Other causes of the
   E-57 H: GIL contention (E-57 ran the engine in process with its threads), cold CPU caches in a 60-layer step, and
   GPU gaps between the small kernels.
4. The kernels flag (C4). The flag holds by 0.0025 ms (-0.5025 against -0.5). The read alone gives -0.493, which does
   not pass; the write adds -0.009. INTERPRETATION: the X graph rows show almost no GPU time for the write (X STOCK in
   a graph, 5.070 ms, is below R S32, 5.076 ms), so the W row (0.49 ms) is mostly eager launch time. At D64 the flag
   fails (-0.23 ms).
5. The read saving and bandwidth (C7). FACT: the arithmetic holds (A100: 1,587 and 1,062 GB/s, 77.8% and 52.1% of
   2,039 GB/s; A40: 541 of 696 GB/s, 77.7%). The A40 compact time ratio is 0.721 at D32 and 0.634 at D64. The guess
   "L2 traffic of the two views" is weak: the traffic from L2 to the SMs is 8.05 GB in both layouts; only the DRAM
   bytes fall. A latency or wave limit fits as well (192 programs on 108 SMs at D32). The A40 differs in L2 size (6 MB)
   and SM count, so the A40 and A100 contrast is confounded. HYPOTHESIS: the read saving grows on GPUs with less DRAM
   bandwidth per unit of compute; one A40 rehearsal supports it, and no run tests it.
6. The scratch overflow (C8). FACT (check G): with 8 decode requests of 2,048 tokens, the 3D decode path of the pinned
   build writes 4 MiB past a scratch of the engine size. G does not show the run-time head size of the engine or the
   real neighbor of the buffer. E-57 sent 400 GSM8K requests in one `generate` call with a cap of 16 sequences, and
   E-56 did the same with a cap of 64 (plus 32 needle prompts); both tails drain through decode-only steps of 5 to 8
   requests. The E-57 capacity rows (batches 2 to 4) are not in that range. Every arm ran the same code, so the
   comparisons hold; the risk to the absolute accuracy numbers is not known. The author read the fix in the vLLM
   main source (Section 10 item 7; R-85 could not check it, no network); no run tests the fixed build.
   RETRACTED (2026-11-05, E-59 README Section 13.6; R-87 C9; journal Section 8.1 row 77): the pinned engine sizes the 3D
   scratch for head 512 with Gemma 4 (every builder of the 31B and 12B engines reports `headdim` 512 and a 512-wide
   scratch before any change), because `get_head_size()` gives the largest head size of the per-layer architecture
   config. The overflow does not happen in the real engine. So there was no risk to the E-56 and E-57 accuracy numbers
   from this source (inferred for E-56 and E-57, which recorded no scratch; measured in E-59 on the same build).
7. Safe wording (R-85 Section 4): "In a weight-free test of 10 global layers on one A100, the compact K=V read was not
   slower than the stock read (0.3 to 0.5 ms faster per step)." "The 110 small kernels of the run-time emulation cost
   3.1 ms of CPU issue time and 0.4 ms of GPU time in a CUDA graph per step." "The test reproduces about one quarter
   of the 4.6 ms step overhead that E-57 measured; the rest is not located." Do not claim that E-58 explains the E-57
   overhead, that the E-57 step was CPU-bound, that the byte saving pays on low-bandwidth GPUs, that E-56 and E-57
   were untouched by the overflow, or that the overflow is fixed.
8. Next step (R-85 Section 5; a recommendation, not a decision): profile one real decode step (batch 3, 32K) for
   STOCK, EMULV and COMPACT, with the CPU forward time, the GPU busy time and the GPU gaps, and repeat the stock engine
   in a second process to measure the process effect. A GPU run with the model (R-85 estimate: $2 to $3). A cheaper
   side test (under $0.20): vary the segments or the batch at equal bytes, or run `ncu` on one kernel, to separate a
   DRAM limit from a latency limit.

## Amendments

### A1 (2026-11-05, before any run; R-84 AMEND_THEN_RUN)
R-84 (`dossier/REVIEW_E-58_PRERUN_2026-11-05.md`) confirmed the fidelity (the kernel arguments, the
3D path with threshold 8, the 16-token decode tile, the C48 strides equal to the E-57 compact tensor) and found three
blockers and seven findings. No data existed. The changes:
- B1: four hook words that exclude each other (C >= 1.0 ms, G >= 0.5 ms), and the flag "explains E-57" (C >= 2.0 ms);
  before: HOOK-GPU first at G >= 0.5 ms (the top of the plausible range of 110 small kernels), and a HOOK-MIXED that
  ignored G.
- B2: a profiler error gives a kernel count of None and keeps the arm; the H NONE arm has a graph time of 0 with no
  empty capture.
- B3: V4 at 3% (10% before) after one warm round; V1 bit-identical or at most 0.002 (0.02 before).
- F1: the closure is judged on the X wall row; the CPU row is an upper bound; HOOK-CPU does not show the critical path.
- F2: the eager hook number is an event time bound by the CPU, not a GPU time.
- F3: V3 also bounds each CPU loop to half of the measured sleep; 3 steps per loop (5 before).
- F4: the KILL wording names the compact read (the byte saving is in "views").
- F5: the fidelity notes and the 3D scratch of the pinned build (Section 10 items 7 and 8), and the report-only check
  G.
- F6: the S5 time check for D32 (Section 9). F7: GO with CACHE-NONE says that the E-57 cache cost stays open.
- R-84 predictions (before data): GO about 85%; the compact read faster by 0.7 to 2.5 ms at D32; hook CPU about 1.8 ms
  (1.2 to 2.8), graph GPU 0.25 to 0.5 ms; HOOK-CPU about 75% under the A1 rule.
- Verification: 6 Mac tests pass (the words and checks of A1 included).
