# E-68: does the measured compact read keep the E-67 offload GO? (A100 read microbenchmark on Modal + CPU model; preregistration)

Status: PREDECLARED 2026-11-11. S2 R-107 AMEND_THEN_RUN, applied as A1; S3 done; S4 D-118 (we, "Approve, cap
$3.50"); S5a and S5 passed (A2 after the rehearsal, R-108); S6 attempt 1 INVALID (an A100 PCIe; V4); A3 (D-119, the
user: "Yes, amend and rerun"; R-109); S6 attempt 2 INVALID (a second A100 PCIe; A3 tie band); S6 attempt 3 (D-120,
Runpod A100 SXM, frozen rules) INVALID (V3, a first-round warm-up). CLOSED 2026-11-12: frozen word INVALID; the
exploratory part 2 (we: "Yes, run part 2") gives EXPLORATORY MEAS-CONFIG-GO, frozen word INVALID, decides
nothing (R-110 ACCEPT_WITH_CHANGES).
Idea: I-182 (v2 of the paper). Journal entry: Section 6.1 E-68. Origin: E-67 (OFFLOAD-CONFIG-GO), whose GO needs the
unmeasured read ratio 0.625; the E-65 GPU package that measures it is ready but never ran. We, 2026-11-11:
"Yes, plan it", and "If runpod is not available, I also have 30USD credits in modal".
Labels: FACT, INTERPRETATION, HYPOTHESIS. Thresholds are project heuristics unless a source is given.

## 1. Question
E-67 found that, with `offload_prompt_only=false`, D-D3 and D-M2 reach GO (1.35x and 1.40x) if the compact global
read takes 0.625 of the stock read time, and that no workload reaches GO at ratio 1.0. On one A100 80GB, with the
unchanged E-65 read microbenchmark, what is the real read time over batches 1 to 64 and contexts 1K to 128K, and with
that measured step table in the E-67 model, does any decode-node workload still reach GO? "No": no workload is GO in
`all200` (and in `prompt200`) under the measured table.

## 2. Rubber duck
- What is known (FACT): E-58 measured the 10 global-layer read at 3 x 32K on one A100 SXM: ratio 0.90 (stock 5.076 ms,
  compact 4.582 ms; bytes 0.625); at 2 x 64K, 0.97. E-59 (the engine, graphs, batch 3) agrees. Nothing is measured at
  large batch, where the decode-node workloads spend most steps (R-101 B3).
- What decides: E-67 brackets the answer between ratio 0.625 (GO on D-D3, D-M2) and 1.0 (no GO). E-65 R-101
  predicted Phi 0.10 to 0.30, so a read ratio near 0.9 at large batch is likely. Then the E-67 GO does not hold.
- Why this test and not an engine test of the offload connector: if the measured read kills the GO, an engine test of
  the connector (a 62 GB model, NIXL, more money) answers nothing that v2 needs. If the GO holds, the engine test is the
  next step (not decided). So the cheap number goes first.
- Why Modal: the Runpod balance is $2.64 (live read 2026-11-11), below the protocol floor, and six of six Secure creates
  failed for capacity on 2026-11-10. We have about $30 of Modal credits. Modal bills per second, builds the image
  on CPU (no GPU billing during the image pull), and stops a function at its timeout (a hard cost limit).
- What is new code: only the launcher (`modal_e68.py`, `e68_stage.py`) and part 2 (`e68.py`). The GPU code is the
  E-65 package with its A1 changes, byte for byte, checked by SHA-256 in the container before the session starts.
- What could make it wrong: the Modal A100 80GB can be PCIe (about 5% less DRAM bandwidth than SXM), so the V4 tie to
  E-58 can fail; Modal adds its client to the image; a preemption restarts nothing (no retries).

## 3. Hypotheses (written before any run)
- H1 (primary): MEAS-NO. The measured read ratio at B x L >= 2^18 is 0.80 to 0.95 (Phi 0.1 to 0.35), and no decode-node
  workload is GO in `all200` under the measured table. Prediction: D-D3 and D-M2 in `all200` about 1.22 to 1.30 (MID).
- H0: MEAS-CONFIG-GO. The read ratio at large batch is near the byte ratio, and D-D3 or D-M2 stays GO.
- Secondary prediction: no GO in `prompt200`; the E-65 part 2 labels (no offload) under the measured table are all MID or
  KILL (between the two bracket columns of E-65 Section 14).

## 4. Setup (frozen)

### 4.1 Hardware and provider
- Main run (S6): Modal `gpu="A100-80GB"`, 4 cores, 16 GiB memory, no region selection, no retries, function timeout
  1,500 s. List prices (modal.com/pricing, read 2026-11-11): A100 80GB $0.000694/s ($2.50/h), core $0.0000131/s, GiB
  $0.00000222/s; total $2.82/h. The GPU name and form (SXM or PCIe) are recorded (`nvidia-smi -q`, hardware record).
- Rehearsal (S5): Modal `gpu="A10"` (Ampere, 24 GB), same CPU and memory, timeout 900 s; $1.42/h.
- Two CPU-only checks before any GPU (S5a): `probe` (the image builds; `python` is on PATH; vLLM 0.33.0rc1.dev229+g3966924d5
  imports; the 13 code hashes match) and `timeout` (a 60 s function that sleeps 120 s must be stopped by Modal).
- Provider rules that replace RUNPOD.md for this run: the Modal timeout is the boot watchdog; the E-65 session watchdog
  (POD_LIMIT 900 s main, 600 s rehearsal) ends the bench first; there is no SSH and no Pod; results come back as the
  function return value, with a copy in the Modal volume `msthesis-e68-results` (registered in `pods.jsonl`, kind
  `modal-volume`). The Modal app is named `msthesis-e68-readratio`.

### 4.2 Software
- Image `vllm/vllm-openai@sha256:f255c4fad2f30ff5ee80db14d3a03afe9ae0f10d3470339864ff04745fe44cb3` (vLLM
  0.33.0rc1.dev229+g3966924d5, torch 2.14.0+cu130, Triton 3.8.0), the image of E-49 to E-59, with the ENTRYPOINT
  removed and a `python` link if missing. Modal client 1.6.3 (PyPI 2026-10-27, Apache-2.0) in an isolated uv
  environment on the Mac. No model download.
- GPU code: the E-65 package (`bench_e65.py`, `analyze_e65.py`, `session_e65.sh` and the files they import), unchanged
  since the E-65 freeze except `poolsim.py`, which E-66 and E-67 extended with fields and modes that default off
  (`test_e66.py` V1). Frozen by `code_manifest_e68.json` (13 SHA-256 values of the working files, 2026-11-11).

### 4.3 The GPU part
Exactly E-65 Sections 4.3 and 4.4 with amendment A1 (README.md): the tie point (3, 32K) first with V4 at once; 33 core
points (L rows 1K to 128K, B x L <= 2^19) by tokens; the extra points (16, 64K) and (8, 128K) last; G (5 global
layers, S32 and C48, CUDA graphs, 3 rounds of 50 replays), SL (5 sliding layers with the engine window), H (the hook
transform); `analyze_e65.py` writes `fit.json` (the W-M table) with the E-65 word VALID or INVALID and Phi.

### 4.4 Part 2 (the Mac, free)
- `e65_model.py` with the measured `fit.json`: the E-64 grid with no offload (10 workloads; the `none` labels).
- `e68.py`: the E-67 decode-node grid (6 workloads, pools, seeds 0 to 4, horizon, warm-up, filter, link arms, link
  guard) in modes `prompt200` and `all200`, timed with three tables: the two E-65 brackets (r1, r0625: they reproduce
  E-67) and the measured table (meas). The E-67 code runs unchanged; e68.py adds the third tag only.
- One core at nice 10 (`run_e68.sh`), about 3 hours in all.

## 5. Metrics
- Primary (decides): the E-67 workload label in `all200` at `L20` under the measured table (Q_M, 5-seed rule, the
  E-65 A1 gate of M, M_pf_equal and M_gemm).
- Secondary (reported): the label in `prompt200`; the read ratio G_C / G_S per grid point and Phi; the E-65 part 2
  labels with no offload under the measured table (they complete the E-65 question); the bracket columns (they must
  equal E-67); hit rates; link use.
- MDE and repeats: as E-65 (a read difference below 2% is no change; 3 rounds per layout and point; the A/A is the round
  spread, V3) and E-67 (5 seeds).

## 6. Validity checks (a failed check makes the run INVALID)
- V1 to V5 of E-65 (A1): the layouts give the same output; the GPU is an A100 and the vLLM build is the pinned one; round
  spreads; the tie to E-58 within 5%; every core point ran.
- V7: the 13 code hashes in the container equal `code_manifest_e68.json` (checked before the session; a difference stops
  the function before the bench).
- V8: the bracket columns of part 2 equal the E-67 labels (`results/e67/`) for every workload and mode.
- Report only: the GPU form (SXM or PCIe), telemetry, the Modal region and cloud, the E-65 V6 hook tie.

## 7. Decision rule (frozen; project heuristics)
- INVALID: the E-65 GPU word is INVALID, or V7 or V8 fails. Part 2 does not decide.
- At `L20` under the measured table:
  - MEAS-DEFAULT-GO if any decode-node workload is GO in `prompt200`;
  - MEAS-CONFIG-GO if none is GO in `prompt200` and any is GO in `all200`;
  - MEAS-NO if none is GO in either.
- What v2 may say: MEAS-NO, "with the read time measured on one A100, the offload GO of E-67 does not hold; the best
  decode-node workload is <x> (MID)". MEAS-CONFIG-GO, "the GO holds in a model with read times measured on one A100;
  the connector on Gemma 4 and the compact layout is untested". Either way it stays a model result for throughput.
- Next step (a recommendation, not a decision): after MEAS-CONFIG-GO, an engine test of the offload connector on the
  compact build; after MEAS-NO, no further GPU work on the offload question.

## 8. Controls and baselines
- Baseline: S32 at every point (E-65). A/A: the round spread (V3).
- Positive control: V4 reproduces E-58 at its shape; the E-65 `compact_fast` plant test.
- Negative control: V1 (same logical K and V, same output).
- Part 2 control: V8 (the bracket tags reproduce E-67; `test_bracket_tables_reproduce_e67_cell`).

## 9. Budget
- Caps (the Modal timeout times the list price): probe and timeout checks about $0.02 (CPU); rehearsal $0.36 (900 s);
  main $1.17 (1,500 s). Total hard cap $1.55; the approval asks for $2.00 of Modal credits, with the margin for the image
  build on Modal CPU. Plan (INTERPRETATION, from E-58 and the E-65 plan): container start 1 to 3 min, Triton compile about
  2 min, bench 5 to 8 min; main about 12 min, about $0.56; rehearsal about 8 min, about $0.19.
- Order: probe, timeout check, rehearsal, main, then part 2 on the Mac. Each GPU stage needs the one before it to pass.
- Balance: Modal has no balance tool in this repo. Before each GPU stage, we read the remaining credits on the
  Modal usage page; a GPU stage starts only if the credits exceed its cap. Billing after the run: the same page.
- What the time cap cuts first: the extra points (reported as extrapolated), then the largest core points (V5 fails).

## 10. Threats to validity
1. All of E-65 Section 10 (isolated reads, T_OTHER fixed over B, mean context, synthetic values, sliding layers, the
   prefill fit from one host, one GPU type, the E-64 limits, not blind).
2. A different provider and host than E-58 and E-59. V4 checks the offset at the tie. If the GPU is PCIe, the reads are
   slower in both layouts; the ratio is less sensitive than the absolute time, but V4 decides.
3. Modal adds its client to the image. The bench does not use the vLLM server; V2 checks the vLLM version; the hardware
   record keeps `pip freeze`.
4. All of E-67 Section 8 and R-106 (a model, not the engine; the connector untested; async load and store latency not
   modelled; the link guard uses the time of the r0625 table, which is shorter than the measured one, so it is
   conservative).
5. A preemption of the Modal container: the call fails and is not retried; a new attempt needs a new approval.

## 11. Files
| File | Role |
|---|---|
| `modal_e68.py` | the Modal app: `probe`, `timeout`, `reh`, `main` (imports `modal`) |
| `e68_stage.py` | the launcher parts with no `modal` import: file list, image build, the session call, the stub, pack and unpack, caps |
| `code_manifest_e68.json` | the 13 frozen SHA-256 values |
| `e68.py`, `run_e68.sh` | part 2 on the Mac and the word |
| `test_e68.py` | the S3 tests |
| E-65 files | the GPU code (README.md Section 11), unchanged |

## 12. Verification record (stage S3)
- Tests (2026-11-11): `uv run --no-project --with numpy --with torch --with pytest python -B -m pytest -q -p
  no:cacheprovider experiments/gemma4_kv_decode_scaling/ experiments/gemma4_kv_capacity_model/` gives 55 passed (10 new):
  every mounted file exists; the manifest differs from E-65 only in `poolsim.py`; the caps; the stub session end to end
  (the manifest check, the unchanged E-65 session with STUB=1, the REHEARSAL line with no failed check, the pack, the
  unpack and the hashes); a changed code file stops the function before the session; unpack refuses a path outside the
  folder and an existing folder; part 2 on the stub table; one E-67 cell with three tags equals the cell with two; the
  word rule; V8 against the stored E-67 labels (a changed label fails it); an INVALID GPU word stops part 2.
- Not testable on the Mac (VERIFY in S5a and S5): the Modal image build and the `python` link; the vLLM import in the
  container; the Modal timeout; the A10 rehearsal items of E-65 Section 9 (the Triton kernels, the graph capture, the 2D
  path, the round spread, the times), the return of the results and the volume copy.

## Amendments

### A1 (2026-11-11, after R-107, before any run)
R-107 (reviewer; `dossier/REVIEW_E-68_PRERUN_2026-11-11.md`): AMEND_THEN_RUN, 4 blockers and 7 findings.
The changes below replace the text of Sections 3 to 10 where they differ. No data existed.
- B1: the image did not hold `e68_stage.py` (Modal 1.0 does not add local modules), so every function would fail at
  start. It is now added (`add_local_python_source`). The container records the SHA-256 of `modal_e68.py` and
  `e68_stage.py`; the Mac compares them with its own (`launcher_sha_match`).
- B2: the cost cap was not hard. Each function now has `startup_timeout` 300 s, `scaledown_window` 2 s and
  `retries` 0; the A100 limit is 1,200 s (not 1,500 s). A preemption can restart the input on the same call (Modal
  docs), so the session refuses a second attempt in the same run folder (a `started.txt` is committed to the volume
  first) and the worst case counts two attempts. New caps (list prices; `e68_stage.cost_cap`, `worst_case`):
  rehearsal $0.47 per attempt, worst $0.95; main $1.17 per attempt, worst $2.35; probe and timeout checks about
  $0.02; the image build on Modal CPU, estimate at most $0.10. Worst case in all: $3.42. Expected: about $0.85. This
  replaces Section 9 and Section 10.5.
- B3: H1 contradicted its own rule. R-107 interpolated the stored E-67 cells at a uniform read ratio r (exact for the
  planted tables) and found, in `all200`: D-D3 GO to r 0.85 (5 of 5 seeds), MID at 0.875 (4 of 5) and above; D-M2 GO
  to r 0.80, MID with 4 of 5 seeds GO from 0.85 to 0.95, 2 of 5 at 1.0; `prompt200` never GO for r in 0.625 to 1.0.
  New H1: the median read ratio over the points with B x L >= 2^18 is at least 0.88, and the word is MEAS-BORDERLINE or
  MEAS-NO. New H0: the median read ratio is at most 0.85, and the word is MEAS-CONFIG-GO. A measured table is not
  uniform in r, so these thresholds are a guide; the word decides.
  New word (Section 7), between MEAS-CONFIG-GO and MEAS-NO: MEAS-BORDERLINE if no workload is GO and at least one
  `all200` workload has 4 or more of 5 seeds GO. A median of Q_M at least 1.25 alone does not make it BORDERLINE,
  because D-M2 has a median of 1.255 even at r = 1.0 (no read saving). What v2 may say for MEAS-BORDERLINE: "within
  the seed noise of GO; no GO is claimed". MEAS-DEFAULT-GO is expected to be unreachable (it needs a read below the
  byte ratio), so it carries no weight. The per-seed GO counts and the median read ratio are printed with the word.
- B4: the protocol deviations need a decision by us (journal D-118): Modal in place of Runpod for this run; no
  balance tool, so before each GPU stage we read the remaining credits on the Modal usage page and the number
  goes into the run record; the $35 floor of abort rule 5 is replaced by "a GPU stage starts only if the credits are
  at least 2 x its worst case". The Modal Starter plan has no budget setting (modal.com/pricing: environment-level
  budgets are not in Starter), so the function limits are the only cap. `pods.py add` takes any kind text, so the
  volume is registered with kind `modal-volume`. The design is frozen by a commit before S4 if we approve it;
  until then by the SHA-256 values in `code_manifest_e68.json` (GPU code) and this README.
- F1: V4 stays as frozen in `analyze_e65.py` (not changed). If the GPU is a PCIe A100 and V4 fails, the word is
  INVALID; the tie runs first, so this stops the run after a few minutes. Reported with it: the form, the tie ratio
  C48 / S32 against the E-58 ratio 0.903.
- F2: the probe also checks `pip`, and the torch (2.14.0+cu130) and Triton (3.8.0) versions; all are part of its pass
  test. If the image fails Modal's Python rule, do not use `add_python`; stop and ask.
- F3: the stub tests run on a tree that holds only the mounted files.
- F4: `ok` is true only if the bench exited 0 and the GPU word is VALID or REHEARSAL; the bench exit line is recorded.
- F5: the launcher prints the size of the returned results; if the call fails after the GPU ran, read the volume copy
  (`modal volume get msthesis-e68-results <run id>`).
- F6: V8 compares the medians within 1e-12; `run_e68.sh` prints the median read ratio first (report only).
- F7: the A10 rehearsal checks mechanics only (sm_86, 24 GB). Memory at the largest A100 points is not rehearsed; at
  2^19 tokens two layouts use about 43 GB of 80 GB.
- Not changed: the question, the GPU code and its rules (V1 to V5), the E-67 grid and gate, the 5-seed rule, the order
  of the stages.
- Verification after A1: 56 tests pass (11 for E-68): the mounted-tree stub session, the restart refusal, the
  BORDERLINE word and the seed counts, the read-ratio report, the caps.

### S5 record (2026-11-11)
- S5a probe (CPU, Modal app <modal-app-id>): the image built in 167 s on Modal CPU; Python 3.12.3, pip
  26.2.2, vLLM 0.33.0rc1.dev229+g3966924d5, torch 2.14.0+cu130, Triton 3.8.0; 13 of 13 code hashes match; the
  launcher hashes match the Mac. PASS.
- S5a timeout check (<modal-app-id>): the 60 s function that sleeps 120 s was stopped with
  FunctionTimeoutError after 70 s on the Mac clock. PASS.
- S5 rehearsal (one NVIDIA A10, <modal-app-id>, run e68-reh-20261111T215042Z): the session ran in 63 s, the
  call in 76 s; 8 of 8 points ran with no error (the 3D path, the 2D path at B 16 to 64, the one-after-the-other path);
  V1, V2, V4 (rehearsal), V5 pass; the CUDA graphs captured; the results came back (19,470 bytes) with the volume copy;
  the word is REHEARSAL. VERIFY item 5 (round spread) FAILED: V3 at (32, 4K, H) 5.5%, (4, 32K, SL) 8.3%, (2, 64K, SL)
  5.2%; every G spread is at most 2.3%. FACT (telemetry): every busy sample has throttle reason 0x4 (software power cap;
  the A10 limit is 150 W, power up to 153 W); the SL rounds fall monotonically (0.914, 0.879, 0.844 ms). Report only:
  the A10 read ratio is 0.64 to 0.68 at the points with 8 or more requests or 16K or more tokens per request (the A10
  has about 600 GB/s); this says nothing about the A100.

### A2 (2026-11-11, after the S5 rehearsal, before the main run; no A100 data exists)
- Change: an SL or H round spread fails V3 only if its effect on the modelled stock step at that point, the layer count
  (10 SL, 2 H) x (max - min of the 3 rounds) / t_stock, is at least 2% (the MEASUREMENT.md MDE for speed). Reason: SL is
  the same in both layouts and H adds 2 x about 0.2 ms, so a spread below 2% of the step cannot move a ratio by more
  than the MDE; the rehearsal failure came from the A10 power cap. The G spread rule (3%, the layouts that decide) is
  not changed. On the rehearsal data, the three failures have step effects of 1.2%, 0.5% and 0.03% (R-108 values).
- How: the frozen GPU code and `fit.json` do not change (the A100 run uses the same 13 hashes). On the Mac,
  `e68.py --a2` writes `part2/fit_a2.json` with the recomputed word (VALID if no other check fails); part 2 uses it.
  The A2 record lists each waived and kept point.
- Not changed: every other check, the word rule, the caps. Tests: 58 pass (2 new: A2 on the rehearsal data; A2 keeps a
  G failure and an SL spread of 4% of the step).
- R-108 (reviewer; `dossier/REVIEW_E-68_A2_2026-11-11.md`): ACCEPT_WITH_CHANGES. A2 is a loosening
  after an A10 power cap, not a required fix; it is allowed only because no A100 data exists. The 2% step bound was
  too loose (H on its own is about 0.2 ms, and the tie SL range enters T_OTHER at every point); MEASUREMENT.md asks for
  half the MDE. Applied (this replaces the first bullet of A2): the bound is on the decision quantity, the step ratio
  rho = t_c / t_s at the point. SL is timed once and enters both layouts, and T_OTHER takes -10 SL(tie), so an SL error
  moves only the denominator: bound = (|t_s - t_c| / t_s) x 10 (range SL(point) + range SL(tie)) / t_s. H enters only
  the compact step: bound = 2 range H / t_s, and 2 range H must also be below 25% of |t_s - t_c|. A spread is waived if
  its bound is below 1%. G is never waived. A report-only line gives, per sequential point, the last-over-first round
  ratio of G_S32 and G_C48 (drift between the two blocks). On the rehearsal data the bounds are 0.03%, 0.15% and 0.06%
  (all waived). The frozen in-container record has `ok` false when only V3 fails on SL or H; the A2 file decides.
  `run_e68.sh` stops if the A2 step fails. Tests: 58 pass.

### S6 attempt 1 (2026-11-11 22:00 UTC, Modal app run e68-main-20261111T215927Z, design commit 8663bb9): INVALID
- FACT: the GPU was an NVIDIA A100 80GB PCIe (`gpu.csv`: power limit 300 W, max SM clock 1,410 MHz, PCIe gen 4 x16),
  not the SXM of E-58 and E-59. Modal does not let a function ask for the SXM form of the A100.
- FACT: the tie (3, 32K) ran twice (the A1 B4 rule) and V4 failed both times: 2 G_S 5.377 ms and 2 G_C 4.754 ms
  against the E-58 values 5.076 and 4.582 (+5.9% and +3.8%). The G round spread was 3.8% in try 2 (round 1 slow:
  2.790, 2.687, 2.688 ms), so V3 also failed at the tie. The bench stopped after the tie, as designed. Session 56 s,
  call 124 s; estimate $0.10.
- FACT (telemetry): busy samples at 286 to 323 W, SM clock 1,335 to 1,410 MHz, throttle reason 0x4 (software power cap)
  in 3 of 5 busy samples.
- Report only (not a decision): the tie read ratio C48 / S32 is 0.884 (medians 2.375 / 2.688 ms); E-58 gave 0.903. V1,
  V2 and the hook tie (0.393 against 0.396 ms) pass.
- Word: INVALID (V3, V4, V5). No part 2. The decision rule is not changed by this record.

### A3 (2026-11-11, after S6 attempt 1; we: "Yes, amend and rerun")
The rules below changed after the attempt-1 tie data was seen. The decision points (B x L >= 2^18) have no data yet.
By PROTOCOL.md principle 1, the result of the rerun is labelled EXPLORATORY and cross-host (A100 PCIe); the word
carries this label.
- V4 by ratio: the tie read ratio C48 / S32 (medians of the 3 rounds) within 5% of the E-58 ratio 0.9027 (4.582 /
  5.076). The absolute offset is reported (attempt 1: +5.9% and +3.8%, the PCIe bandwidth and power limit). In the bench,
  `tie_ok` uses this rule only when the environment sets `E68_TIE_RULE=ratio` (the E-68 launcher sets it); the E-65
  default is not changed. This is the one change to the GPU code: `code_manifest_e68.json` now differs from the E-65
  manifest in `bench_e65.py` and `poolsim.py`.
- V3 for G: a point that fails the 3% round spread is waived if, in each layout that fails, round 1 is the slowest and
  rounds 2 and 3 agree within 1% (a warm-up effect; the W-M table uses the median of the 3 rounds). Any other G spread
  still makes the run INVALID.
- Where: `e68.py --amend` applies A2, then A3, to the frozen `fit.json` and writes `part2/fit_amended.json`;
  `run_e68.sh` uses it. The frozen in-container word stays as `analyze_e65.py` computes it.
- Known bias (report only): T_OTHER = the E-59 SXM step at the tie minus the PCIe attention at the tie. The PCIe reads
  are about 0.6 ms slower at the tie, so T_OTHER is about 1.2% low; a lower T_OTHER raises the KV share of the step and
  the compact gain a little (E-65 R-102 item 4). The word is read with this in mind.
- On the attempt-1 data: the ratio V4 passes (0.884, -2.1%); the tie G spread is waived (S32 2.790, 2.687, 2.688 ms);
  V5 still fails (no core points), so attempt 1 stays INVALID.
- Not changed: V1, V2, V5, the A2 rule for SL and H, the word rule, the caps. Tests: 63 pass (5 new).
- R-109 (reviewer; `dossier/REVIEW_E-68_A3_2026-11-11.md`): ACCEPT_WITH_CHANGES. Applied (they replace
  the bullets above where they differ):
  - V4: the ratio within 3% (not 5%) of 0.9027, and each read within 10% of its E-58 value. Reason: a 5% band on the
    ratio is a band of about 45% on the read gain (E-58 9.7%, attempt 1 11.6%), and a uniform slowdown passes a ratio
    test alone. Attempt 1 passes (ratio -2.1%; reads +5.9% and +3.8%).
  - The G waiver applies only to the tie and to interleaved points (never to a sequential point at 2^19 or more tokens,
    where S32 and C48 are timed in separate blocks), and to at most 3 points; more makes the run INVALID.
  - Bias (report only, corrected): T_OTHER is about 0.42 ms (0.8%) low (the attempt-1 tie attention 10.94 ms against
    the planted E-58 and E-59 tie). A second bias goes the same way: the PCIe read gain at the tie (11.6%) is about 20%
    larger than the E-58 SXM gain (9.7%). Both favour the compact layout, so a GO under A3 is weaker evidence than a
    MEAS-NO.
  - The bench records `tie_rule` in `bench.json` meta; a test shows that the launcher flag reaches the bench.
  - Protocol status: A3 is a principle-1 amendment after data, not a fix that the rehearsal required. The changed
    `tie_ok` has no GPU rehearsal; it ran only in the Mac stub and in unit tests. We decided the rerun (D-119).
  - Tests: 64 pass in the project (19 in `test_e68.py`).

### S6 attempt 2 (2026-11-11 22:44 UTC, run e68-main-20261111T224104Z, design commit 604ecc2, A3): INVALID
- FACT: a different NVIDIA A100 80GB PCIe host (UUID GPU-e1d9ab3d-..., 300 W); `bench.json` meta `tie_rule` is `ratio`
  (the flag reached the bench).
- FACT: the ratio tie failed twice: try 1 ratio 0.8662 (-4.0% against 0.9027), reads +5.2% and +0.9%; try 2 ratio
  0.8696 (-3.7%), reads +5.1% and +1.3%. The R-109 band is 3%, so the bench stopped after the tie, as frozen. Session
  36 s, call 237 s; estimate $0.19.
- This is abort rule 3 (the reference fails twice). The rule is not amended again after this data.
- Report only: on two A100 PCIe hosts the tie read ratio is 0.866 to 0.884, below the E-58 SXM value 0.903. The compact
  read gain at 3 x 32K is larger on these PCIe hosts (12% to 13%) than on the SXM (10%). This says nothing about large
  batch.

### S6 attempt 3 plan (D-120, we: "Run on Runpod SXM")
- One Runpod A100 SXM 80GB, Secure (live read 2026-11-11: stock LOW in US-KS-2, CUDA 13.2, $1.79/h), the E-65 image by
  digest, container disk 30 GB, volume 20 GB at /workspace, ports 22/tcp, CUDA floor 13.0, boot watchdog 1,500 s
  (`pod_cmd.py --cap 1500`; cap $0.75), the E-65 session with AUTO_STOP=1 (POD_LIMIT 900 s, AUTO_STOP_DELAY 90 s).
- `run_pod_e68.sh` uploads the 13 files and `code_manifest_e68.json` and checks the hashes on the Pod. No
  `E68_TIE_RULE`, so the bench uses the frozen absolute tie of A1. On the Mac, `e68.py --amend` applies A2 only (A3
  applies only to a bench with `tie_rule` ratio), so the word is not labelled exploratory.
- The Runpod balance ($2.64) is below the $35 floor; D-120 accepts this as D-116 did. If the create fails for capacity,
  E-68 stops at INVALID.
- Tests: 65 pass (1 new: A3 applies only to the ratio tie).

### S6 attempt 3 record (in progress)
- Balance (live, `balance.py --need 0.75 --hours 0.42`): $2.64, no other spend running; the floor check fails, as D-120
  accepts. SSH key `runpod-masters-thesis` registered.
- Pod `<pod-id>` (`msthesis-e-68-main-20261111`, A100 SXM 80GB, US-WA-1, Secure, $1.79/h, CUDA 13.2) created
  23:06:43 UTC and registered in `pods.jsonl` at once.
- Incident (no blame, facts): the SSH wait script polled `rest.runpod.io/v1/pods/<id>` with the read-only key and got
  an HTTP error on every call, so it never saw SSH. The Pod waited about 10 minutes with no session (about $0.30).
  A `get-pod` MCP read showed direct SSH at 195.26.233.70:52244 (the image pull took about 6 minutes). Next time: wait
  with `get-pod` reads, not the REST v1 endpoint with the read key.
- `run_pod_e68.sh`: the upload worked and 13 of 13 hashes matched on the Pod; the session start command ran; the
  check 15 s later could not connect (SSH timed out).
- A `get-pod` read then showed the Pod EXITED (INTERPRETATION: AUTO_STOP stopped it 90 s after the session ended, so
  the session was short; the bench may have stopped at the tie). A `start` returned HTTP 400: "There are not enough
  free GPUs on the host machine to start this pod." The results are on the Pod's volume disk.
- One use of `RUNPOD_API_KEY_WRITE` (we: "Yes, use it once"): GraphQL `podResume` with `gpuCount: 0`. Refused with
  the same message. INTERPRETATION: the host-local volume can be read only on its own host, and this host refuses
  even a 0-GPU start while its GPUs are rented. The Pod stays EXITED (disk billing only, about $4 a month for 20 GB).
- 15 normal starts failed for capacity between 2026-11-11 23:30 and 2026-11-13 00:25 UTC (one 30-minute retry loop,
  paused once by a usage limit). The start at 00:56:54 UTC 2026-11-13 worked (price now $2.19/h). Direct SSH was not
  assigned, so the results were copied with the `exec-pod` tool: a tar.gz of `/workspace/results/e65` (26,056 bytes,
  27 files) as base64; the SHA-256 on the Mac equals the Pod value (f272464b...). The Pod was deleted after the copy
  (`pods.py end`). Billing read: $0.57 so far ($0.43 GPU, $0.14 disk); the restart minutes are not yet posted. Runpod
  balance after: $1.81 (live read), no spend running.

### S6 attempt 3 result (frozen rules: A1 and A2; no A3)
- FACT: NVIDIA A100-SXM4-80GB (400 W), vLLM 0.33.0rc1.dev229+g3966924d5, `tie_rule` absolute. The tie passed at the first
  try: 2 G_S 5.049 ms, 2 G_C 4.564 ms against the E-58 values 5.076 and 4.582 (V4 PASS). The bench ran all 35 points
  in 124 s (V5 PASS); V1 and V2 PASS.
- FACT: V3 FAILED at 6 G points: (3, 32K), (8, 16K), (64, 4K), (8, 32K), (8, 64K), (4, 128K), with G spreads of 3.1%
  to 11.1%. At each point round 1 of the S32 read is the slowest and rounds 2 and 3 agree within 0.2% (for example
  (8, 64K): 14.702, 14.155, 14.151 ms); at the tie the C48 read shows the same pattern. A2 waived the one SL point
  (bound 0.03%). A3 does not apply (absolute tie, D-120).
- **Frozen word: INVALID (V3).** Part 2 does not decide.
- Report only (EXPLORATORY; not a decision): the read ratio G_C / G_S. Median over the 11 points with B x L >= 2^18:
  0.948 (range 0.833 to 1.084); Phi 0.121. In the 3D path with long contexts (B 4 to 8, L 32K to 128K) the ratio is
  0.83 to 0.87; in the 2D path at B 32 and 64 the compact read is slower (1.05 to 1.10). By the R-107 interpolation of
  the stored E-67 cells (a uniform ratio), a ratio near 0.95 gives no GO in `all200` (D-D3 0 of 5 seeds GO, D-M2 4 of 5),
  that is MEAS-BORDERLINE territory; the measured table is not uniform, so this is a guide only.
  (Correction, R-110 F7: this interpolation is true for the bracket tables only. It does not transfer to the measured
  structure; see the exploratory part 2 below.)
- H1 (median ratio at least 0.88, word BORDERLINE or NO) agrees with these exploratory numbers; it is not tested by a
  valid run. (Correction, R-110 F1: the ratio condition holds, but the exploratory word is MEAS-CONFIG-GO, so the H1
  word does not agree. See below.)

### Exploratory part 2 (2026-11-12; we: "Yes, run part 2")
- Method: the frozen path (`run_e68.sh`, `e68.load`, `e65_model.main`) refuses part 2 on an INVALID GPU word. The new
  `explore_e68.py` (with `run_explore_e68.sh`; 4 tests in `test_explore_e68.py`) calls the same part 2 code on the same
  fit (`fit_amended.json`) and skips only the GPU-word gate. Each output is labelled EXPLORATORY and keeps the frozen
  word. Mac, one core, nice 10, about 3 hours, no spend. It ran once; nothing was tuned after the output. It is not
  blind: we saw the read ratio and the R-107 forecast before the run. Results:
  `results/e68-main-sxm-20261111/part2_exploratory/`.
- FACT: **EXPLORATORY MEAS-CONFIG-GO, frozen word INVALID, decides nothing.** V8 passed (the bracket columns give the
  E-67 labels and medians). `all200`, L20, measured table: D-D3 GO 1.303 (5 of 5 seeds GO), D-M2 GO 1.375 (5 of 5).
  `prompt200`: no GO (best D-D3 1.257, 3 of 5 seeds). No offload under the measured table: D-D3 MID 1.107, D-M2 MID
  1.159. Link use at most 7.4%.
- R-110 (reviewer, ACCEPT_WITH_CHANGES; `dossier/REVIEW_E-68_POSTRUN_2026-11-12.md`; scripts in `review_r110/`):
  - V3 is a warm-up effect. The fit uses the median of 3 rounds; dropping round 1 moves G by at most 0.07% and the
    read ratio by at most 0.1 points.
  - Mechanism of the conflict with H1 (FACT, from reruns): mostly not the read ratio. On the measured structure a
    uniform ratio of 0.95 still gives GO, and the break-even uniform ratio is about 1.0 (D-D3 3 of 5, D-M2 4 of 5).
    The bracket tables plant SL linear in B (1.41 ms at B 8, 10.93 ms at B 64); the measured SL is much flatter (0.73
    ms at B 8, 16K; 3.3 ms at B 64, 1K). The compact layout runs a larger batch, so the bracket charged it too much.
    H1 and the R-107 threshold 0.85 were set on the planted structure.
  - The read ratio is also non-uniform, and the workloads sit where it is low: D-D3 and D-M2 spend 89% to 96% of
    stock time at B 5 to 6 and mean context 44K to 58K. The time-weighted read ratio is 0.85 to 0.88 (report only;
    R-110 F6). The 0.948 median is an unweighted median over 11 points, and it includes 2D-path points that these
    workloads never visit. Do not use 0.948 to explain the result.
  - Margins: the lowest per-seed gate value (min of M, M_pf_equal, M_gemm) is 1.291 for D-D3 and 1.295 for D-M2 (+3.3%
    and +3.6%). A compact read 6% slower, a stock read 3% faster, or H doubled keeps 5 of 5 GO. The GO flips only if
    the compact read is about 14% slower than measured in the used states.
  - Model limits (R-110 F5): T_OTHER is fixed over B; the step uses the mean context, and the report-only M_max arm
    (longest context) gives D-D3 1.235 (MID) and D-M2 1.330; SL and G are isolated kernels on random values; the
    offload connector, async load and store, and the 62 GB model are not measured; one GPU, one session, one host.
- Allowed claim (R-110 Q5): an exploratory model result, 1.30x (D-D3) and 1.37x (D-M2) with CPU offload of all
  preempted requests (`offload_prompt_only=false`), 5 of 5 seeds, with the V3 failure and the model limits in the same
  paragraph. Do not claim a confirmed GO, a measured vLLM speed-up, or "MEAS-CONFIG-GO" without the EXPLORATORY label.
- A confirmatory number needs a new preregistration (E-69), not an amendment: a warm-up phase or a discarded round 1,
  a step-time-weighted ratio or the model Q as the statistic, and kill criteria that name the break-even ratio of
  about 1.0. Do not loosen V3 on this data. A rerun needs a decision by us (not decided).
- D-121 (we, 2026-11-12, "Yes, record D-121 and commit"): keep this as an exploratory model result and spend
  nothing now. When GPU money comes, one Pod session runs the engine test of the offload connector and the E-69
  warm-up rerun.
