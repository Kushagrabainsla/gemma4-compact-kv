# E-63: how much of the compact K=V saving becomes serving capacity (CPU block-pool model, I-182 v2)

Status: PREDECLARED 2026-11-10 (this README, before any run of `sweep.py`). Free: Mac CPU, no GPU, no spend.
Labels: FACT, INTERPRETATION, HYPOTHESIS. Project heuristics are labelled as heuristics.

## 1. Question
The compact K=V cache cuts the steady decode footprint of one Gemma 4 31B request by 1.338 times at 32K and 1.404
times at 64K (E-57, measured). In E-57 the decode batch rose only 3 to 4 at 32K and stayed 2 at 64K, in one 15 GiB
pool (reviewer weakness W3 of R-90 to R-92: "an integer effect at one pool size"). Over realistic pool sizes,
context lengths and mixed request lengths, how much of the footprint saving becomes more requests that decode at
the same time, under the vLLM admission rule?

## 2. Rubber duck
- What sets capacity in vLLM (FACT, E-57 README Section 2, `scheduler_reserve_full_isl` on by default): a waiting
  request is admitted only if its full prefill reservation fits the free blocks now. The reservation is the global
  blocks of the prompt plus at most 321 blocks per sliding group (window 1,024 plus two chunks of 2,048 tokens in
  flight). A decoding request holds its global blocks plus 65 blocks per sliding group. So b requests decode together
  if b - 1 steady footprints plus one reservation fit the pool. The reservation is about 1,605 sliding blocks in both
  layouts, so the admission ratio is smaller than the footprint ratio.
- Why one pool is not enough: b = 1 + floor((pool - reservation) / steady) is a step function of the pool. A
  different pool moves the steps. A sweep shows the average effect.
- Why mixed lengths matter: with mixed lengths, the free blocks that one long request cannot use can admit a shorter
  one. The FCFS head-of-line rule (the scheduler stops at the first request that does not fit) can block that.
- What the model leaves out: step time (so no tokens per second), prefix caching, FP8, CUDA graphs, tensor
  parallelism, speculative decoding. The claim is about concurrency only.
- What could make the model wrong: a different admission or preemption rule in vLLM main. The model copies the rules
  of vLLM 3966924d5 that E-57 ran, and V1 ties it to the E-57 measurement.

## 3. Hypotheses
- H-FIX (HYPOTHESIS): with fixed lengths, the concurrency ratio compact / stock averages about 1.25 to 1.30 over pools
  of 15 to 60 GiB at 16K to 128K, below the footprint ratio because of the reservation.
- H-MIX (HYPOTHESIS): with mixed lengths, the mean decode concurrency ratio is closer to the footprint ratio of the
  workload, because short requests fill the gaps, unless head-of-line blocking stops them.

## 4. Setup (frozen)
- Model of the engine (`poolsim.py`), the rules of `run_e57.StubEngine` (the gate model that predicted all four E-57
  batches), plus decode growth and preemption:
  - The pool: `blocks` usable blocks of 2,621,440 B (one page of each of the 10 layers of a group). E-57: 6,143.
  - Groups: 1 global group (block 32 tokens stock, 48 compact; the B3 plan) and 5 sliding groups (block 16, window
    1,024).
  - A step: budget 2,048 tokens; decodes first (1 token each), then running prefills, then new admissions in FCFS order.
  - Admission: a waiting request is admitted only if free blocks >= reservation(prompt). The first waiting request
    that does not fit stops the admission loop (FCFS head-of-line rule).
  - Held blocks: ceil(c / global block) + 5 x min(ceil(c / 16), ceil((1,023 + in flight) / 16) + 1), where in flight
    is the sum of the last two chunks in prefill and 1 in decode (the StubEngine `held`).
  - Growth and preemption: if held blocks exceed the pool after a step, the most recently admitted running request
    is preempted (recompute: it goes back to the head of the waiting queue, with prompt = prompt + output so far,
    c = 0). This is the vLLM V1 FCFS rule (the last running request is the victim).
  - A request finishes after its output tokens.
- Layouts: STOCK and COMPACT (BF16 KV, the B3 block plan, the same pool). Only the global block differs.
- Fixed-length grid (FIX): pool blocks on a log grid of 25 points from 2,000 to 40,000 (about 4.9 to 98 GiB); prompt
  L in {8K, 16K, 32K, 64K, 128K} tokens; output 256 tokens. The concurrency b is computed from the rule above and
  checked by a short simulation for every cell.
- Mixed workloads (MIX), each a seeded backlog of 160 requests (seed 63), all waiting at time 0 (saturated):
  - M1 long-mixed: prompt lognormal, median 16K, sigma 1.0; output lognormal, median 300, sigma 0.8.
  - M2 agent: prompt lognormal, median 48K, sigma 0.6; output median 500, sigma 0.8.
  - M3 bimodal: 70% prompts uniform 500 to 4K, 30% uniform 32K to 128K; output median 300, sigma 0.8.
  - M4 short (control): prompt lognormal, median 2K, sigma 0.8; output median 300, sigma 0.8.
  - Prompts are clipped to [256, 131,072], outputs to [16, 4,096].
  - Requests whose STOCK reservation of prompt + output does not fit the empty pool are removed from the backlog of
    that pool, for both layouts (the same set), so no request can wait forever. The count is reported.
  - Pools: 9 points of the same log grid from 4,000 to 32,000 blocks.
- Measurement window (MIX): from the first step to the step where the waiting queue first becomes empty (the backlog
  is no longer saturated). Metric: the mean number of decoding requests per step in the window.
- Reference pools (ASSUMPTION, for the figure only): E-57 6,143 blocks (15 GiB, A100 80GB, BF16 weights); about 16,000
  blocks (about 40 GiB, an 80 GB GPU with FP8 weights); about 40,000 blocks (an H200 or a B200 with BF16 weights).

## 5. Metrics
- FIX: r_fix(P, L) = b_compact / b_stock (cells with b_stock >= 1). The footprint ratio R(L) for the same cell.
- MIX: r_mix(P, W) = mean decoding requests per step, compact / stock. Also reported: the workload footprint ratio
  (stock / compact steady footprint, summed over the backlog), the realized share (r - 1) / (R - 1), preemptions,
  recomputed tokens, and head-of-line stops (steps where a later waiting request would fit but the head does not).
- Longest prompt (reported, not deciding): for each pool, the longest prompt that one request can have (its
  reservation fits the empty pool), stock against compact. This is the context-length side of the capacity.
- Sensitivity (reported, not deciding): `max_num_batched_tokens` 8,192 instead of 2,048 (a larger prefill
  reservation), and the admission rule with the reservation of the first chunk only (`scheduler_reserve_full_isl` off).

## 6. Validity checks (a failed check makes the run INVALID; fix and rerun, there is no cost)
- V1 (tie to E-57): at 6,143 blocks the model gives the four E-57 batches (3 / 4 at 32,768; 2 / 2 at 65,536) and the
  steady footprints 1,349 / 1,008 and 2,373 / 1,690 within 1 block.
- V2 (conservation): held blocks never exceed the pool after preemption; every MIX request finishes; the tokens
  decoded equal the sum of the outputs.
- V3 (determinism): two runs of one MIX cell give the same numbers.
- V4 (limit): with fixed lengths and a pool of 10^6 blocks, r_fix is within 2% of R(L).

## 7. Decision rule (frozen; project heuristics)
- Q_fix = the median of r_fix over the cells with pools 6,000 to 24,000 blocks (about 15 to 60 GiB) and L in
  {16K, 32K, 64K, 128K}. Q_mix = the median of r_mix over M1, M2 and M3 at pools 6,000 to 24,000 blocks (M4 is a
  control and does not decide).
- GO-V2: Q_fix >= 1.25 and Q_mix >= 1.25. The v2 paper can say that the footprint saving becomes about that much more
  concurrent decoding over common pool sizes, with the figure.
- KILL-V2: Q_fix < 1.10 or Q_mix < 1.10. The v2 paper must say that the admission rule loses most of the saving.
- PIVOT-V2: every other case. The v2 paper reports the realized share and names the admission rule as the limit, with
  the sensitivity rows.
- The decision is about the v2 claim only. It is a model of the vLLM scheduler, not a measurement: SCREEN ONLY.

## 8. Files
- `poolsim.py`: the block-pool and scheduler model.
- `sweep.py`: the FIX and MIX grids; writes `results/e63_<date>.json` and prints one decision line.
- `test_poolsim.py`: V1, V2, V3 and V4 as tests, and unit tests of the rules.
- `results/`: outputs.

## 13. Amendment A1 (2026-11-10, after the R-97 pre-run review, before any run of `sweep.py`)
R-97 (reviewer): AMEND_THEN_RUN, 4 blockers, 9 fixes. The admission arithmetic matches vLLM
(reservation, free count, FCFS break, victim `running[-1]`, recompute of prompt plus output). A1 replaces Sections
4 to 7 where they differ:
- B1 (the MIX metric): a stream, not a finite backlog. The waiting queue is refilled to 8 requests from the seeded
  generator, so the backlog never drains. The window is steps 2,000 to 12,000, the same for both layouts. The deciding
  metric is decode tokens per step; a time-weighted rate (0.02 s per step plus 0.5 s per 2,048 prefill tokens, an
  A100-like weight, ASSUMPTION) is reported. `pool_bound_share` = the share of the steps in which the admission loop
  looked at a waiting request and stopped because its reservation did not fit the free blocks. A MIX cell counts for
  Q_mix only if the STOCK share is at least 0.5 (the pool binds); the count of excluded cells is reported. With
  fewer than 5 counted cells, Q_mix is NA and the word is PIVOT-V2.
- B2 (a bug): a preempted request now prefills its original prompt plus its output so far (`base + out`); the old
  code added the output twice after a second preemption. A unit test preempts one request twice.
- B3 (the scope of GO): the rule holds at `max_num_batched_tokens` 2,048 (the vLLM server default on an A100).
  FACT (R-97): the default is 8,192 on H100 or H200 and 16,384 at 160 GB or more. Q_fix and Q_mix at 8,192 are
  reported on the same grids and do not decide. A GO reads: "at 2,048 batched tokens".
- B4 (the rule): per axis, GO at Q >= 1.25, KILL at Q < 1.10, else MID. GO-V2 only if both axes are GO; KILL-V2 only
  if both are KILL; else PIVOT-V2 with the two axis labels. Step-weighted Q_mix decides; if the time-weighted Q_mix
  gives a different word, the word is PIVOT-V2. V6 (the M4 control): INVALID unless 0.97 <= r_mix(M4) <= the M4
  footprint ratio + 0.03 at every decide pool.
- F1: no admission in a step that preempted (vLLM `if not preempted_reqs`).
- F2: FIX output length = max(256, 2 x (b + 1) x ceil(L / budget)) with b the larger closed-form batch; r_fix comes
  from the simulated maximum of simultaneous decoding requests. The closed form is reported, and V5 counts the cells
  where the simulation is more than 1 below it (reported, not a gate: preemption can make the simulation lower).
- F3: V1 adds the E-57 prefill peaks 2,602 / 2,262 / 3,626 / 2,944 within 6 blocks (one request of L - 64 tokens).
  The E-57 "lacks" (151 and 181 blocks) are not checked: the model does not copy the timing of the E-57 probe.
- F4: V2 asserts that the held blocks never exceed the pool in every FIX and MIX run.
- F5: a run that reaches `max_steps` raises an error with the cell; the JSON is written after each part.
- F6: the longest-prompt search is capped at 1,048,576 tokens and not clamped to the model's context limit (stated).
- F7: the MIX decide set is the 5 pools 6,727 to 19,027 blocks (the grid point 24,676 is just above 24,000), so
  Q_mix is a median of at most 15 values.
- F8: the sensitivity grids (budget 8,192; first-chunk reservation) report Q_fix, Q_mix and the word with the same
  decide sets and statistics.
- F9: robustness. The word is recomputed with sync scheduling (sliding cap 193 instead of 321) and with a watermark of
  1% of the pool; if either changes the word, the word is PIVOT-V2.
- Optional items adopted: H-MIX is now numeric (the median realized share of the counted cells is at least 0.5).
  Not adopted: V3 across processes (string seeds of `random.Random` are deterministic), a dynamic tie to the E-57
  traces (no per-step traces were saved for every row).
- Tests: 11 pass (A1 adds the peaks, the double preemption and the stream determinism).

## 14. Amendment A2 (2026-11-10, after run attempt 1 stopped)
- Run attempt 1 stopped with an error in the robustness grid: a FIX cell (pool 2,000 blocks, stock) reached
  `max_steps`. Cause (FACT, a debug script): with sync scheduling (sliding cap 193) the model still counted two
  chunks in flight, so a lone request outgrew its own reservation, preempted itself and was admitted again, forever.
  Fix: the chunks in flight follow the scheduling mode (2 async, 1 sync). The default (async) path does not change,
  so the deciding grids do not change. The partial JSON of attempt 1 was not read; the whole sweep is run again.
- The same loop could happen in the first-chunk-reservation grid. Guard: a FIX cell has b = 0 for a layout if one
  request with its full prompt and output does not fit the empty pool; FIX runs stop at 500,000 steps with an error.
- A test checks that a lone request under sync scheduling is not preempted. 12 tests pass.

## 15. Amendment A3 (2026-11-10, after run attempt 2 stopped)
- Run attempt 2 stopped in the robustness MIX grid with a division by zero: the stock layout decoded nothing in one
  watermark cell. Cause (INTERPRETATION from the code): the stream's skip rule ignored the watermark, so a request
  whose reservation fit the pool but not the pool minus the watermark sat at the head of the queue forever.
- Fix: the skip rule uses the same budget, sliding cap and watermark as the run. So under robustness and sensitivity
  the skipped set can differ from the main grid (reported per cell). The default path does not change. A run with no
  stock decoding now stops with an error naming the cell. A test covers it. 13 tests pass. The partial JSON of
  attempt 2 was not read; the whole sweep is run again.

## 16. Result (attempt 3, 2026-11-10, 54 s on the Mac at nice 10)
Frozen word: **PIVOT-V2** (step and time labels differ: GO-V2 on the step axis, KILL on the time axis).
`results/e63_2026-11-10.json`. All checks pass (V1 batches, footprints and peaks; V3; V4; V6 the M4 control); V5 0.

| Statistic | Value | Label |
|---|---|---|
| Q_fix (44 cells: pools 15 to 60 GiB, 16K to 128K) | 1.369 | GO |
| Q_mix, step-weighted (M1 to M3, 15 of 15 cells pool-bound) | 1.432 | GO |
| Q_mix_time (time weights 0.02 s per step, 0.5 s per 2,048 prefill tokens) | 1.053 | KILL |
| Median realized share (H-MIX) | 1.17 | passes (at least 0.5) |
| Robustness: sync scheduling (cap 193) / 1% watermark | Q_fix 1.333 and 1.360; Q_mix 1.418 and 1.424 | GO-V2 both |
| Sensitivity: 8,192 batched tokens | Q_fix 1.333; Q_mix 1.396; Q_mix_time 1.12 | GO-V2 (steps) |
| Sensitivity: first-chunk reservation | 0 pool-bound cells | NA (says nothing) |

- FIX by length (median over pools of 15 to 60 GiB; footprint ratio in brackets): 8K 1.167 (1.171), 16K 1.300 (1.255),
  32K 1.364 (1.338), 64K 1.500 (1.403), 128K 1.500 (1.447). No preemptions in any FIX cell.
- Longest single prompt: 1.5 times longer with the compact cache (the global block ratio 48 / 32) for pools of 4,000
  to 19,027 blocks; the rows at 24,675 and 32,000 blocks hit the search cap and must not be quoted.
- INTERPRETATION (confirmed by R-98): in a saturated stream with long prompts and short outputs, completions per second
  equal the prefill rate divided by the mean prompt, so decode tokens per second are set by prefill, not by
  concurrency. The step metric is a batch size, not a rate.

## 17. Post-run review R-98 (reviewer, 2026-11-10): CONFIRM_WITH_CORRECTIONS
- FACT: its own script reproduces Q_fix, Q_mix, Q_mix_time and the word; A2 and A3 leave the deciding path unchanged
  (the A2 guard zeroed 0 main-grid cells; the A3 skip rule equals the old rule at watermark 0).
- Correction 1: H-FIX failed. It predicted 1.25 to 1.30, below the footprint ratio; Q_fix (1.369) is about equal to
  the footprint median of the same cells (1.371), and 35 of 44 cells are above their own footprint ratio, because of
  integer steps when b is 1 to 3 (10 of 44 cells have b_stock of 2 or less). Cells with b_stock of 8 or more give 1.306
  (their footprint median 1.255). Claim "about 1.3 to 1.4". (R-98 wrote "above the 1.371 median"; 1.369 is not above
  1.371, so the author restated the point.)
- Correction 2: the time label is fragile in value but not in word. Other weights give Q_mix_time 1.05 to 1.18 (0.05 s
  per step: 1.10; 0.05 s and 1.0 s per 2,048: 1.06; 0.05 s and 0.25 s: 1.18; 0.1 s and 0.5 s: 1.18), never GO. The
  weights are an ASSUMPTION; a GPU measurement could change them.
- Correction 3: do not quote the first-chunk sensitivity (its gate counts admission stops, but there the pool binds by
  preemption), M3 at 4,000 blocks (realized share 3.5, a small denominator; outside the decide set), or the capped
  longest-prompt rows.
- Claims allowed for v2: in a CPU model of the vLLM admission rule at 2,048 batched tokens, over pools of 15 to 60 GiB
  and long mixed workloads, the compact cache raises the mean number of decoding requests per step by about 1.3 to 1.4
  times (footprint ratio 1.31 to 1.39), so the admission rule does not lose the saving; and one request can be 1.5
  times longer at the same pool. Not allowed: any throughput (tokens per second) gain, or the time label as a
  measurement.
- Process note: R-98 previewed a decode-heavy variant in its scratch scripts (time ratios 1.03 to 1.18). A later
  decode-heavy test (E-64) is therefore not blind; its rule must be stated with that preview disclosed.
