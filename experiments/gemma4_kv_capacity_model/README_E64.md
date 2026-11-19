# E-64: does the compact K=V capacity gain become decode throughput? (CPU model, I-182 v2)

Status: RUN 2026-11-10, SCREEN ONLY. No workload GO; prompt median 1K KILL on both nodes; all others MID (Sections 9
and 10). Predeclared 2026-11-10 (Sections 0 to 8, before any run of `e64.py`). Free: Mac CPU at low priority, no GPU, no
spend. Same model as E-63 (`poolsim.py`) plus a decode-only node mode and a byte-based step-time model.
Labels: FACT, INTERPRETATION, HYPOTHESIS. Thresholds are project heuristics.

## 0. Not blind (disclosure)
R-98 (the E-63 post-run reviewer) previewed a decode-heavy unified-node variant in its scratch scripts with the E-63
time weights (constant 0.02 s per step): time ratios 1.06 and 1.03 (prompt median 1K), 1.18 and 1.13 (8K), 1.09 and
1.03 (32K), at pools of 6,727 and 11,314 blocks. The author read these numbers before writing this rule. The
disaggregated decode-node scenario (D) and the byte-based time model were not previewed.

## 1. Question
E-63: the compact cache gives about 1.3 to 1.4 times more requests decoding at once, but in a prefill-bound stream
decode tokens per second rose only about 1.05. Where does the capacity gain become throughput?
- U: a unified node (prefill and decode on one GPU) with decode-heavy traffic (long outputs).
- D: a disaggregated decode node (prefill on another node; the decode node receives the prompt KV and only decodes).

## 2. Rubber duck
- In a closed, saturated stream, decode tokens per second = completions per second x mean output. On a unified node
  the completions are limited by prefill time when prompts are long; on a decode node there is no prefill, so the
  limit is the decode step time and the batch.
- A decode step reads the weights once (a fixed cost) plus the KV of every decoding request. A larger batch spreads
  the fixed cost over more tokens, so more concurrency gives more tokens per second until KV reads dominate.
- The compact cache reads 37.5% fewer global KV bytes, but E-59 measured almost no step gain at batch 3 and 32K
  (50.52 against 50.73 ms). So the time model has a conservative arm where the compact KV read costs the same as stock.
- What can make the model wrong: the decode-node admission (vLLM P/D connectors allocate the prompt blocks when the
  KV arrives; this model reserves the steady footprint of the prompt at admission), KV transfer time (ignored), and
  the step-time weights (assumed, then checked against E-59 at one point).

## 3. Hypotheses
- H-U (HYPOTHESIS; informed by the R-98 preview): on a unified node the time ratio stays below 1.25.
- H-D (HYPOTHESIS): on a decode node the time ratio is at least 1.15 for long prompts (the batch is KV-bound and the
  fixed weight cost is shared by more requests).

## 4. Setup (frozen)
- Engine model: `poolsim.PoolSim` with the E-63 rules (async scheduling, budget 2,048, full-ISL admission, watermark
  0, max_num_seqs 256). New mode `decode_only=True` for D:
  - The reservation of a request is its steady footprint (`steady(prompt)`): its global blocks plus 65 sliding blocks
    per group. No prefill reservation.
  - An admitted request holds its prompt KV at once (c = prompt) and decodes from the next step; it uses no token
    budget. A preempted request goes back to the head of the waiting queue and is admitted again with prompt + output
    so far (the KV is sent again; transfer time is ignored).
- Workloads (seeded generator, refilled stream as in E-63 A1; prompts clipped to [256, 131,072], outputs to [16,
  16,384]; lognormal, sigma 0.6 for outputs):
  - D1: prompt median 1K (sigma 0.8), output median 2K.
  - D2: prompt median 8K (sigma 0.8), output median 2K.
  - D3: prompt median 32K (sigma 0.6), output median 2K.
  - D4 (reasoning): prompt median 4K (sigma 0.8), output median 8K.
  - For D only, also the E-63 prompt-heavy workloads M1 (prompt 16K, output 300) and M2 (prompt 48K, output 500).
- Scenarios: U runs D1 to D4 on the unified node. D runs D1 to D4, M1 and M2 on the decode node.
- Pools: the E-63 decide set, 6,727 / 8,724 / 11,314 / 14,672 / 19,027 blocks (about 16 to 46 GiB).
- Window: steps 8,000 to 40,000 (outputs are long, so the warm-up is longer than in E-63), the same for both layouts.
- Step-time model (ASSUMPTION, A100-like), computed from the step trace:
  - decode part = 0.040 s (weights and fixed cost) + KV bytes read / 1.5e12 B/s, if any request decodes;
  - KV bytes read per decoding request with context c: global 10 x 4 x 1,024 x 2 B x c (stock) or 10 x 4 x 640 x 2 B
    x c (compact), plus sliding 50 x 16 x 512 x 2 B x min(c, 1,024);
  - prefill part = 0.5 s per 2,048 prefill tokens;
  - weight set W-A as above; W-B (pessimistic for compact): the compact global read costs the same bytes as stock,
    and prefill is 1.0 s per 2,048 tokens.
- Metric: r_time = decode tokens per second, compact / stock, per cell and weight set. Reported: r_step (decode tokens
  per step), the pool-bound share, preemptions, the mean batch.

## 5. Validity checks (a failed check makes the run INVALID; fix and rerun, no cost)
- V1: the E-63 tests still pass (E-57 batches, footprints, peaks; the default path is unchanged).
- V2: conservation in every run (held blocks never exceed the pool; no stock run with zero decoding).
- V7 (time model calibration): the W-A step time of 3 stock requests at 32,768 tokens is within 30% of the E-59
  measurement (50.73 ms). Reported: the compact step at the same point against 50.52 ms.
- V8 (decode-only rule): for equal requests on the decode node, the simulated batch equals floor(pool / steady(p))
  (or that value minus 1 when decode growth forces a preemption), checked at 3 points.

## 6. Decision rule (frozen; heuristics)
- For each scenario S in {U, D} and weight set W in {W-A, W-B}: Q(S, W) = the median r_time over the cells of S with a
  stock pool-bound share of at least 0.5 (at least 5 cells, else NA).
- Scenario label: GO if Q >= 1.25 under both W-A and W-B; KILL if Q < 1.10 under both; else MID (NA if a Q is NA).
- The v2 paper may claim a throughput gain only for a scenario labelled GO, and only in the words "in a model with
  assumed step times". MID: report the range. KILL: say that the capacity gain does not become throughput there.
- One line: `E-64: U <label> (Q_A, Q_B) | D <label> (Q_A, Q_B) | checks`.

## 8. Amendment A1 (2026-11-10, after the R-99 pre-run review, before any run of `e64.py`)
R-99 (reviewer): AMEND_THEN_RUN. FACT (vLLM 2026-09-16 checkout): the decode-node admission (the steady footprint, no
token budget with an async KV load) matches `allocate_slots`. A1 replaces Sections 2, 4, 5 and 6 where they differ:
- B1 (preemption on a decode node): a preempted request prefills again LOCALLY (prompt plus output so far) through the
  token budget and the full-ISL reservation (FACT: the NIXL connector clears `do_remote_prefill` after the first
  allocation). Section 4 "the KV is sent again" is withdrawn. The prefill time share of each D cell is reported, and a
  cell is flagged when it is above 15%. The skip rule uses the full-ISL reservation of prompt plus output, so a
  re-prefill always fits.
- B2 (labels): the label is per workload, not pooled. For each workload and seed, Q(W) = the median r over its
  counted pools (at least 3 of 5). Thresholds as Section 6. The pooled medians are reported only. H-D maps to the D3
  and M2 labels on the decode node.
- B3 (noise): 5 seeds per workload, warm-up 40,000 steps, horizon 200,000 steps. A workload label holds only if all 5
  seed labels agree; otherwise it is MID. The median and the range over the seeds are reported. Run time about 35 to
  45 minutes on one core at nice 10.
- F4: V7 has little power (the weights alone pass it); the model's compact step is 4.5% faster at batch 3 and 32K,
  against 0.4% measured, so W-B (no compact read gain, plus 0.4 ms of compact hook cost per decode step, prefill 1.0 s
  per 2,048) is the arm consistent with E-59. Reported, not deciding: both arms at 0.75e12 and 3e12 B/s.
- F5: a cell counts only if the stock pool-bound share is at least 0.5 AND the share of steps at max_num_seqs (256)
  is at most 0.1.
- F6: V8b compares the sliding blocks of `steady()` with the vLLM count cdiv(c,16) - (c-1,023)//16 + 1 at 4 contexts;
  they must agree within 1 block per group (`steady()` is the E-57-validated rule; it can be 1 block low, about 0.4%).
- F7 (scope): D is a saturated decode node with an infinite backlog, the prefill node assumed fast enough, the hybrid
  KV manager on (a connector with HMA support, such as NIXL), and KV transfer time ignored. No end-to-end throughput
  claim follows. The pool sizes need not fit an A100 with 62 GB of BF16 weights: the result depends on the KV pool and
  the step-time weights, not on the GPU.
- F8 (known biases): mixed steps count the fixed 0.040 s on top of the prefill time (about 8% too much); prefill
  attention FLOPs are not charged (favors compact in U); transfer time is ignored (understates compact on D).
- Tests: 18 pass (A1 adds the local re-prefill on a decode node).

## 9. Result (2026-11-10, one run, 3,704 s on one core at nice 10)
FACT (`results/e64_2026-11-10.json`): all checks pass (V7 47.05 ms stock and 45.03 ms compact against 50.73 and 50.52
measured; V8; V8b diffs 0, 1, 0, 1). Every cell counts (pool-bound share 0.99 to 1.00, no step at max_num_seqs).
Frozen line: `E-64: U-D1 KILL | U-D2 MID | U-D3 MID | U-D4 MID | D-D1 KILL | D-D2 MID | D-D3 MID | D-D4 MID | D-M1 MID
| D-M2 MID`. No workload is GO.

| Workload | Label | r_step | Q_A (W-A) | Q_B (W-B) | Seed labels | Re-prefill flags |
|---|---|---|---|---|---|---|
| U-D1 (1K, 2K) | KILL | 1.085 | 1.085 [1.083, 1.086] | 1.047 [1.041, 1.048] | 5 KILL | n/a |
| U-D2 (8K, 2K) | MID | 1.230 | 1.181 [1.165, 1.190] | 1.085 [1.069, 1.100] | 5 MID | n/a |
| U-D3 (32K, 2K) | MID | 1.423 | 1.274 [1.260, 1.305] | 1.133 [1.121, 1.167] | 5 MID | n/a |
| U-D4 (4K, 8K) | MID | 1.203 | 1.206 [1.198, 1.213] | 1.109 [1.104, 1.119] | 5 MID | n/a |
| D-D1 (1K, 2K) | KILL | 1.086 | 1.092 [1.088, 1.093] | 1.052 [1.046, 1.054] | 5 KILL | 0 of 25 |
| D-D2 (8K, 2K) | MID | 1.231 | 1.198 [1.181, 1.201] | 1.095 [1.079, 1.104] | 5 MID | 10 of 25 |
| D-D3 (32K, 2K) | MID | 1.390 | 1.299 [1.280, 1.384] | 1.127 [1.085, 1.215] | 5 MID | 10 of 25 |
| D-D4 (4K, 8K) | MID | 1.215 | 1.219 [1.209, 1.228] | 1.121 [1.111, 1.136] | 5 MID | 0 of 25 |
| D-M1 (16K, 300) | MID | 1.345 | 1.300 [1.246, 1.314] | 1.156 [1.080, 1.167] | 5 MID | 11 of 25 |
| D-M2 (48K, 500) | MID | 1.452 | 1.390 [1.297, 1.395] | 1.168 [1.099, 1.258] | 4 MID, 1 GO | 8 of 25 |

The values are the median over the 5 seeds, with the seed range in brackets. Pooled (reported only): U 1.197 (W-A)
and 1.096 (W-B); D 1.230 and 1.111. Bandwidth arms (0.75e12 and 3e12 B/s) move each workload median by 0.045 or less in both
weight sets, and they change no label.

FACT (mechanism, from the cells):
- Short prompts (D1): the KV is a small part of the footprint (the sliding window dominates), so r_step is only 1.09
  on both nodes. The capacity gain is small, and so is the throughput gain.
- Long prompts: r_step is 1.39 to 1.45 (D3, M1, M2), the E-63 capacity gain. W-A turns most of it into time (1.27 to
  1.39). W-B, with no compact read gain, keeps only 1.13 to 1.17.
- Unified node, long prompts: the stream is still partly prefill-bound (W-A prefill time share 0.17 to 0.37 stock
  and 0.24 to 0.45 compact in U-D3; R-100 estimates about 0.3 to 0.6 under W-B). So U is "long outputs", not
  "decode-heavy". The gap between r_step (1.42) and Q_A (1.27) is this prefill share.
- Decode node: admission at the steady footprint leaves no headroom, so decode growth forces preemptions. Compact
  admits more requests and preempts more often (D3 at 19,027 blocks: 225 stock against 439 compact). Each preempted
  request prefills again locally (A1 B1), so the compact prefill time share rises to 0.10 to 0.22 and r_time falls
  below r_step. Single cells can fall below 1.0 under W-B (D3 at 19,027 blocks, seeds 0 and 2: 0.939 and 0.952).
  The M2 dip at 11,314 blocks is the same seed noise (r_B per seed 1.220, 1.132, 0.989, 1.038, 1.009).
- The unified node has almost no preemptions for D2 and D3: full-ISL admission leaves slack.

## 10. Post-run review R-100 (2026-11-10, reviewer)
CONFIRM_WITH_CORRECTIONS. FACT: its own script recomputes all 10 labels from the 250 cells; all agree. All 250 cells
pass the filter, so F5 and the 3-of-5 rule did not bite; `skipped` is 0 in every cell. No code deviation from A1.
- The doubled compact preemptions on D are a model mechanism, not a bug (zero headroom at steady admission; compact
  packs more requests; the re-prefill needs the larger full-ISL reservation). R-100 did not verify the thrash share by
  simulation. Real decode nodes usually cap concurrency, so a saturated D with an infinite backlog is a stress case.
- Correction: call U "long outputs", not "decode-heavy" (Section 9).
- Hypotheses, both arms side by side (W-B is the arm consistent with E-59): H-U holds under W-B for all workloads and
  under W-A except D3 (1.274, all seeds 1.26 to 1.31). H-D holds under W-A (D3 1.30, M2 1.39); under W-B it fails for
  D3 (1.127, 3 of 5 seeds below 1.15) and is borderline for M2 (1.168, range 1.099 to 1.258).
- Allowed v2 claims: no throughput claim (no GO; the one GO seed of D-M2 is excluded by the 5-seed rule). KILL (prompt
  median 1K, both nodes): the capacity gain does not become throughput there (Q 1.04 to 1.09). MID: report the ranges,
  for example D-D3 1.28 to 1.38 (W-A) and 1.09 to 1.22 (W-B), U-D3 1.26 to 1.31 and 1.12 to 1.17. Always say "in a
  model with assumed step times", with the F7 scope and the F8 biases.

Hypotheses: H-U (time ratio below 1.25 on a unified node): holds under W-B for all workloads; fails under W-A for D3
(1.274). H-D (at least 1.15 for D3 and M2 on the decode node): holds under W-A (1.299, 1.390); under W-B it fails for
D3 (1.127) and holds for M2 (1.168).

## 7. Files
- `poolsim.py` (the decode-only mode and the step trace; the E-63 default path unchanged), `e64.py` (the grids, the
  time model, the checks, the rule), `test_poolsim.py` (E-64 tests added), `results/e64_<date>.json`.
