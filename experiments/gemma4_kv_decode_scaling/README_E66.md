# E-66: does the prefix cache remove the recompute cost of preempted requests? (CPU model, I-182 v2)

Status: CLOSED 2026-11-10: CLAIM-STANDS, confirmed by R-104 (Sections 10, 11). Predeclared before any run of `e66.py`.
Free: Mac CPU at low priority, no GPU, no spend.
Idea: I-182 (v2). Origin: R-102 (E-65). We, 2026-11-10: "sure, go ahead!".
Labels: FACT, INTERPRETATION, HYPOTHESIS. Thresholds are project heuristics.

## 1. Question
E-65 found no workload at a 1.25x decode-throughput gain, but its model recomputes every preempted request in full.
vLLM enables prefix caching by default. With the vLLM eviction and hit rules, how many re-admitted requests get a
cache hit, and does any workload reach GO? "No": no workload is GO in the engine arm at read ratio 0.625.

## 2. Rubber duck
FACT (vLLM main, 2026-11-10 clone; the pinned commit is to be checked by the review):
- `enable_prefix_caching` defaults to True (`config/cache.py`).
- A preempted request frees its blocks; `free()` runs per group, each in reverse order, so the newest blocks are
  reused first (`single_type_kv_cache_manager.py`, `block_pool.free_blocks`). Blocks with no hash go to the front of
  the free queue, hashed blocks to the back. `get_new_blocks` takes from the front and evicts the hash.
- A hit needs the global prefix, and in each sliding group `cdiv(1023, 16)` = 64 contiguous cached blocks that end at
  the hit point (`SlidingWindowManager.find_longest_cache_hit`; the hybrid coordinator reconciles the groups).
- A running request holds only about 65 sliding blocks per group. Older sliding blocks left the window earlier, went
  into the queue, and were reused first.

INTERPRETATION: a preemption happens because a running request needs a block and none is free. That request then takes
blocks from the victim. After the few unhashed blocks, it reuses hashed newest blocks. Losing one global tail block
moves the hit point below the sliding window; losing one sliding tail block breaks the 64-block run. So one reused
hashed block removes the hit. A hit needs re-admission before that, and re-admission needs free blocks from a finished
request. So hits should be rare under memory pressure.

What could make this wrong: the older sliding blocks may survive when the pool is not full; the pinned commit may
differ; on a decode node, blocks loaded by a connector may be cached differently.

## 3. Hypotheses
- H1 (primary): the engine arm has a hit rate below 10% of re-admissions, and no workload is GO at ratio 0.625.
- H0: the hit rate is high, and at least one decode-node workload is GO.
- Report: the `perfect` arm is an upper bound. R-102 found GO with zero prefill cost on three decode-node workloads, so
  `perfect` may show GO. That does not change the rule.

## 4. Setup (frozen)
- Engine model: `poolsim.PoolSim(prefix_cache=...)`, the E-64 rules otherwise. The default `none` is unchanged (V1).
- Free-queue model: an exact position count. `popped` is the queue head. Each step adds the gross new blocks of every
  scheduled request (global blocks plus 5 sliding groups). A preemption records the victim's hashed range at the queue
  tail and moves 6 unhashed blocks (1 + 5 groups, an optimistic count) to the front. The blocks that forced the
  preemption are taken after it. A hit removes the surviving range from the queue.
- Modes, applied only when a preempted request is admitted again:
  - `engine` (decides): hit = the aligned full prefix if no hashed block was reused, else 0. An upper bound on vLLM: it
    does not check the window alignment at the hit point.
  - `global` (report only): hit = the surviving global prefix; sliding blocks are assumed to be reused first. Not
    a real engine path.
  - `perfect` (report only): the aligned full prefix always.
  - With a hit, the admission needs the global blocks of p plus 5 x the sliding blocks of the window and the remaining
    tokens (fewer than the full-ISL reservation), and only p - hit tokens prefill.
- Workloads, pools, seeds, horizon, warm-up, filter: the E-64 grid. Time models: both E-65 bracket tables (read ratio
  1.0 and 0.625), `prefill_fit.json`, the E-65 gate arms.
- Prompts are unique, so a first admission never hits. Shared prefixes in real traffic are out of scope.

## 5. Decision rule (frozen; heuristics)
- Per mode, ratio and workload: the E-65 A1 label (GO only if M, M_pf_equal and M_gemm are all at least 1.25; KILL only
  if all are below 1.10; else MID; all 5 seeds must agree, else MID).
- Word, from the `engine` mode at ratio 0.625: CLAIM-STANDS if no workload is GO; CLAIM-CHANGES if any workload is GO.
- CLAIM-STANDS: the E-65 wording stays, and "the model has no prefix cache" is replaced by the E-66 result.
- CLAIM-CHANGES: the E-65 wording is withdrawn for those workloads.
- Reported: the hit rate per workload and layout, the hit tokens against the recomputed tokens, the `global` and
  `perfect` labels, the `engine` labels at ratio 1.0.

## 6. Validity checks (a failed check makes the run INVALID; fix and rerun, no cost)
- V1: mode `none` reproduces a stored E-65 bracket cell (simulation exact; time ratios within 1e-4).
- V2: conservation in every mode, unified and decode node; every request finishes.
- V3: the recorded victim positions never pass the queue tail.
- V4: the hit rules on set cases; `perfect` hits every re-admission.
- V5: `engine` gives no hit after a reused hashed block.

## 7. Budget
Mac CPU, one core, nice 10: three modes, about 40 minutes each. No spend.

## 8. Threats to validity
1. A model of the queue, not the engine. The gross block count is approximate during prefill chunks.
2. Read on vLLM main; the pinned commit may differ (the review checks it).
3. The `engine` arm ignores window alignment, so it can only overstate hits.
4. A hit in the model costs no extra time for touching blocks.
5. The E-64 and E-65 limits stay.

## 9. Files
`../gemma4_kv_capacity_model/poolsim.py` (`prefix_cache`), `e66.py`, `test_e66.py`, `results/e66/`.

## Amendments

### A1 (2026-11-10, after R-103, before any run of `e66.py`, and before the output of `refsim.py` was read)
R-103 (reviewer; it read the pinned commit 3966924d566340e971a6da313e790d05492aa505): AMEND_THEN_RUN. FACT: the Section 2
rule holds for decode victims at the pin. Its instrumented cells show long waits (D-M2 compact: median 152 steps, the
median victim fully reused) and 18 hits in 1,056 re-admissions. Changes (they replace Sections 3 to 8 where they differ):
- B1: V3, V4 and V5 test the rule against itself. New check V6: `refsim.py`, a block-level reference (real block ids,
  one FIFO queue, per-group hashes, a copy of the vLLM hit search), replays 6 cells for 60,000 steps with both group
  free orders. Per re-admission it compares the reference hit with the model hit. V6 passes if (a) the cases "the
  reference hits, the model does not" are at most 5% of re-admissions, and (b) the free counts of the two differ by at
  most 16 blocks. V3 also checks that victim ranges lie inside the queue and do not overlap.
- B2: `engine` is an upper bound on the alignment for decode victims only (a hit needs a multiple of lcm(16, global
  block), so even with no reuse a hit is possible in only 47% of states for stock and 31% for compact). A victim
  preempted during prefill holds more sliding blocks and can keep a partial hit, which the model misses. V6 measures it.
- F6, the rule: the word is CLAIM-STANDS only if (a) no workload is GO in `engine` at ratio 0.625, (b) V6 passes, and (c)
  the `engine` hit rate over all cells is at most 10% of re-admissions. If (a) fails: CLAIM-CHANGES. If (a) holds and
  (b) or (c) fails: CLAIM-REVIEW. Reported: the hit-token share, and Q_M of `engine` minus Q_M of `none` per workload.
  R-103 notes that with few hits this rule has little power by design; the evidence is the hit rate and the reference.
- F6, the arms: `global` ignores the recompute of the sliding window; vLLM cannot use a global-only hit. Report only.
- F5, limits: on a decode node with async scheduling, vLLM defers block frees and skips a victim with a step in flight
  (`_request_blocks_can_be_freed`). The model has neither, in any mode. The pinned sliding cap has no "+1"; the model
  has it in every mode. Finished requests also prepend unhashed blocks; the model ignores this (conservative, at most
  6 blocks per finisher).
- Not changed: the question, the modes, the grid, the time models, the gate, the 5-seed rule.

### A2 (2026-11-10, after `refsim.py` attempt 1, before any run of `e66.py`)
Attempt 1 of V6: 1,386 re-admissions over 12 replays: both zero 1,350, both hit 14, model only 0, reference only 22
(1.6%, part (a) passes). Part (b) failed: the free counts differed by up to 30 blocks (limit 16), and the reference
ran short 3,450 times. Cause (a bug in the reference): its filler blocks were released only at the end of a step, so
the growth inside a step ran out of blocks and created virtual blocks that never left the queue. Fix: a growing
request converts its own filler blocks first. The thresholds of V6 are unchanged.

Attempt 2 of V6 (the final `refsim.py`, before any run of `e66.py`; recorded here after the run): PASS. 1,386
re-admissions: both zero 1,352, both hit 14, model only 0, reference only 20 (1.44%, limit 5%). Model hit tokens 13,280,
reference hit tokens 22,816. Prefill victims 0. Short 20. Largest free-count difference 5 blocks (limit 16).
Disclosure (R-104): A2 is a fix made after part (b) failed. With attempt 1 alone, the rule gives CLAIM-REVIEW.

## 10. Result (2026-11-10)
The run: `run_e66.sh`, three modes in sequence, 250 cells each, about 53 to 55 minutes each. Files: `results/e66/`.
`results/e66/manifest.json` holds the hashes and modification times; it was written after the run (R-104 finding 1).

Hit rate, `engine`, all cells: 1,239 hits in 165,263 re-admissions (0.75%). Hit tokens are 0.38% of the discarded tokens
(5,735,504 of 1,505,347,190; `recomputed_tokens` counts the context discarded at each preemption, before any hit).
For comparison: `global` 27.1% and 26.9% of tokens; `perfect` 100% and 99.3% of tokens.
Per workload, the rate is 0% to 1.5%, except U-D2 and U-D3, where 1 or 2 re-admissions exist in all and all hit.

Q_M, 5-seed median, read ratio 0.625 (label, then the change against `none`, the E-65 bracket):

| Workload | `none` | `engine` | `global` (report only) | `perfect` (report only) |
|---|---|---|---|---|
| U-D1 | KILL 1.048 | KILL 1.048 (+0.000) | KILL 1.048 | KILL 1.047 |
| U-D2 | MID 1.097 | MID 1.097 (+0.000) | MID 1.097 | MID 1.097 |
| U-D3 | MID 1.102 | MID 1.102 (+0.000) | MID 1.102 | MID 1.102 |
| U-D4 | MID 1.137 | MID 1.137 (+0.000) | MID 1.137 | MID 1.134 |
| D-D1 | KILL 1.051 | KILL 1.052 (+0.000) | KILL 1.051 | KILL 1.052 |
| D-D2 | MID 1.109 | MID 1.107 (-0.001) | MID 1.134 | MID 1.167 |
| D-D3 | MID 1.117 | MID 1.117 (-0.000) | MID 1.140 | GO 1.349 |
| D-D4 | MID 1.147 | MID 1.147 (+0.000) | MID 1.143 | MID 1.150 |
| D-M1 | MID 1.178 | MID 1.174 (-0.004) | MID 1.261 | GO 1.275 |
| D-M2 | MID 1.203 | MID 1.203 (+0.000) | GO 1.354 | GO 1.398 |

`engine` at read ratio 1.0: KILL on U-D1, U-D2, U-D4, D-D1, D-D2; MID on U-D3, D-D3, D-D4, D-M1, D-M2; best median
1.102 (D-M2). No GO.

Rule (A1, F6): (a) no workload is GO in `engine` at ratio 0.625: holds. (b) V6 passes (A2). (c) the hit rate is 0.75%,
at most 10%: holds. Word: **CLAIM-STANDS**. H1 is supported.

INTERPRETATION: the GPU-memory prefix cache almost never saves a preempted request in this model. The request that
forces a preemption takes the blocks of the victim first, and one reused hashed block removes the hit. The gain that
`perfect` shows is real in the model, but no default path in vLLM recovers it from GPU memory.

## 11. Post-run review R-104 (2026-11-10, reviewer)
ACCEPT_WITH_CHANGES. No blocking finding. Word: CLAIM-STANDS. Scripts: a temporary folder `r104/`.
1. FACT: all 60 labels and all 60 Q_M medians, recomputed from the cells, match the stored values.
2. FACT: the D-M2 seeds in `engine` at ratio 0.625 have the labels MID, GO, MID, KILL, MID. The 5-seed rule makes D-M2 MID.
3. Negative changes against `none` above 0.002 (D-M1, and D-D2 at ratio 1.0) are trajectory noise. FACT: 115 of 250
   cells have zero hits and are identical to `none`. In cells with 1 to 8 hits, r_M moves by up to 0.19 in both
   directions, because a hit lowers the admission need and moves later preemptions. Do not read these as an effect.
4. FACT: no over-count of `popped` and no early eviction mark for decode victims were found. A reviewer replay of one
   cell reproduced `refsim.json` exactly. Code modification times come before the results.
5. INTERPRETATION: the model misses partial hits of short-context victims, mainly D1 (the reference finds about 2.4 to
   3 times the hits). The `engine` rate could reach about 2%, or about 5% in an extreme case. Condition (c) does not
   change, and D1 stays KILL even in `perfect`.
6. Limits: V6 covers 6 cells, seed 0, 30% of the horizon, 0.84% of the re-admissions. No prefill victim occurred, so
   that path is unmeasured. `perfect` has zero load cost and says nothing about a CPU offload tier or a remote
   connector. The F5 limits, the unique prompts and the wide D-M2 seed range stay.
7. Open point, checked after the review. FACT: at the pinned commit, a comment in `vllm/v1/core/sched/output.py` marks
   the bounded-replay path "DeepSeek-V4.1 only", and the path needs a KV spec with `prefix_replay_tokens > 0`. On the
   main-branch clone, only `models/deepseek_v41` sets this field. The model files of the pinned commit were not read.

Allowed wording (R-104): "In a CPU model of the vLLM 3966924d5 scheduler with prefix caching, the Gemma 4 31B compact
K=V cache did not reach a 1.25x decode-throughput gain in any of 10 workloads at read ratio 0.625 (5-seed median and
agreement). In the model, 0.75% of re-admitted preempted requests hit the cache (0.38% of discarded tokens). A
block-level reference gave a hit rate of about 2.5% on 6 cells. With zero-cost recovery, three decode-node workloads
exceed 1.25x. This is a model result, not a GPU measurement."
Do not say: "prefix cache has no effect" or "cannot help"; any claim about offload, connectors or tiers that keep the
KV; "measured" or "validated against vLLM"; "never" or "every seed"; any claim for shared prefixes; that the change
against `none` is an effect; "weak point closed" without "GPU-memory prefix cache, in the model".
