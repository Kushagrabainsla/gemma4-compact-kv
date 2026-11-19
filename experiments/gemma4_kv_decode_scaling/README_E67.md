# E-67: does a CPU offload tier turn the compact capacity into decode throughput? (CPU model, I-182 v2)

Status: CLOSED 2026-11-11, word OFFLOAD-CONFIG-GO (Section 10), R-106 ACCEPT_WITH_CHANGES (Section 11). Predeclared
2026-11-10, before any run of `e67.py`. Free: Mac CPU at low priority, no GPU, no spend.
Idea: I-182 (v2). Origin: E-66, R-104. We, 2026-11-10: "Yes, model it".
Labels: FACT, INTERPRETATION, HYPOTHESIS. Thresholds are project heuristics.

## 1. Question
E-66 found that zero-cost recovery of preempted requests gives GO on three decode-node workloads, and that the
GPU-memory prefix cache gives almost no recovery. vLLM also has a CPU offload tier. With its real store, lookup and
eviction rules, and a load cost, does any decode-node workload reach GO? "No": no workload is GO in either config.

## 2. Rubber duck
FACT (vLLM pinned commit 3966924d5, `offloading/scheduler.py`, `kv_offload/base.py`, `policies/lru.py`):
- `offload_prompt_only` defaults to True: only prompt blocks are stored. Generated tokens are recomputed after a hit.
- A hit needs the global prefix and, per sliding group, a run of `cdiv(1024, 16)` = 64 stored chunks ending at the hit
  point, which is a multiple of the global chunk (32 stock, 48 compact).
- For a block size of 16 and an alignment of 32 or 48, `reachable_block_mask` returns dense, so every live sliding block
  is stored. During a local prefill, all sliding blocks are stored (about 900 KB per token). A request whose KV arrived
  from a prefill node holds only its last window.
- The LRU rank of a request is set at its first store and renewed when it finishes. Inside one request, tails go first.
- Loads are asynchronous by default; a request waits while its blocks load.

INTERPRETATION: on a decode node with the default config, the stored window ends at the prompt end, so a hit needs
`base mod global_chunk <= 14`: 47% of prompts for stock, 31% for compact. The compact layout is penalized. Generated
tokens are recomputed. With `offload_prompt_only=false`, decode blocks are stored too, and most victims can hit.
A load is small next to a recompute: a 32K stock context is about 3.5 GB (0.2 s at 20 GB/s), against about 19 s of
prefill. So the load cost should matter little, and the store and lookup rules should matter most.

What could make this wrong: NIXL and the offload connector may not combine on a decode node for this hybrid model;
the compact layout may not work with the connector; the host link may be slower than assumed.

## 3. Hypotheses
- H1: `prompt200` has no GO. `all200` is GO on some of D-D3, D-M1, D-M2 at read ratio 0.625, and on none at ratio 1.0.
- H0: `all200` has no GO, because evictions or the load cost remove the gain.

## 4. Setup (frozen)
- Engine model: `poolsim.PoolSim(offload=(prompt_only, cpu_bytes))`, with the E-64 rules otherwise. Default `None` is
  unchanged (V1). The GPU prefix cache is off in these modes (E-66: under 1% of re-admissions).
- Store model, per request: the stored global chunks and one interval of sliding blocks. Remote arrival stores the
  prompt global chunks and the last window. A local chunk stores the blocks it computed. A cache page is one block
  (2,621,440 bytes).
- Eviction: LRU at request level; an evicted request loses all its chunks. The rank is set at the first store and
  renewed at finish. After an eviction, a running request stores nothing until it restarts from token 0.
- Hit: `h = min(p - 1, global chunks x global block, sliding end)`, rounded down to the global block; it is a hit if
  the 64 blocks before `h` are stored, or all blocks from 0 if `h` is under 1,024 tokens.
- Load: `h / global block + 5 x min(h / 16, 64)` blocks. The time is added after the simulation, as one serial charge
  per load, at a link speed.
- Modes: `prompt200` (the vLLM default store, CPU tier 200 GB), `all200` (`offload_prompt_only=false`, 200 GB), and
  `all64` (report only, 64 GB).
- Link arms: `L20` (20 GB/s, decides), `L10` and `Linf` (report only). These speeds are assumptions (A100 PCIe 4.0 x16);
  the project has no measurement.
- Workloads: the six decode-node workloads of E-64 only. E-66 `perfect` shows no GO on unified nodes, and a change of at
  most 0.003 there, so offload cannot change those labels. Pools, seeds, horizon, warm-up, filter: the E-64 grid. Time
  models: both E-65 bracket tables, `prefill_fit.json`, the E-65 gate arms.

## 5. Decision rule (frozen; heuristics)
- Per mode, read ratio, link and workload: the E-65 A1 label, with the load charge added to each gate arm.
- Link guard: if (load + store bytes) / (time x link speed) is above 1 in any included cell of a workload at `L20`, a GO
  label of that workload becomes MID.
- Word, at read ratio 0.625 and `L20`:
  - OFFLOAD-DEFAULT-GO if any workload is GO in `prompt200`;
  - OFFLOAD-CONFIG-GO if none is GO in `prompt200` and any is GO in `all200`;
  - OFFLOAD-NO if none is GO in either.
- If V6 fails, the word is OFFLOAD-REVIEW, whatever the labels.
- Reported: hit rates, hit tokens against discarded tokens, CPU evictions, link use, labels at ratio 1.0, `L10`, `Linf`,
  `all64`, and the stock throughput with offload against stock without it.

## 6. Validity checks (a failed check makes the run INVALID; fix and rerun, no cost)
- V1: `offload=None` leaves the E-65 and E-66 results unchanged (`test_e66.py`).
- V2: conservation in every mode; every request finishes.
- V3: CPU blocks never exceed the capacity, and the per-request counts sum to the total.
- V4: the hit rule on set cases, including the alignment rule above.
- V5: with an unlimited tier and `offload_prompt_only=false`, every re-admission that passed the window hits.
- V6: `ref_e67.py`, an independent chunk-level reference (per-chunk keys, a copy of the pinned lookup and of the LRU rank
  rules), replays 6 cells for 60,000 steps. V6 passes if disagreements on hit or no hit are at most 5% of re-admissions
  and the hit tokens of the model and the reference differ by at most 10%.

## 7. Budget
Mac CPU, one core, nice 10: three modes, about 35 minutes each. No spend.

## 8. Threats to validity
1. A model, not the engine. No GPU run tests the connector on Gemma 4 or on the compact layout.
2. The combination of NIXL and the offload connector on a decode node is assumed, not tested.
3. A load is a serial charge, and the request runs at once. A real asynchronous load overlaps with compute but delays
   the request. `L20` and `Linf` bracket the time cost, not the delay.
4. The link speed is assumed. Store traffic has no time charge, only the link guard.
5. Unique prompts only. Finished requests occupy the CPU tier as in vLLM, but never get a hit.
6. The E-64, E-65 and E-66 limits stay, including the unmeasured read saving at ratio 0.625.

## 9. Files
`../gemma4_kv_capacity_model/poolsim.py` (`offload`), `e67.py`, `test_e67.py`, `ref_e67.py`, `run_e67.sh`, `results/e67/`.

## Amendments

### A1 (2026-11-10, after R-105, before any run of `e67.py`)
R-105 (reviewer; it read the pinned commit): AMEND_THEN_RUN. FACT: the Section 2 statements hold. MultiConnector can
combine NIXL and the offload connector, and a preempted request gets no tokens from NIXL. Changes (they replace Sections
4 to 8 where they differ):
- B1: an evicted request keeps its old LRU rank when it stores again (`lru.py` keeps the rank in the request context).
  `poolsim.py` now holds a persistent rank per request; a finish renews it.
- B2: V6 was underpowered. V6 now also requires at least 200 re-admissions in total and at least 20 in each cell, and
  cells that have CPU evictions. FACT: the first V6 pass (before B1; 307 re-admissions, 2 disagreements) had cells under
  20, so it does not count. Limit: `ref_e67.py` shares the arrival geometry with the model (the first live sliding block)
  and follows the model's hit for its store cursors. The lookup and the LRU are independent.
- F2: a hit is not always rounded down to the global block; it can be `p - 1`. The window test uses the real hit point,
  with the larger window for a partial last chunk.
- F3: the plan uses the connector defaults: one block per chunk, no common `block_size` (setting it asserts for this
  mixed model), tensor parallel 1. Whether the connector accepts the compact layout is not verified.
- F4 (added to Section 8): an asynchronous load holds blocks while the request waits for a few steps, and a request with
  in-flight stores is deferred. A store can stall a step when a fenced block is reused. INTERPRETATION: a few steps
  against seconds of recompute does not move the labels; the model has neither effect.
- F5: the link guard is expected to be inert (link use under 0.04 in short runs). Link speed is not the lever.
- F6: each offload label is reported next to its `none` label (E-65 bracket), so an existing GO is not credited to offload.
- F8: the model evicts a whole request; the pinned LRU evicts chunks, tails first. A lost tail also breaks the window, so
  the difference is small. V6 measures it.
- Not changed: the question, the modes, the grid, the time models, the gate, the 5-seed rule, the word.

### Run record (2026-11-10 and 2026-11-11)
- V6 attempt 1 (before B1): 307 re-admissions, 2 disagreements, cells under 20. It does not count (B2).
- V6 attempt 2 (final code, `results/e67/ref.json`): 7 cells, not 6; one cell (M2, pool 19,027, compact, `all200`)
  ran 120,000 steps, the others 60,000. The cells were chosen from a simulation-only scan to meet B2. FACT: 317
  re-admissions, 0 disagreements on hit or no hit, hit tokens equal (4,947,958), every cell at least 21 re-admissions
  and with CPU evictions. PASS.
- The first main run stopped when the app session ended (about 21:40, 2026-11-10), after `prompt200` had finished and
  `all200` had done 2 of 6 workloads. `all200` and `all64` were run again from the start with the same code. The seeds
  are fixed, so the restart cannot change a value; the two repeated `all200` workloads gave the same labels and medians.
- V1 to V5: `test_e66.py` and `test_e67.py` pass (18 tests; 45 in the project).

## 10. Result
FACT (`results/e67/`, both layouts with offload; Q_M median over 5 seeds, [min, max]; `none` is the E-65 bracket):

| Workload | `none` 0.625 | `prompt200` 0.625 | `all200` 0.625 | `none` 1.0 | `prompt200` 1.0 | `all200` 1.0 |
|---|---|---|---|---|---|---|
| D-D1 | KILL 1.051 | KILL 1.049 | KILL 1.050 | KILL 1.033 | KILL 1.029 | KILL 1.031 |
| D-D2 | MID 1.109 | MID 1.116 | MID 1.147 | KILL 1.071 | KILL 1.059 | MID 1.083 |
| D-D3 | MID 1.117 | MID 1.268 [1.199, 1.299] | GO 1.345 [1.336, 1.361] | MID 1.067 | MID 1.166 | MID 1.220 |
| D-D4 | MID 1.147 | MID 1.137 | MID 1.150 | MID 1.090 | KILL 1.082 | MID 1.094 |
| D-M1 | MID 1.178 | MID 1.196 | MID 1.216 | MID 1.115 | MID 1.113 | MID 1.132 |
| D-M2 | MID 1.203 | MID 1.238 [1.194, 1.322] | GO 1.399 [1.349, 1.422] | MID 1.102 | MID 1.109 | MID 1.255 |

- Labels at `L20`. `L10` and `Linf` give the same labels; the medians move by under 0.001. The link guard never fires:
  the largest link use in an included cell is 0.074.
- `all64` gives the same labels and the same throughput as `all200` in all 300 cell-layouts. CPU evictions differ in
  every one (about 2% more at 64 GB).
- Hit rate of re-admissions: `prompt200` stock 68.2%, compact 54.3%; `all200` stock 96.3%, compact 88.7%. Hit tokens
  against recomputed tokens: `prompt200` 50.6% and 32.0%; `all200` 94.0% and 79.1%.
- Offload also raises the stock layout (stock with offload against stock without, median over included cells, ratio
  0.625): `all200` D-D3 1.293, D-M2 1.284; `prompt200` D-D3 1.212, D-M2 1.216.
- R-106 per-seed counts: in `prompt200`, D-D3 is GO in 3 of 5 seeds and D-M2 in 2 of 5. In `all200` at ratio 1.0,
  D-M2 is GO in 2 of 5 seeds.

Word (Section 5): no GO in `prompt200`, GO on D-D3 and D-M2 in `all200`, V6 passes: **OFFLOAD-CONFIG-GO**.
H1 holds: `prompt200` no GO; `all200` GO on D-D3 and D-M2 (not D-M1) at 0.625 and none at 1.0. H0 is rejected.

INTERPRETATION:
- The GO comes from offload: the `none` labels of D-D3 and D-M2 are MID (1.117 and 1.203). `all200` reaches about the
  E-66 `perfect` bound (D-D3 1.35, D-M2 1.40), so cheap recovery carries the capacity gain into throughput.
- With the default store, the hit rule (`base mod global_chunk <= 14`) and the recompute of generated tokens hold the
  compact layout back. Its hit rate is 14 points lower than stock.
- The tier size does not matter in this model from 64 GB up (R-106 check 4): a victim is the newest admitted request,
  so it has the highest LRU rank among unfinished requests and is never the one evicted. Non-hits in `all200` come from
  the alignment rule just after arrival, which is harder for the compact global block (48).
- The load is at most 0.27% of the window time at `L20`. R-106 lowered the link to 0.25 GB/s: D-D3 and D-M2 stay GO.

## 11. Post-run review R-106 (reviewer, 2026-11-11): ACCEPT_WITH_CHANGES
- PASS: all 108 labels and medians recomputed independently match; the word; B1, F2; units of the load charge and of
  link use; H1. All runs used the final code (file times).
- CONCERN, applied here: F6 `none` labels beside each label (Section 10 table); the per-seed counts; the V6 text (7
  cells, one at 120,000 steps, chosen by a scan); the `all64` explanation.
- CONCERN, open: V6 never sees an eviction that changes a victim's hit, so it tests the window and alignment rules, not
  the eviction rules of F8. A small tier (for example 8 GB) would test the cap. Optional, free.
- CONCERN, open: the code is untracked in git. Commit it before the result is cited.
- Minor: the compact global page is charged at the padded 2.62 MB (6.7% high, conservative); the link guard uses the
  window mean and does not see bursts (a latency effect).
- Main threats: the GO needs the non-default `offload_prompt_only=false` and the unmeasured read ratio 0.625; the
  connector on Gemma 4, on the compact layout, and with NIXL on a decode node is untested; async load and store
  latency are not modelled; decode-only, unique prompts; 200 GB of pinned host memory is assumed.

Allowed wording: "In the CPU model of E-67, no decode-node workload reaches GO with the default vLLM offload store
(prompt blocks only); the best is 1.27x (D-D3, GO in 3 of 5 seeds) at read ratio 0.625. If decode blocks are also
stored (`offload_prompt_only=false`) and the compact read ratio is 0.625, D-D3 and D-M2 reach GO (1.35x and 1.40x,
against MID 1.12x and 1.20x without offload, and close to the zero-cost recovery bound of E-66); at read ratio 1.0 no
workload is GO. This is a model result with an assumed link and an unmeasured read ratio; the connector on Gemma 4 or
on the compact layout is untested."

Not allowed: "the default vLLM config gives GO"; any real-hardware speed-up; "the CPU tier size does not matter" or
"the load cost is negligible in practice" outside this model; a GO credited to offload without the flag, the read
ratio and the `none` labels; "verified against vLLM" (R-105 read the source; nothing ran it).
