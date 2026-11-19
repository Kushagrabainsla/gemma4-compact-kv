# E-70 stage 2: on a decode engine fed by NIXL, does the offload connector reload preempted requests of the compact K=V cache? (preregistration)

Status: CLOSED 2026-11-14, word RELOAD-WORKS (A3) (Section 11). PREDECLARED 2026-11-13; S2 R-116 AMEND_THEN_RUN
(A1); S4 D-124 (we, "Yes, run stage 2"); S6 one try; S9 R-117 ACCEPT_WITH_CHANGES. Cost about $0.6.
Idea: I-182 (v2). Origin: E-69 (the offload connector reloads on one engine, CONNECTOR-WORKS (A3)) and E-70 stage 1
(NIXL moves the compact cache with the byte-view patch, PATCH-WORKS). E-67 assumes both together on a decode node.
Labels: FACT, INTERPRETATION, HYPOTHESIS. Thresholds are project heuristics.

## 1. Question
On a decode engine D that receives each prompt's KV from a prefill engine P over NIXL, with the offload connector next
to NIXL (MultiConnector) and the E-57 compact cache on Gemma 4 12B, does a forced preemption on D reload the request
from the CPU tier with its generated tokens and the exact bytes? "No": D does not start or crashes, a reloaded block
differs (under the E-69 A3 rule), or the reload stops at the prompt end.

## 2. Rubber duck
- FACT (feasibility read Section 2): after a NIXL load, `do_remote_prefill` is cleared, so at a re-admission NIXL
  returns 0 tokens and only the offload connector can load; MultiConnector gives the load to the first child with
  tokens; the offload connector stores the blocks of every scheduled request, also the ones that NIXL wrote.
- New against E-69: the first blocks of each request on D were written by NIXL, not computed on D. The offload
  connector must store them (as the decode blocks) and reload them.
- New against stage 1: the decode runs long (preemptions need decode tokens), and the D connector is a MultiConnector.
- Kill early: the MultiConnector config with the compact patch fails at start; the offload connector stores nothing for
  NIXL-arrived blocks (coverage at the prompt end or 0).

## 3. Hypotheses
- H1: RELOAD-WORKS on COMPACT2 and STOCK2: median decode coverage at least 0.9; 0 deciding mismatches (A3 rule).
- H0: the reload of NIXL-arrived requests fails, stops at the prompt end, or moves wrong bytes.

## 4. Setup (frozen)
- Hardware, image, model, P engine, prompts: as stage 1 (two L40S on Modal; Gemma 4 12B; the 8 needle prompts;
  `ucx_error_handling_mode` none; the byte-view patch on both engines of the compact pair).
- D connector: `{"kv_connector": "MultiConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"connectors":
  [NixlConnector kv_both with the UCX setting, OffloadingConnector kv_both with cpu_bytes_to_use 16 GiB and
  offload_prompt_only False]}}`, `prefix_cache_retention_interval=None`.
- D workload: the 8 requests with P's transfer parameters, `max_tokens` 1,536, `ignore_eos`, greedy; D is stepped by
  the E-69 driver, and every running request is preempted every 256 steps (the E-69 A2 forced event).
- Pairs: STOCK2 (stock layout) and COMPACT2 (compact layout with the patch), in this order.
- Hooks on D: the stage-1 NIXL byte check (the first load of each request, against P's hashes) and the E-69 hooks
  (the lookup log for the decode coverage, V-RELOAD with the A3 rule and the first-version falsifier, the stats).

## 5. Metrics
- Primary (COMPACT2): the median decode coverage c of re-admissions with at least 512 generated tokens; V-RELOAD
  deciding mismatches (at most 24 checked reloads); the stage-1 NIXL byte check.
- Secondary: the same for STOCK2; preemptions; load bytes and GB/s; the first lookup of each NIXL-arrived request on
  the offload tier (expected 0 hits, as the prompts are new to D's tier); the gold answer in D's text.

## 6. Validity checks
- V1: both GPUs are L40S. V2: the code hashes match. V3: STOCK2 completes with at least 10 forced preemptions, at least
  10 decode events and 0 NIXL byte mismatches. V4: the compact spec and the layout LBHNC on both compact engines.

## 7. Decision rule (frozen)
- RELOAD-WORKS: V1 to V4 pass; COMPACT2 has at least 10 decode events and at least 10 valid checked reloads, median c
  at least 0.9, 0 deciding mismatches, and 0 NIXL byte mismatches.
- RELOAD-PROMPT-ONLY: as above but median c at most 0.1 (tested first). RELOAD-PARTIAL: between.
- RELOAD-FAILS: COMPACT2 crashes in connector, patch or layout code, or any deciding mismatch.
- INVALID: a V check fails.

## 8. Budget
Two L40S on Modal ($4.79/h); a 25-minute cap per try ($2.40); at most two tries ($4.80). Expected about $1.

## 9. Threats
Forced preemptions (not memory pressure); one host; 12B only; 8 requests; eager; the transport is unknown.

## 10. Files
`run_e70.py` (P as stage 1; D role `d2`), `nixl_compact.py`, the E-69 `run_e69.py` and `e69_hooks.py` (imported),
`e70_stage.py` (pairs STOCK2, COMPACT2), `analyze_e70s2.py`, tests in `test_e70.py`.

## Amendments

### A1 (2026-11-13, after R-116 AMEND_THEN_RUN, before any run; `dossier/REVIEW_E-70S2_PRERUN_2026-11-13.md`)
Not changed: the question, the hypotheses, the pairs, the prompts, the thresholds 0.9 and 0.1.
- B1: D's `max_model_len` is 9,216 (prompt 7,667 + 1,536 output tokens; 8,704 would stop each request at about
  1,038 tokens). P keeps 8,704 (the NIXL compatibility hash does not include it).
- B2: a forced event fires when every running request has at least the next multiple of 256 generated tokens (engine
  steps also count the idle steps of the NIXL loads, so the first event would come at about 1 token, where the reload
  is near zero by design). This gives about 4 decode events per request.
- F1: a NIXL check counts only with blocks compared, a full-prompt hit and no failed load; V3 and the COMPACT2 word
  need 8 of 8 complete requests; RELOAD-WORKS also needs 0 zero-hit re-admissions and a minimum c of at least 0.5;
  RELOAD-PROMPT-ONLY needs 0 zero-hit re-admissions (zero hits are PARTIAL).
- F2: the MultiConnector nests the stats by connector name; the stats reader takes the offload child's payload.
- F3: CPU tier 32 GiB; container memory 96 GiB; cap per try $2.53 (two tries $5.06, within the $5 cap if the first
  try ends early; a second try starts only if the spend so far plus $2.53 stays under $5).
- Notes: P stops 20 s after D ends; the metric "first offload lookup of each NIXL-arrived request" is dropped (the
  hooks cannot measure it); a hang (the D watchdog) is INVALID, not RELOAD-FAILS.
- 15 tests pass (new: the D engine config, the event rule, the stats reader, the new word cases).

## 11. Result (2026-11-14)
Run `results/e70s2-20261114T003825Z` (Modal, 2 x L40S; session 364 s: STOCK2 195 s, COMPACT2 161 s; credit read
before the run: at least $26 left). Word: **RELOAD-WORKS (A3)** (the V-RELOAD rule of E-69 A3 with its first-version
falsifier; the analysis printed RELOAD-WORKS without the tag, R-117 F1). V1 to V4 pass; R-117 re-derived it from the raw
records.

| Item (FACT) | COMPACT2 | STOCK2 |
|---|---|---|
| NIXL blocks compared / mismatches (8 of 8 requests) | 3,192 / 0 | 3,512 / 0 |
| Forced preemptions (5 events x 8 requests at 256 to 1,280 generated tokens) | 40 | 40 |
| Decode events; median c (min, max); zero-hit events | 32; 0.953 (0.885, 0.986); 0 | 32; 0.939 (0.895, 0.961); 0 |
| Checked reloads (the first 3 events); blocks compared; deciding mismatches | 24; 9,192; 0 | 24; 10,128; 0 |
| Recomputed-range blocks: first version / third value; A1-strict count | 160 / 0; 160 | 240 / 0; 240 |
| Loads; stored and loaded bytes | 24.7 GB/s; 7.2 and 16.9 GB | 24.7 GB/s; 7.5 and 18.9 GB |
| Complete requests; gold answers | 8; 8 | 8; 8 |

- FACT (R-117): about 62% of the checked blocks lie in the range that NIXL wrote on D (the prompt's global blocks 0 to
  78 and the last 64 sliding blocks), so the offload connector stored and reloaded NIXL-written blocks too.
- FACT (R-117): the coverage loss is the alignment only: every hit equals the request length rounded down to 96
  tokens (compact) or 64 (stock); the summed hits over the 40 preempted lookups equal that prediction exactly
  (335,616 and 335,360 tokens).
- Report only (R-117): the 24 checks cover events 1 to 3 (256, 512, 768 tokens); events 4 and 5 are not byte checked.
  The passphrase comes in the first tokens, before any preemption, so the run makes no claim that the output is
  unchanged after a reload. The tier fill was 0.21 (COMPACT2) and 0.22 (STOCK2), so eviction was not tested. The
  14,430 and 16,650 idle steps before the first event are the NIXL load wait (do not read `steps` or
  `decode_seconds` as decode data). P ended by itself (exit 0) in both pairs.

What it means (INTERPRETATION): on the P/D path that E-67 models, with the compact cache, the stock NIXL connector (with
the byte-view back-port at the pin) and the stock offload connector together give the E-67 mechanism: a preempted
request on the decode engine reloads from the CPU tier with its generated tokens and the stored bytes, instead of a
local prefill. Not shown: the 31B, memory-pressure preemption, throughput, eviction, a real network.

Allowed wording (R-117, STE): "Two L40S GPUs ran Gemma 4 12B with the compact K=V cache. A prefill engine sent 8
prompts to a decode engine over NIXL, with a byte-view patch. The decode engine used MultiConnector with the offload
connector. Forced preemptions reloaded generated tokens. The median coverage was 0.95, with the hit rounded down to 96
tokens. In 24 checked reloads (the first 3 events), 9,192 blocks matched, including blocks that NIXL wrote. Blocks
recomputed after an earlier reload came back as the first stored version. The load speed was about 24.7 GB/s. The test
used eager mode, one host, synthetic decode and a pinned vLLM commit. The transport is not known."

Not allowed: "byte for byte" without the recompute exception; "outputs equal after a reload"; memory-pressure
preemption, throughput, the 31B, other hosts or a real network; "checked at every event"; "the patch is needed
upstream" (main fixed the check, PR #65685).

Post-run review: R-117 ACCEPT_WITH_CHANGES (`dossier/REVIEW_E-70S2_POSTRUN_2026-11-14.md`), applied here.
