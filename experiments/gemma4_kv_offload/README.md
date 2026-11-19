# E-69: does the vLLM CPU offload connector reload preempted requests with the compact K=V cache? (engine test; preregistration)

Status: CLOSED 2026-11-13, word CONNECTOR-WORKS (A3) (Section 13). PREDECLARED 2026-11-12; S2 R-111 AMEND_THEN_RUN
(A1); S4 D-122; S5 rehearsal tries 1 to 3 (A2, A3), R-112 ACCEPT_WITH_CHANGES, reh-cc; S6 main run; S9 R-113
ACCEPT_WITH_CHANGES. Cost about $3.3 of Modal credits.
Idea: I-182 (v2 of the paper). Journal entry: Section 6.1 E-69. Origin: E-67 (OFFLOAD-CONFIG-GO, a CPU model) and
E-68 (EXPLORATORY MEAS-CONFIG-GO). Both say that the compact capacity turns into throughput only if preempted requests
reload their KV from a CPU tier. Nothing ran the connector on Gemma 4 or with the compact layout. We,
2026-11-12: "sure, go ahead and run your experiments on modal and runpod, whichever is available! use them".
Labels: FACT, INTERPRETATION, HYPOTHESIS. Thresholds are project heuristics.

## 1. Question
In vLLM 0.33.0rc1.dev229+g3966924d5 with the compact K=V cache of E-57 on Gemma 4 31B, does the stock
`OffloadingConnector` with decode blocks stored run, reload the exact bytes of preempted requests, and reload past the
prompt end (the generated tokens too)? "No": the compact engine with the connector does not run, a reloaded block
differs from the stored block, or the reload stops at the prompt end.

## 2. Rubber duck
- FACT (code reading, 2026-11-12, `results/feasibility_2026-11-12.md`, pin `3966924d566340e971a6da313e790d05492aa505`):
  - The connector supports the V2 model runner and hybrid models (`SupportsHMA`). It copies raw bytes per block:
    `unpadded_page_size_bytes` of each block, at the padded block stride. For the 31B compact global page this is
    245,760 of 262,144 B. No assert rejects Gemma 4.
  - Decode-phase sliding blocks are stored only if `prefix_cache_retention_interval=None`. The default is 0. At 0, a
    preempted decoding request reloads only up to the aligned prompt end. E-67 `all200` assumed the dense store, so
    the E-67 and E-68 GO needs this second non-default setting (to record in the journal Section 8).
  - The connector asks for the KV layout LBHNC. E-57 and E-59 ran LBNHC. A connector run silently changes the layout
    of the compact views.
  - The CPU tier is an mmap in `/dev/shm`. Startup fails if `/dev/shm` is too small. A failed `cudaHostRegister` only
    makes the copies slower.
  - The hits of re-admitted preempted requests are in `SchedulerStats.connector_prefix_cache_stats.preempted_*`. No
    log line or Prometheus metric shows them. `make_stats` returns None unless stats are on.
- What would kill it early: `/dev/shm` too small on Modal (the probe), the compact views wrong under LBHNC (V5), no
  preemption in the workload (V4), the stats or digest hooks not found at the pin (the probe and the rehearsal).
- Why this order: a CPU probe (cents), a 12B rehearsal on a cheap 48 GB GPU, then the 31B run.
- Out of scope: a decode node with NIXL (two GPUs), the E-68 read-time rerun (it needs an A100 SXM, which Modal does
  not offer), throughput claims (a unified node; E-64 shows no GO there).

## 3. Hypotheses
- H1 (primary): CONNECTOR-WORKS. Every checked reload is byte-exact; with `prefix_cache_retention_interval=None` the
  median reload covers at least 90% of the request length at preemption, so generated tokens reload. The compact
  re-admission hit rate is a few points below stock (E-67: 88.7% against 96.3%, the 48-token alignment).
- H0: the compact engine with the connector fails to start, a reloaded block differs, or the reload stops at the
  prompt end.

## 4. Setup (frozen)
- Hardware: one NVIDIA A100 80GB on Modal (the type Modal gives: PCIe in E-68). All arms run in one container, so no
  comparison crosses hosts. Rehearsal: one NVIDIA L40S 48 GB on Modal (Gemma 4 12B; 24 GB weights need 48 GB). Probe:
  a Modal CPU container from the same image.
- Software: image `vllm/vllm-openai@sha256:f255c4fad2f30ff5ee80db14d3a03afe9ae0f10d3470339864ff04745fe44cb3` (vLLM
  0.33.0rc1.dev229+g3966924d5, torch 2.14.0+cu130, Triton 3.8.0; E-57, E-59). Models `google/gemma-4-31B-it` rev
  `842da3794eaa0b77d5f08bae87a17459d91ff475` (public, Apache-2.0) and `google/gemma-4-12B-it` rev
  `43c9d014b231474fa7212dbb48a762566c2b255d`. The weights go to a Modal volume once and are reused.
- Engine (all arms, the E-57 settings): TRITON_ATTN, V2 runner, in process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`),
  eager, BF16 KV, seed 0, prefix caching off, `max_model_len` 66,048, `max_num_batched_tokens` 2,048,
  `max_num_seqs` 16, `kv_cache_memory_bytes` 15 GiB (6,144 blocks of 2,621,440 B), `disable_log_stats=False`,
  `VLLM_KV_CACHE_LAYOUT=LBHNC` (the connector layout, pinned in every arm).
- Connector (arms with "C"): `kv_transfer_config = {"kv_connector": "OffloadingConnector", "kv_role": "kv_both",
  "kv_connector_extra_config": {"cpu_bytes_to_use": 64 GiB, "offload_prompt_only": False}}` (a Python bool), no
  `block_size`; `prefix_cache_retention_interval=None`. The arm C-C0 uses the default 0.
- Arms (one engine each, run in the order of Section 9): S-R (stock, recompute preemption, no connector), S-C (stock,
  connector), C-R (compact, no connector), C-C (compact, connector), C-C0 (compact, connector, retention 0; report only; the time
  cap cuts it first). Stock arms run the workload in mode OFF and the EQ set in mode EMULV; compact arms run VIEWS.
- Workload W (the only factor that differs between arms is the arm): 12 synthetic prompts of 16,384 token ids (the
  E-57 `cap_prompt` generator, seed `"e69"`, ids in [1,000, 100,000)), greedy, `max_tokens` 3,072, `ignore_eos`, all
  submitted at once. Peak demand is about 1.5 times the stock pool, so every arm preempts. Rehearsal (12B): 12
  prompts of 8,192 tokens, `max_tokens` 1,536, pool 4 GiB, CPU tier 16 GiB, `max_model_len` 16,896.
- EQ set (every arm, before W): 8 synthetic prompts of 2,048 tokens, greedy, 128 tokens, one batch; no preemption.

## 5. Metrics
- Primary (decides): per arm C-C, the reload coverage of each checked re-admission, `h / n_pre` (hit tokens over the
  request length at its preemption); the median over re-admissions; and the V-RELOAD byte check (Section 6).
- MDE: not a speed test. Byte check: 0 mismatched blocks. Coverage: the 0.9 bar.
- Secondary (report only): preemptions; re-admission hit rate (hits over queries, tokens); load and store bytes and
  seconds, load GB/s; output tokens per second of W and the decode window; token agreement of C-C with C-R and S-C
  with S-R (first divergence per request); engine start time; C-C0 coverage; `/dev/shm` size, host registration.
- Repeats: one run of W per arm. The arms are a functional test, not a speed comparison.

## 6. Validity checks (a failure makes the run INVALID)
- V1: the GPU name contains "A100" and "80GB" (rehearsal: "L40S").
- V2: the SHA-256 of every code file on the container equals `code_manifest.json` from the design commit.
- V3: the stats wrapper returns stats in every step of every arm, and its preemption count equals the compact-engine
  counter (`ce.STATE["preemptions"]`).
- V4: W has at least 10 preemptions in each of S-C and C-C, and at least 10 checked reloads in C-C.
- V5 (layout): on the EQ set, the compact arms give the same greedy tokens as the stock arm in mode EMULV with the same
  connector setting (C-R against S-R, C-C against S-C), as E-57 EQ; and `fix_reads` > 0 in the compact arms.
- V6: the engine log names the layout LBHNC in every arm.

V-RELOAD (decides, not a validity check): at a preemption, the hook hashes every GPU block of the request, per KV
cache group and per layer (int32 sum with position weights, on the GPU). When the request is RUNNING again after a
load, it hashes the blocks for the loaded token range. Every block with both hashes must match. At most 20 reloads per
arm are checked (the first 20). A check needs at least one global and one sliding block.

## 7. Decision rule (frozen)
- CONNECTOR-WORKS (GO for the next engine step): S-C and C-C complete W; V1 to V6 pass; V-RELOAD has 0 mismatched
  blocks over at least 10 checked reloads in C-C; the C-C median coverage is at least 0.9.
- CONNECTOR-PROMPT-ONLY (PIVOT): as GO, but the C-C median hit is at most the prompt length plus 48 tokens.
- CONNECTOR-PARTIAL (PIVOT): as GO, but the median coverage is below 0.9 and the median hit is above the prompt
  length plus 48.
- CONNECTOR-FAILS (KILL for the stock connector with the compact layout): C-C does not start or crashes in two tries
  while S-C completes, or any V-RELOAD mismatch in C-C.
- INVALID: any V check fails, or S-C does not complete (the connector fails without the compact layout; then report
  CONNECTOR-FAILS-STOCK as exploratory).
- What each word means: WORKS: the E-67 mechanism exists in the engine with two non-default settings
  (`offload_prompt_only=False`, `prefix_cache_retention_interval=None`); the next step is a decode-node test.
  PROMPT-ONLY or PARTIAL: the E-67 and E-68 `all200` GO needs a connector change. FAILS: the v2 claim needs an
  upstream fix first.

## 8. Controls and baselines
- Baseline: S-C (stock with the connector) for the hit rate and coverage; C-R (compact without the connector) for the
  token agreement and the throughput report.
- A/A: none for speed (no speed decision). V5 is the twin check of E-57.
- Positive control: S-C must reload (V4 and coverage are reported for S-C too).
- Negative control: S-R and C-R have no connector, so their hit counters stay 0; C-C0 should stop near the prompt
  end (report only).

## 9. Budget
- Prices (live read of modal.com/pricing, 2026-11-12): A100 80GB $2.50/h, L40S $1.95/h, CPU $0.0472 per core-hour,
  memory $0.0080 per GiB-hour. Containers: main A100 + 8 cores + 128 GiB = $3.90/h; rehearsal L40S + 8 cores + 64 GiB =
  $2.84/h; probe 2 cores + 8 GiB = $0.16/h.
- Time caps: probe 10 min ($0.03); rehearsal 40 min ($1.89); main 100 min ($6.50). Worst case, two tries of each:
  $16.84. Cap: $17 of Modal credits (about $29.60 left by our last read). Expected: about $6.
- Order: probe, rehearsal, main. In main: setup and download, S-C, C-C (the decisive pair first), then S-R, C-R, C-C0.
- The time cap cuts C-C0 first, then C-R, then S-R. The word needs only S-C and C-C. V5 checks each complete pair
  (S-C with C-C, S-R with C-R); it passes if the S-C/C-C pair is complete and every complete pair agrees.

## 10. Threats to validity
1. A unified node, synthetic prompts, one workload. Throughput numbers are not the E-64 decode-node numbers.
2. Eager mode only. CUDA graphs (E-59) are not tested with the connector.
3. Synthetic token ids give unusual outputs. The byte check does not depend on the text.
4. A100 PCIe (Modal) and 64 GiB tier. Load GB/s is for this host only.
5. The digest hook reads the scheduler and the KV tensors in process; if vLLM moves blocks between the hash and the
   free, a false mismatch is possible. The rehearsal checks the hook on the 12B (a mismatch there stops the plan).

## 11. Files
`run_e69.py` (one arm: the engine, the driver with forced events, the hooks), `e69_hooks.py` (hash, V-RELOAD, event
log, stats sum), `stub_e69.py` (the Mac stand-in of one arm), `analyze_e69.py` (the word and the VERIFY items),
`probe_e69.py` (container facts and hook functions), `e69_stage.py` (the session: hash check, probe, telemetry,
download, arms, analysis, packing), `modal_e69.py` (the Modal stages), `make_manifest.py`, `code_manifest.json`,
`test_e69.py`, `results/`. (A1: no `session_e69.sh`; the session is `e69_stage.session`.)

## 12. Verification record
S3 (2026-11-12): see A1, "S3 record".
S4 (2026-11-13): D-122, we: "Approve, cap $17". Credit read by us before the rehearsal: "close to 30"
(dollars), at least 2 x $2.79.
Probe (2026-11-13, `results/e69-probe-*`): versions match (vLLM 0.33.0rc1.dev229+g3966924d5), manifest 11 of 11, all 8
hook functions exist, `/dev/shm` 444 GiB, cgroup memory limit 396 GiB (the requested 8 GiB is a soft value), 52 s.
Fetch: Gemma 4 12B into `msthesis-e69-weights` in 133 s (CPU); Gemma 4 31B in 283 s (CPU).

## Amendments

### A1 (2026-11-12, after R-111 AMEND_THEN_RUN, before any run; `dossier/REVIEW_E-69_PRERUN_2026-11-12.md`)
These changes replace Sections 4 to 9 where they differ. Not changed: the question, H1 and H0, the arms S-C, C-C,
S-R and C-R, the engine settings, the connector settings, V1, V2, V3, V6, the byte check by hash, the GPU types.
- B1 (W never preempts: the admission gate `full_sequence_must_fit` admits a request only if its full reservation
  fits). Forced preemption: during W the driver calls `reset_prefix_cache(reset_running_requests=True)` every 256
  engine steps (`EVENT_EVERY`). It preempts every running request through `_preempt_request`; the reload path after it
  is the normal one. An event waits for a step with no request in WAITING_FOR_REMOTE_KVS (the reset raises otherwise,
  scheduler.py:3211). The preemptions in W are forced, not from memory pressure; the report says so.
- New W (main): 6 prompts of 8,192 tokens, `max_tokens` 1,536, CPU tier 64 GiB, `max_model_len` 16,896. Rehearsal
  (12B): 6 prompts of 4,096 tokens, `max_tokens` 1,536, pool 4 GiB, tier 16 GiB, `max_model_len` 8,192. F3: the stored
  bytes of the main W are about 6 x 9,728 x 0.9 MB = 52.5 GB, 0.76 of the tier (at most 0.8).
- B2 (the coverage metric). The primary metric is the decode coverage of each re-admission of a preempted request
  with at least 512 generated tokens: c = (hits - P) / (tokens - P), clipped to [0, 1], from the lookup log (B4).
  Rule order: PROMPT-ONLY if the C-C median c is at most 0.1; else WORKS if it is at least 0.9; else PARTIAL. V4 now
  needs at least 10 such decode events and at least 10 valid V-RELOAD checks in C-C.
- B3 (the hash hooks). At a preemption only the completed blocks are hashed: blocks that end at or below
  `num_computed_tokens - num_in_flight_tokens`, after `torch.cuda.synchronize()`. The load hook wraps
  `_update_waiting_for_remote_kv` (it returns None) and reads the hit as `num_computed_tokens` before the call. A
  block is compared only if its block id changed (a stale block with the same id would hash equal); the share of
  same-id blocks is reported. The KV tensors come from `OffloadingConnectorWorker.register_kv_caches` (the manager
  views; the padding is not hashed). The cap is 24 checks (F1).
- B4 (per-event data). `PrefixCacheStats.record` is wrapped; each call with the `preempted` argument is logged with its
  tokens and hits, per phase. The `make_stats` wrapper stays for the transfer stats and V3.
- Rehearsal controls (B2, B3, F6), all on the L40S: C-C0 (retention 0) must give a median c of at most 0.1; C-CX (the
  planted control: one value of the first loaded global block is changed after the load) must give at least one
  mismatch; C-C64 only starts the 12B engine with a 64 GiB tier (128 GiB container) and must start in at most 300 s.
  The main run starts only if every one passes (VERIFY, below). C-C0 is removed from the main run.
- F2 (an end-to-end row, report only): each arm first runs one request of 2,048 tokens with 512 output tokens; the
  connector arms force one event at step 384. The tokens of C-C and S-C are compared with those of C-R and S-R. This
  stays report only: after a reload, the tail of up to one block is recomputed by the prefill path, which can change
  greedy tokens in BF16 without a fault.
- F4: the main run order is C-C, S-C, S-R, C-R, so a 31B failure of the compact path shows first. Each arm records the
  model runner attributes on kernel blocks (best effort). The 12B has 1 global KV head and the 31B has 4, so the
  rehearsal cannot clear the kernel-block path for the 31B.
- F5: V5 also needs `kv_offload_load_bytes == 0` in the EQ set of the connector arms.
- F7: the downloads run on a CPU container (`fetch-reh`, `fetch-main`) into the volume `msthesis-e69-weights`.
- F8: time caps: rehearsal 45 min (128 GiB), main 75 min (128 GiB); one arm at most 8 min (rehearsal) or 20 min
  (main); an arm starts only with at least 4 or 10 min left.
- F9: a C-C crash counts as CONNECTOR-FAILS only if its traceback is in the connector, the compact patch or the layout
  code, in two tries, while S-C completes. A crash on memory, `/dev/shm` or host registration is INVALID (ENV). Zero
  checked reloads is INVALID.
- F10, VERIFY list of the rehearsal (each is checked and written in the rehearsal note): (1) the log lines of the mmap
  file, the offload region and the host registration, and no "cudaHostRegister failed"; (2) the layout LBHNC; (3) the
  kernel-block attributes; (4) at least 10 forced preemptions with hits in S-C and C-C; (5) V-RELOAD with 0
  mismatches in C-C and at least one in C-CX; (6) C-C median c at least 0.9 and C-C0 at most 0.1; (7) load bytes above
  0; (8) the C-C64 start time; (9) the `/dev/shm` size and the cgroup memory limit; (10) the stored bytes against the
  tier (at most 0.8).
- F10, abort rules (in addition to PROTOCOL.md): stop if V1 fails; if the cost so far plus the plan is above the cap;
  if C-C crashes twice; if S-C or C-C has 0 preemptions (the session stops itself); if a rehearsal VERIFY item fails
  (no main run). A second main try runs only if the first failed for a cause that a fix removes, with the fix as a new
  amendment. The Modal function timeout is the watchdog.
- F10, money: Modal has no balance tool. Before each GPU stage we read the credits on the Modal usage page
  (as D-118); a GPU stage starts only if the credits are at least 2 times its cap.
- Budget (Modal prices of 2026-11-12): probe $0.03; each fetch about $0.16 (CPU, 30 min at most); rehearsal L40S + 8
  cores + 128 GiB = $3.35/h, cap per try $2.79 (startup 300 s + 2,700 s); main A100 80GB + 8 cores + 128 GiB = $3.90/h,
  cap per try $5.20 (300 s + 4,500 s). Worst case, two tries of each GPU stage plus probe and fetches: $16.3. Cap $17
  (unchanged). Expected about $5.
- N1: the H1 hit-rate comparison with E-67 (88.7% against 96.3%) is dropped; forced events have no arrival alignment.
- S3 record (2026-11-12): `test_e69.py` 14 tests pass on the Mac (the hash, the padding, the completed-block limit,
  the same-id rule, the planted mismatch, the event log, the stats sum, the word rule and the rehearsal VERIFY, the
  manifest, the cost caps, a wrong code file stops the session, and the stub session of all five rehearsal arms end
  to end through the real hook classes, which gives `REHEARSAL | CONNECTOR-WORKS` with every VERIFY item true). The
  stub cannot test the vLLM call order; the rehearsal does.

### A2 (2026-11-13, after rehearsal try 1; fixes that the rehearsal required; PROTOCOL.md "Rules for agents")
Rehearsal try 1 (`results/e69-reh-20261113T084436Z`, L40S, 156 s): S-C started (77 s, 48 connector layers, layout
LBHNC, a 17.18 GB tier, host registration done, stores at about 12.6 GB/s), then failed at the first forced event of
the E2E row: `reset_prefix_cache(reset_running_requests=True)` preempted the request, but its block-pool reset
refused ("Failed to reset KV cache even when all the running requests are preempted ..."). INTERPRETATION: connector
stores still held blocks. The session stopped (S-C failed), so no other arm ran. Fixes (no rule changes):
- The forced event calls the preemption part of `Scheduler.reset_prefix_cache` directly (pop each running request,
  `_preempt_request(..., drop_stale_output=True)`, clear `prev_step_scheduled_req_ids`, scheduler.py:3192-3206 at
  the pin), without the block-pool reset. Prefix caching is off, so the reset has no purpose here. The event no longer
  waits for loads to end (only a block-pool reset needs that).
- `StatsSum` reads the real offload payload `{"types", "data", "info"}` of the pinned `offloading/metrics.py`
  (counters add, histograms count and sum, gauges keep the last value). In try 1 it added the three top keys.
- The layout comes from the resolved engine config (`kv_record`, `cfg.kv_cache_layout`) when the log handler sees no
  line (vLLM sets up its own logging at import; the handler is now added after the import). The session also copies
  the VERIFY log lines (layout, mmap file, MADV_POPULATE, offload region, host registration) of each arm into its
  record.
- VERIFY facts from try 1: `MADV_POPULATE_WRITE` is not supported in the Modal container, so the tier is pre-filled
  page by page (17.18 GB in about 5 s); the CPU-tier mmap and the host registration worked.
- 15 tests pass (a new test of the real payload). Try 1 cost about $0.3 (L40S, under 5 minutes with the startup).
- Rehearsal try 2 (`results/e69-reh-20261113T084945Z`, 162 s): S-C started (70 s, layout LBHNC). The E2E row ran
  with the A2 forced event at step 384; the re-admission queried 2,431 tokens and hit 2,368 (the 2,048-token prompt
  and 320 generated tokens); one load of 374 MB at 12.8 GB/s. Then the arm failed at the EQ set:
  `set_mode("EMULV")`; the stock twin EMULV is mode VIEWS on the stock engine (run_e57.py:831). Fixed. Try 2 cost
  about $0.3. With try 3, the rehearsal spend is at most $0.6 + $2.79, inside the $17 cap with two main tries.

### A3 (2026-11-13, after rehearsal try 3; before the main run; checked by R-112)
Rehearsal try 3 (`results/e69-reh-20261113T085417Z`, L40S, 886 s, about $0.9): all five arms completed. Word under the
A1 rule: `REHEARSAL | CONNECTOR-FAILS`, because V-RELOAD found changed blocks in S-C (165) and C-C (185). V1 to V6
pass (V5: the compact and EMULV EQ tokens are equal under LBHNC with the connector). VERIFY: C-C0 median c 0.0 (prompt
only), C-CX planted change found, C-C64 start 65 s with a 68.7 GB tier (18 s of page-by-page fill), loads above 0, no
hook error, layout LBHNC, the mmap and host registration lines present. Decode coverage: S-C median 0.962, C-C 0.961
(36 decode events each, 0 zero-hit events). Loads at 13.3 GB/s. 49 forced preemptions per arm.
- FACT (`results/e69-reh-20261113T085417Z/a3_analysis/`): every changed block is in a sliding group; first reloads
  have 0 changed blocks in S-C and C-C (6 in C-CX: the planted values); at every later reload the changed blocks lie in
  [previous hit, previous n_pre) of the same request, the tail that vLLM recomputed by prefill after the previous
  reload. In C-C the 185 changed blocks equal the 185 fully stored sliding blocks in those ranges; in S-C 165 of 170.
  No reload has more changed blocks than such blocks; none of the listed changed blocks lies outside a range.
- FACT (pinned `kv_offload/cpu/manager.py` `prepare_store`): a key that is already in the CPU tier is not stored again.
- INTERPRETATION: the CPU tier keeps the first (decode-written) copy of a block; after a reload the GPU recomputes the
  tail with the prefill kernel, which gives slightly different BF16 values; at the next reload the CPU copy comes back.
  The reload returns the stored bytes; the GPU and CPU copies differ only where the GPU holds a recomputed version.
  The global group shows no effect because the hit is aligned to its larger block, so its recomputed tail is never a
  full stored block. This is a property of recompute plus store deduplication, not a corruption.
- Change 1 (V-RELOAD): a block fully inside [previous hit, previous n_pre) of the same request is counted apart
  ("recomputed", report only). Every other block decides as before (any mismatch is CONNECTOR-FAILS). The previous
  range is kept for every load, checked or not. The lists of changed blocks now hold up to 64 entries.
- Change 2 (VERIFY 10): the rehearsal stored 18.4 GB into a 17.2 GB tier (1.07, above 0.8; the coverage was not
  hurt). The main run also stores the EQ and E2E prompts (about 18 GB on the 31B), so the main tier is 96 GiB (about
  0.68) and the main container 160 GiB. Main cap per try: $5.54. Spend so far about $1.7; the plan stays inside $17.
- Re-analysis of try 3 under change 1 (exploratory; the counts above): C-C and S-C have 0 changed blocks outside the
  recomputed ranges, so the rehearsal word would be `REHEARSAL | CONNECTOR-WORKS`, and C-CX still finds its 6 planted
  values. The main run uses change 1 from its start, so its word is not exploratory.
- Not changed: the question, the coverage rule (0.1 and 0.9), V1 to V6, the arms, W, the controls.
- Report (new): the recomputed-version blocks are a finding of their own: after two preemptions with recompute, the
  reloaded KV of those blocks is the first version, not the KV the GPU held at the second preemption.
- 17 tests pass (two new tests of change 1).
- R-112 (reviewer, ACCEPT_WITH_CHANGES, no blocking finding; `dossier/REVIEW_E-69_A3_2026-11-13.md`). It re-derived the
  counts (C-C 185 of 185 in 24 of 24 reloads; S-C 165 of 170, the gap is one block that ends at the previous n_pre;
  excluded share about 2% of checked blocks) and confirmed the mechanism in the pinned code (`prepare_store` skips
  present keys; recomputed blocks get the same hash keys; the tail is recomputed by the prefill path; the hit aligns
  to the LCM of the block sizes). Applied:
  - F1, the falsifier: the hook keeps the first hash seen of each (group, block index) of a request. A block in the
    recomputed range passes only if it equals the GPU copy at this preemption or the first version; a third value, or
    a block without a first hash, is a mismatch and decides (fail safe). So the claim is measured, not assumed.
  - F2: a short check stage `reh-cc` runs C-C alone on the 12B with the new hook code before the main run.
  - F3: the word carries the tag "(A3)", and each arm reports `a1_strict_mismatch` (every changed block, as A1) next
    to it. With F1 in place the main word is not labelled exploratory.
  - F4: a Mac test of a third value inside the range (it decides) and of a block without a first hash.
  - F5 (note): a re-admission with zero hits leaves the range of the earlier load; this can only give a false failure.
  - N: each arm reports the tier fill (stored bytes over the tier); loads are async (`sync_load_budget_bytes` 0).
  - 18 tests pass.
- Check stage `reh-cc` (2026-11-13, `results/e69-reh-cc-20261113T092622Z`, L40S, 298 s, about $0.3): C-C alone on the
  12B with the R-112 hook code. 24 checked reloads: 0 deciding mismatches; 235 blocks in recomputed ranges, all equal
  to their first version (`recomputed_first_version` 235, `recomputed_third` 0); A1-strict count 235. Median decode
  coverage 0.949 (24 decode events), loads at 24.5 GB/s, tier fill 1.05. The word INVALID is expected (S-C did not
  run in this stage). INTERPRETATION: the recompute-version explanation of A3 holds as a measurement.
- Credit read before the main run: `modal billing summary` (Modal CLI 1.6.3), live: metered $1.79 this period, $1.73
  from credits, so about $28.3 of the $30 credits remain, above 2 x $5.54. Our read was "close to 30".

## 13. Result (S6 main run, 2026-11-13; S8 and S9)
Run `results/e69-main-20261113T093148Z`: Modal, one NVIDIA A100-SXM4-80GB (Modal gave an SXM A100 this time, not the
PCIe of E-68), driver 610.57.04, region ap; session 1,438 s; arms C-C 402 s, S-C 346 s, S-R 316 s, C-R 331 s. Cost:
the main run about $1.7; E-69 in total about $3.3 of Modal credits (billing summary; E-68 was $0.11).

**Word: CONNECTOR-WORKS (A3).** V1 to V6 pass. R-113 re-derived the word from the raw records.

| Item (FACT) | C-C (compact, connector) | S-C (stock, connector) |
|---|---|---|
| Forced preemptions in W | 64 | 80 |
| Median decode coverage c (decode events) | 0.977 (45) | 0.984 (60) |
| Zero-hit re-admissions | 0 | 0 |
| Checked reloads; blocks compared | 24; 11,676 | 24; 13,990 |
| Deciding mismatches | 0 | 0 |
| Recomputed-range blocks: first version / equal to the GPU copy / third value | 135 / 0 / 0 | 45 / 5 / 0 |
| A1-strict count (every changed block) | 135 (1.2%) | 45 |
| Load speed | 23.8 GB/s | about 24 GB/s |
| Tier fill (stored bytes over the 103 GB tier) | 0.66 | 0.68 |

- The 31B compact engine (block 48, 4 global KV heads, `state_content_bytes` 1,280) started and ran with the connector;
  the kernel-block risk of R-111 F4 did not occur. The EQ tokens of C-C and S-C are equal (V5).
- Report only, with R-113 F1: the E2E rows of C-C and S-C equal those of C-R and S-R for 512 tokens, but the outputs
  of these synthetic prompts are degenerate (the E2E row is 512 copies of one token; the EQ and W outputs have 1 to 4
  distinct tokens). Token agreement here has no power; only the byte check supports exactness.
- Report only, with R-113 F5: W took 116 to 120 s with the connector and 226 to 237 s with recompute. This is not a
  throughput result: the experimenter sets the preemption rate, the six requests recompute in one burst, and prefix
  caching is off.
- R-113 F2: the 24 checks cover the first four events of each request (231 to 628 generated tokens); the 39 later
  re-admissions are not byte checked; about 27% of the compared blocks are decode-written (R-113 estimate).
- R-113 F3: under the A1 rule the word would be CONNECTOR-FAILS (135 changed blocks); every one of them is a sliding
  block in a recomputed range and equals its first version. In S-C, 5 such blocks equal the GPU copy, so the tier
  keeps the version that was stored first.
- R-113 F4: the main run shows that the two non-default settings are enough; that the default retention 0 reloads only
  the prompt comes from the rehearsal control C-C0 on the 12B (median c 0.0), not from the 31B.

What it means (INTERPRETATION): the E-67 mechanism exists in the engine. On Gemma 4 31B with the compact K=V cache,
the stock connector stores decode blocks and reloads a preempted request with its generated tokens, with the exact
stored bytes, if both `offload_prompt_only=False` and `prefix_cache_retention_interval=None` are set. Measured now:
decode blocks reload, the compact layout works with the connector, a load runs at about 24 GB/s (E-67 assumed 20).
Not measured: the read ratio (E-68), preemption under memory pressure, a decode node with NIXL, throughput, CUDA
graphs. The E-67 and E-68 GO stays a model result.

Allowed wording (R-113, STE): "The stock vLLM OffloadingConnector ran with the compact K=V cache of Gemma 4 31B on one
A100. With `offload_prompt_only=False` and `prefix_cache_retention_interval=None`, it reloaded generated tokens
(median coverage 0.977). In a forced-preemption test, 24 checked reloads matched byte for byte. The exception is
blocks that vLLM recomputed after an earlier reload. Those came back as the version stored first. The load speed was
about 24 GB/s. The test used eager mode, one unified node and synthetic prompts. It did not test a decode node with
NIXL, memory-pressure preemption, or throughput."

Not allowed: a throughput or speed-up claim from W; "the outputs are equal after a reload" as evidence; "the default
connector works" (it needs the two settings); any claim about memory-pressure preemption or a decode node.

Recompute-version effect (R-113: a determinism note, not a correctness concern): after two preemptions, the reloaded
KV of the blocks recomputed after the first reload is the first stored version, not the KV the GPU held at the second
preemption. Report it as a limit in the paper.

Post-run review: R-113 ACCEPT_WITH_CHANGES (`dossier/REVIEW_E-69_POSTRUN_2026-11-13.md`); its changes are applied in
this section.
