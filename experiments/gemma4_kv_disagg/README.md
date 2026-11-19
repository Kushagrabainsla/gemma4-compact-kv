# E-70 stage 1: does a small patch let NIXL move the compact K=V cache from a prefill engine to a decode engine? (preregistration)

Status: CLOSED 2026-11-13, word PATCH-WORKS, GAP-CONFIRMED (Section 13). PREDECLARED 2026-11-13; S2 R-114
AMEND_THEN_RUN (A1); try 1 INVALID (ENV, A2); try 2 the result; S9 R-115 ACCEPT_WITH_CHANGES. Cost about $1 of Modal
credits.
Idea: I-182 (v2 of the paper). Journal entry: Section 6.1 E-70. Origin: E-69 (CONNECTOR-WORKS (A3) on one GPU) and the
E-70 feasibility read (`results/feasibility_2026-11-13.md`). Decision: D-123 (we, 2026-11-13: "dont ask me, go
ahead, I wont be with you for the next 4-5 hours, dont stop", in answer to the first-stage question): the first stage
only, cap $5 of Modal credits. The full 31B decode-node test is not approved (no money).
Labels: FACT, INTERPRETATION, HYPOTHESIS. Thresholds are project heuristics.

## 1. Question
In vLLM 0.33.0rc1.dev229+g3966924d5 with the E-57 compact K=V cache on Gemma 4 12B, does the stock NIXL connector move a
prompt's KV from a prefill engine (P, GPU 0) to a decode engine (D, GPU 1) byte for byte, after a patch that gives
NIXL a 2D byte view of the padded compact global page? "No": with the patch, the compact pair fails to start or to
transfer, or a transferred block differs from the P block, while the stock pair passes.

## 2. Rubber duck
- FACT (feasibility read, pinned `nixl/base_worker.py:1690-1774`): NIXL registers each attention layer as one region
  of whole pages. For a view in a shared buffer that is not page-contiguous, it needs `cache.nbytes` to be a multiple
  of `num_blocks * block_stride`. The compact global view holds 122,880 content bytes (12B; 245,760 on the 31B) per
  131,072-byte padded block, so it raises "KV cache view cannot be partitioned into NIXL regions". The stock view
  passes.
- The patch (`nixl_compact.py`): wrap `NixlBaseConnectorWorker.register_kv_caches`; for a layer view whose block holds
  fewer bytes than its block stride, pass NIXL a 2D uint8 view `(num_blocks, block_stride)` at the same address. Then
  `cache[0]` is contiguous with `nbytes == page_size_bytes`, and NIXL takes its one-segment path with region
  `(data_ptr, page, page)`. NIXL then also copies the padding (+6.7% bytes on compact global blocks). Later NIXL code
  uses these tensors only for `index_fill_` on dim 0 (stale-block zeroing) when P and D block sizes are equal (here).
- What would kill it early: the transport (NIXL with UCX between two processes inside a Modal gVisor container is
  untested; the stock pair shows it), the patch (the compact pair shows it).
- Out of scope: preemption, the offload connector, throughput, the 31B, CUDA graphs.

## 3. Hypotheses
- H1: the stock pair transfers byte for byte; the compact pair without the patch fails with the NIXL partition
  assert; the compact pair with the patch transfers byte for byte, and D decodes.
- H0: the patched compact pair fails, or moves wrong bytes.

## 4. Setup (frozen)
- Hardware: one Modal container with two NVIDIA L40S 48 GB (Gemma 4 12B needs 24 GB of weights per engine). P on GPU 0,
  D on GPU 1, two processes (each engine in process, `VLLM_ENABLE_V1_MULTIPROCESSING=0`).
- Software: the E-69 image (vLLM 0.33.0rc1.dev229+g3966924d5, `nixl==1.5.0` inside), `google/gemma-4-12B-it` rev
  `43c9d014b231474fa7212dbb48a762566c2b255d` from the Modal volume `msthesis-e69-weights`.
- Engine (both): E-57/E-69 settings: TRITON_ATTN, V2 runner, eager, BF16 KV, seed 0, prefix caching off,
  `VLLM_KV_CACHE_LAYOUT=LBHNC`, `kv_cache_memory_bytes` 8 GiB, `max_model_len` 8,704, `max_num_seqs` 8,
  `max_num_batched_tokens` 2,048. Connector: `{"kv_connector": "NixlConnector", "kv_role": "kv_both"}` (the CI form),
  `UCX_NET_DEVICES=all`, side-channel ports 5600 (P) and 5601 (D).
- Pairs (in this order): STOCK (stock layout, no patch), COMPACT-NOPATCH (compact layout, no patch; P and D only start),
  COMPACT (compact layout with the patch).
- Prompts: real text, the first 8 E-56 "needle" items (`gemma4_kv_tasks/data_e56.json`: about 7,666 tokens of a
  public-domain book with a hidden passphrase and a question; tokenized for Gemma 4; gold answer known).
- Per pair: P first runs the 8 prompts locally with 32 greedy tokens (the reference), then the 8 prompts with
  `max_tokens` 1 and `kv_transfer_params {"do_remote_decode": True}`; P hashes each request's blocks when it finishes
  (`request_finished_all_groups`, before the blocks are released) and writes the transfer parameters. D adds the 8
  requests with those parameters and 32 greedy tokens, hashes each request's blocks when its NIXL load ends
  (`_update_waiting_for_remote_kv`; the E-69 hook), and decodes.

## 5. Metrics
- Primary: per pair, the byte check: every logical block fully inside the loaded range on D, present on both sides,
  hashed over the content bytes of every layer of its group (the E-69 `block_hashes`); mismatches and checked blocks,
  global and sliding.
- Secondary (report): engine start time, the NIXL transfer time and bytes (from the logs or stats), D's tokens against
  P's local tokens (first divergence) and the gold passphrase in each answer (P local and D).

## 6. Validity checks (a failure makes the run INVALID)
- V1: both GPUs are L40S.
- V2: the code hashes equal `code_manifest.json`.
- V3: the stock pair completes all 8 requests with at least one global and one sliding block checked per request.
- V4: both engines of the compact pairs log the compact spec (the E-57 `spec` record: compact block and
  `state_content_bytes`) and the layout LBHNC.

## 7. Decision rule (frozen)
- PATCH-WORKS: V1 to V4 pass; COMPACT completes 8 of 8 requests with 0 mismatched blocks over at least 8 global and 8
  sliding checked blocks, and D finishes 32 tokens for each.
- PATCH-FAILS: V1 to V4 pass and COMPACT fails to start, fails a transfer, or has any mismatched block.
- INVALID: a V check fails (the stock pair failing means the transport or the container, not the patch: report
  INVALID (ENV)).
- Report next to the word: whether COMPACT-NOPATCH failed with the NIXL partition assert (GAP-CONFIRMED) or not
  (GAP-NOT-REPRODUCED; then the patch is not needed for this geometry).

## 8. Controls
- Positive control: STOCK (NIXL with the stock layout) must transfer byte for byte.
- Negative control: COMPACT-NOPATCH must fail at the NIXL registration (the gap).
- The byte check is the E-69 hash, which found planted changes in E-69 (C-CX).

## 9. Budget
- Modal prices (2026-11-12): L40S $1.95/h; 8 cores $0.38/h; 64 GiB $0.51/h; two L40S: about $4.79/h.
- Time cap 25 minutes per try (Modal function timeout, the watchdog); cap per try about $2.40 with the 300 s startup.
  Two tries at most: $4.80, inside the $5 cap (D-123). Expected about $1.
- Before the run: a live credit read with `modal billing summary`; start only if the credits are at least 2 x $2.40.

## 10. Threats to validity
1. One host, two GPUs, PCIe: the transfer path (UCX transports) is the one the container gives; it may be TCP.
2. 8 requests, one prompt length; the 12B (1 global KV head) not the 31B (4); no preemption, no offload connector.
3. The P hash at `request_finished_all_groups` reads blocks that NIXL has not yet read; a P-side write after it would
   make a false mismatch (none is expected: the request is finished).

## 11. Files
`nixl_compact.py` (the patch and the P and D hash hooks), `run_e70.py` (one engine role of one pair), `e70_stage.py`
(the session: hash check, pairs, processes, analysis, packing), `modal_e70.py`, `analyze_e70.py`, `test_e70.py`,
`make_manifest.py`, `code_manifest.json`.

## 12. Verification record
S3 (2026-11-13): `test_e70.py` 12 tests pass on the Mac (the byte view and its guard; a copy of the pinned NIXL region
rule that rejects the compact view and accepts the byte view and the stock view; the byte check; the word rule; the
suffixed request id; `ignore_eos`; the stop rule; the manifest and the cost cap; the stub session end to end, which
gives `PATCH-WORKS | GAP-CONFIRMED`, and a planted changed block, which gives INVALID because the stock pair fails V3).

## Amendments

### A1 (2026-11-13, after R-114 AMEND_THEN_RUN, before any run; `dossier/REVIEW_E-70_PRERUN_2026-11-13.md`)
Not changed: the question, the pairs, the prompts, the engine and connector settings, the byte check, the word classes.
- B1: vLLM renames requests inside the scheduler (`<id>-<8 chars>`), so the hooks and the outputs had different keys.
  `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1` in each engine, and the lookups also match a key that starts with
  `<id>-`.
- B2: the chat-templated needle prompts end with a short answer, so the model stops early; the P reference and the D
  requests use `ignore_eos=True`, so each decodes 32 tokens (the "32 tokens for each" rule of Section 7 stays).
- B3: a role's result is the status it writes in its JSON, not its exit code; each role ends with `os._exit` after a
  flush (no NIXL or UCX teardown at interpreter exit). The stock stop rule reads D's status.
- F1: a role that fails at start records the GPU, the compact spec and the layout setting first, so V4 covers it.
- F2: PATCH-WORKS also needs every compared request to have a global and a sliding block.
- F3: if the stock pair fails, it is run once more with `UCX_TLS=cuda_copy,tcp`, and the compact pairs then use the
  same setting (CUDA IPC under gVisor is unknown).
- N6: the gap check reads the traceback too. N8: a 300 s watchdog in D. N9: V3 also needs 0 failed loads and 8
  complete requests in the stock pair. N10: D records the load time of each request (report only).

### A2 (2026-11-13, after try 1; a transport setting that the run required; no rule changes)
Try 1 (`results/e70-20261113T134046Z`, 2 x L40S, 603 s, about $0.9): INVALID (ENV). Every role of the stock pair
failed at the NIXL agent start, before any engine had weights: `createBackend` for UCX gave `NIXL_ERR_BACKEND`.
FACT (UCX log lines): the gVisor container has no `/sys/class/net`, so UCX has no TCP; the transports it has are cma,
cuda, cuda_copy, cuda_ipc, mm, posix, self, shm, sm and sysv. With the default `ucx_error_handling_mode` "peer", UCX
refused them all ("no peer failure handler"); the F3 retry with `UCX_TLS=cuda_copy,tcp` failed because TCP does not
exist. The pinned NIXL doc names the fix (`nixl_connector_usage.md:68-75`). Changes:
- Both engines: `kv_connector_extra_config = {"ucx_backend_params": {"ucx_error_handling_mode": "none"}}` (read at
  pinned `nixl/base_worker.py:760`).
- The F3 retry of the stock pair uses `UCX_TLS=cuda_copy,sm,self` (host shared memory, no CUDA IPC), in case CUDA IPC
  between two processes does not work under gVisor.
- Spend: try 1 about $0.9 (local wall time; see the correction in Section 13); try 2 at most $2.40; inside the $5
  cap.
- R-115 F3: the feasibility read had named this setting before the run (its Section 2 config), and A1 did not apply
  it. The F3 retry branch (`UCX_TLS=cuda_copy,sm,self`) changed after R-114, was not reviewed, and did not run.

## 13. Result (try 2, 2026-11-13)
Run `results/e70-20261113T135213Z` (Modal, 2 x L40S, gVisor; session 316 s of the Modal record; engine starts 56 to
72 s). Word: **PATCH-WORKS, GAP-CONFIRMED**; V1 to V4 pass; R-115 re-derived it from the raw role records.

| Pair | Role status | Full blocks compared (global, sliding) | Mismatches | Gold answers P / D |
|---|---|---|---|---|
| STOCK | ok / ok | 3,512 (952, 2,560) | 0 | 8 / 8 |
| COMPACT-NOPATCH | failed / failed (NIXL partition assert at layer 5) | none | none | none |
| COMPACT (patch) | ok / ok | 3,192 (632, 2,560) | 0 | 8 / 8 |

- FACT: the patch replaced the 8 compact global layer views of the 12B, (8192, 1, 96, 640) bf16, by (8192, 131072)
  uint8 views at the same address. Without it, both engines raised "KV cache view cannot be partitioned into NIXL
  regions" (`cache_nbytes` 1,006,632,960 is not a multiple of 8,192 x 131,072).
- FACT (R-115): D compared every full block of the loaded prompt: per request 79 global blocks of 96 tokens (COMPACT)
  or 119 of 64 tokens (STOCK; the stock global block is 64, not 32), and 64 blocks in each of the 5 sliding groups. Not
  hashed: the partial tail of each prompt (82 tokens of the compact global group, 50 of the stock global group, 2 of
  the sliding groups). The hash sums over the layers of a group, so it cannot see a swap of two layers; the correct
  answers and the token agreement cover that.
- FACT: D received the whole prompt (hit 7,666 or 7,667 tokens = the prompt), so D did not prefill; each load ended
  3.3 to 4.5 s after D's start; D took 5.8 s in all against 8.7 s for P's prefill alone. The passphrase question sits
  in the unhashed tail, and D answered it 8 of 8.
- Report only: D's tokens equal P's local tokens for at least the first 21 tokens of every request; 3 STOCK and 2
  COMPACT requests differ later (tokens 21 to 28), in text that `ignore_eos` forces after the answer; the split is the
  same for both layouts. No P-against-P repeat was run, so the run-to-run noise of P is not measured.
- UNKNOWN: which UCX transport carried the data (no log line names it); the result holds with
  `ucx_error_handling_mode` "none" (which only turns off UCX failed-peer detection).
- Cost (R-115 F4): the Modal records give 163 s (try 1) and 316 s (try 2) of session time; the 603 s and 793 s were
  local wall times with the container start. The billing summary went from $3.17 to $4.16 of credits used, so E-70 cost
  about $1.0 (the summary can lag).

Allowed wording (R-115, STE): "With a byte-view patch, the stock vLLM NIXL connector moved 8 prompts of 7,666 tokens
from a prefill engine to a decode engine for the compact K=V cache of Gemma 4 12B. All 3,192 full blocks matched
(632 global, 2,560 sliding), and the decode engine gave the correct passphrase for 8 of 8 prompts. Without the patch,
NIXL fails at registration. The test used one Modal host with two L40S GPUs, eager mode and UCX error mode none; the
transport is not known."

Not allowed: "byte for byte" for the whole prompt; "exact" or "lossless" decode; the 31B; other hosts or networks;
any speed or memory gain; preemption or the offload connector. Upstream: stock Gemma 4 has equal pages, so the padded
attention page comes from the thesis compact spec; the patch is an upstream candidate only if a stock model gives a
padded attention page (not checked).

Post-run review: R-115 ACCEPT_WITH_CHANGES (`dossier/REVIEW_E-70_POSTRUN_2026-11-13.md`), applied here.

Upstream status (2026-11-13, after the run; `results/upstream_check_2026-11-13.md`; verified in the GitHub API and the
raw file of main): vLLM main fixed the NIXL check by PR #65685, merged 2026-11-12 17:40 UTC as e5dc65c1c (a day
before this run; the pin is older). Main now rounds the view span up to the block stride (`view_bytes =
cdiv(view_span, block_stride) * block_stride`), so the padded compact view registers as one region without the patch.
So the gap holds at the pinned commit only, and the byte-view patch is a back-port for the pin, not an upstream
contribution. Code reading (not run) also found a stock trigger at the pin: an nvfp4 KV cache with sliding-window skip
layers pads the sliding pages (Blackwell only).
