# E-70 feasibility: compact K=V cache on a P/D decode node (NIXL + OffloadingConnector), vLLM pin 3966924d5

Date: 2026-11-13. Code and docs reading only. No GPU, no spend, no repo file changed.
Labels: FACT (read in the pinned code, a pinned doc or a repo record), INTERPRETATION (my reading or estimate),
UNKNOWN (needs a probe).

Path prefixes used below:
- `N/` = `scratchpad/e70pin/src/vllm/distributed/kv_transfer/kv_connector/v1/nixl/`
- `M/` = `scratchpad/e70pin/src/vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py`
- `O/` = `scratchpad/e67pin/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`
- `R/` = `scratchpad/pin3966/raw/`
- `T/` = `scratchpad/e70pin/src/tests/v1/kv_connector/`
- `D/` = `scratchpad/e70pin/src/docs/features/`
The scratchpad is `/tmp/scratch`.
All files under `e70pin/src/` were fetched from raw.githubusercontent.com at the pin.

## Summary (the answer first)

1. The stock NIXL connector supports hybrid models and Gemma 4 (CI runs gemma-4-E2B). But it will very likely REJECT
   the compact global page at startup. FACT: the region registration needs `cache.nbytes` to be a multiple of
   `num_blocks * block_stride`. The compact global view has 245,760 content bytes in a 262,144-byte padded block, so
   the check raises `AssertionError("KV cache view cannot be partitioned into NIXL regions")`. A small patch (a 2D byte
   view of the padded rows) should fix it. Do not use the BLHNC layout as a work-around: my reading is that it
   silently transfers only one layer slot of the global group.
2. MultiConnector can combine NIXL and the offload connector on the decode node (CI tests this pair). A preempted
   request gets nothing from NIXL again (`do_remote_prefill` is cleared), so the offload connector is the only source.
   The offload connector stores the blocks that NIXL wrote (global prefix plus the last sliding window) once the
   request runs. So a reload is possible, as E-67 modelled.
3. One container with two GPUs and two engine processes is the documented same-host setup. The pinned image has
   `nixl==1.5.0`, `nixl-cu12` and `nixl-cu13` (pip freeze of the same image). UCX comes inside the NIXL wheel
   (INTERPRETATION). Same-host transfer without RDMA is supported. Modal (gVisor) CUDA IPC between two processes is
   UNKNOWN; Runpod Docker Pods are the safer choice.
4. `vllm.general_plugins` is loaded in every vLLM process. The simplest path needs no plugin: run each engine in
   process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`, as E-69) from its own driver script that calls `install()` first.
5. One GPU can host both engines for a functional test (the NIXL test scripts do this). It cannot give a clean decode
   throughput, because P and D share the SMs. The main run needs two GPUs.
6. Drive D with a closed loop that keeps 1 to 2 requests waiting on D; the driver is also the proxy. Measure on D only.
   A risk: one prefill GPU likely cannot feed a saturated D for M2 (and only just for D3).
7. Top risks: (a) the NIXL padded-page assert or a silent wrong transfer; (b) transport (UCX inside the container);
   (c) the workload does not reach the E-64 regime (P-bound, no natural preemption, no offload hit after NIXL arrival).
   Estimate: 12B rehearsal 2 to 3 GPU-hours; 31B main 8 to 12 GPU-hours (more if M2 needs a second prefill GPU).

## 1. NIXL connector: hybrid support, page geometry, layout

FACT:
- `NixlBaseConnector(KVConnectorBase_V1, SupportsHMA)` (`N/connector.py:78`); `supports_divergent_local_hybrid_hits`
  is True (`N/connector.py:84-85`). Sliding groups transfer only the in-window tail
  (`N/base_scheduler.py:253-294`, `get_exchange_clipped_blocks`).
- Gemma 4 is in the NIXL CI sweep: `google/gemma-4-E2B-it` "SW + cross-layer KV sharing"
  (`T/nixl_integration/config_sweep_accuracy_test.sh:41-42`; `config_sweep_accuracy_core_test.sh:41-49`).
  The compatibility matrix marks SWA as supported for dense models (`D/nixl_connector_compatibility.md`).
- Required layout: LBHNC for non-MLA models (`N/connector.py:149-166`). The offload connector also asks for LBHNC
  (`e67pin/.../offloading_connector.py:236-239`). MultiConnector raises only if children ask for different layouts
  (`M/:641-667`). So the pair agrees on LBHNC, the layout E-69 used.
- NIXL does not split K and V by `head_size`: "K and V are packed into the content dim, so each attention layer is a
  single NIXL region whose block transfers as one unit" (`N/base_worker.py:1599-1601`). Head splits happen only for
  heterogeneous TP (not used here).
- The page size it uses is `layer_spec.page_size_bytes` (`N/base_worker.py:1630-1635`), which is `page_size_padded`
  when set (`R/kv_cache_interface.py:597-601`). For the compact 31B global spec: padded 262,144, content
  245,760 (`state_content_bytes` 1,280; E-69 record `results/e69-main-20261113T093148Z/C-C/run_e69.json`). 12B: padded
  131,072, content 122,880 (`results/e69-reh-cc-20261113T092622Z/C-C/run_e69.json`).
- All layers are views of ONE int8 buffer (`R/worker_utils.py:564-627`, `buf = torch.zeros(buf_size, ...)` at :596).
  KV cache tensors of different groups overlay the same bytes (docstring :570-575; the NIXL unit test builds the same
  structure, `scratchpad/e70pin/src/test_nixl_desc_geometry.py:619-643`).
- The registration branch for an attention layer (`N/base_worker.py:1690-1774`):
  - `block_stride = cache.stride(0) * element_size` = 262,144 under LBHNC.
  - `registration_len = storage.nbytes()` = the whole shared buffer (about 10 x num_blocks x 262,144 for the 31B),
    so `storage_is_block_major` (`:1702-1704`) is False.
  - `page_contiguous` needs `cache[0].nbytes == physical_page_size` (`:1707-1711`); the compact view gives 245,760, so
    it is False.
  - So the code reaches the `else` branch (`:1752-1762`): `segment_bytes = num_blocks * block_stride`; `cache.nbytes`
    (numel times element size, which ignores strides) is `num_blocks * 245,760`; the remainder is not 0, and it raises
    `AssertionError("KV cache view cannot be partitioned into NIXL regions ...")`.
  - For the stock spec the remainder is 0, one segment, region `(data_ptr, page, page)`. So only the compact arm fails.
- vLLM's own padded-page tests cover MLA and Mamba pages, not a padded non-MLA attention page
  (`test_nixl_desc_geometry.py:561-575, 605-618`; `T/unit/test_nixl_connector_hma.py:118, 2106`). INTERPRETATION:
  the compact spec is an untested geometry for NIXL.
- Other checks that the compact spec passes (both sides compact): block size is the LCM of group block sizes
  (`N/base_worker.py:656-666`), 48 for compact; Triton accepts it, so no kernel split (`R/worker_utils.py:515-540`);
  handshake block lengths must match per region (`N/base_worker.py:2911-2985`). The compatibility hash does NOT
  include the KV spec or layout (`N/metadata.py:146-205`), so a stock P with a compact D is caught only by the block
  length check. E-70 must check that both engines log the same compact spec.

INTERPRETATION:
- Fix (about 15 lines in `compact_engine`): wrap `NixlBaseConnectorWorker.register_kv_caches` (or MultiConnector's)
  and replace each compact global view by a 2D byte view `as_strided(buf, (num_blocks, page_size_padded), (padded, 1))`
  at the same data pointer. Then `page_contiguous` is True, `cache.nbytes % segment_bytes == 0`, and the region is
  `(data_ptr, 262,144, 262,144)`. NIXL then copies the padding too (+6.7% bytes on global blocks). The first layer that
  `TransferTopology` reads (`N/base_worker.py:1529`) is a sliding layer, so its shape is unchanged.
- Do NOT use `VLLM_KV_CACHE_LAYOUT=BLHNC` to avoid the assert. Under BLHNC the compact global view goes to the
  "whole storage row" branch (`:1736-1747`) with `base_addr` = the buffer start. That address is already registered by
  the first sliding layer, so the dedup at `:1779-1789` merges it into that region and keeps the sliding `block_len`
  (it updates `block_len` only for MLA). The global group would then transfer only slot 0. This would be silent wrong
  KV, not a crash.
- Add a byte check to E-70: hash the global and sliding blocks of a request on P before the send and on D after the
  load (the E-69 hash hook), for at least 20 requests.

## 2. MultiConnector on the decode node; preemption; what the offload connector stores

FACT:
- MultiConnector is `SupportsHMA` and asserts that every child supports HMA when HMA is on (`M/:133, 157-170,
  190-197`). Lookups go through every child in config order; the first child with tokens > 0 owns the load
  (`M/:451-470`). Every other child still gets `update_state_after_alloc(request, blocks, 0)` (`M/:484-494`). Saves go
  to all children.
- CI tests NIXL + Offloading in one MultiConnector, P and D on two GPUs, `kv_role` `kv_both`, LBHNC
  (`T/nixl_integration/run_multi_connector_accuracy_test.sh:58-69, 195-240`; edge cases with decode-side offload hits:
  `run_multi_connector_edge_case_test.sh:5-13, 45-56`; CI steps in `.buildkite/test_areas/disaggregated.yaml:305-386`).
  The default model there is Qwen3-0.6B, not a hybrid model.
- NIXL on D: `get_num_new_matched_tokens` returns the prompt count only while `do_remote_prefill` is True
  (`N/pull_scheduler.py:59-65`). `update_state_after_alloc` sets `do_remote_prefill = False` after the first
  allocation ("Only trigger 1 KV transfer per request", `N/pull_scheduler.py:187-189`). A preemption sets
  `num_computed_tokens = 0` and puts the request back in the waiting queue (`R/scheduler.py:1799-1843`). So at
  re-admission NIXL returns 0 (R-105, R-99 B1 confirmed).
- The offload connector then does its normal lookup; if it hits, MultiConnector assigns the load to it.
- What the offload connector stores on D: it tracks block ids "regardless of num_external_tokens ... blocks allocated
  by other connectors in a MultiConnector setup" (`O/:1398-1407`). Its store jobs run for every request in
  `num_scheduled_tokens` with `num_tokens_after_batch = num_computed_tokens + scheduled` (`O/:1859-1883`). There is
  no check of who wrote the blocks. A null block id (out-of-window sliding) is skipped (`O/:1957-1963`).
- When the NIXL load ends, `_update_waiting_for_remote_kv` sets `num_computed_tokens` (minus one replay token) and
  calls `cache_blocks` (`R/scheduler.py:3465-3515`). So, from the scheduler's view, the NIXL blocks are computed
  tokens.

INTERPRETATION:
- On D, the offload connector stores the global prefix and the last sliding window of the prompt in the first step
  that schedules the request (after the NIXL load completes, so the bytes are valid). With
  `offload_prompt_only=False` and `prefix_cache_retention_interval=None`, it stores decode blocks too (E-69). So a
  preempted request can reload past its prompt. This is E-67's `offload_arrive` model
  (`gemma4_kv_capacity_model/poolsim.py:158-159`). The hit is aligned to the global chunk (48 tokens compact); the tail
  is prefilled locally on D.
- Exact config (D), with the CI child order:
  ```
  {"kv_connector":"MultiConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[
    {"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_buffer_device":"cuda",
     "kv_connector_extra_config":{"ucx_backend_params":{"ucx_error_handling_mode":"none"}}},
    {"kv_connector":"OffloadingConnector","kv_role":"kv_both",
     "kv_connector_extra_config":{"cpu_bytes_to_use":<tier bytes>,"offload_prompt_only":false}}]}}
  ```
  plus `prefix_cache_retention_interval=None`, `VLLM_KV_CACHE_LAYOUT=LBHNC`, `UCX_NET_DEVICES=all`, and a distinct
  `VLLM_NIXL_SIDE_CHANNEL_PORT` per engine. P: `{"kv_connector":"NixlConnector","kv_role":"kv_producer"}`. The
  `ucx_error_handling_mode` "none" item is optional; the doc says the default "peer" mode can exclude intra-node GPU
  transports (`D/nixl_connector_usage.md:66-76`). If you want the CI-tested form, use `kv_both` for both children.
- The offload connector has no `kv_role` check (grep of the pinned offload files), so `kv_consumer` on the parent is
  not needed.

## 3. One container, two GPUs; UCX; Modal versus Runpod

FACT:
- The doc has a "Basic Usage (on the same host)" section: two `vllm serve` processes with `UCX_NET_DEVICES=all` and
  distinct side-channel ports, plus `toy_proxy_server.py` (`D/nixl_connector_usage.md:79-126`).
- The pinned image `vllm/vllm-openai@sha256:f255c4fa...` (E-56/E-57 image) has `nixl==1.5.0`, `nixl-cu12==1.5.0`,
  `nixl-cu13==1.5.0`, `lmcache==0.5.6`, `cupy-cuda13x==14.2.0` (`experiments/gemma4_kv_engine/results/
  e57-a100-20261105/e57/env-20261105T051318Z/hardware/pip-freeze.txt`). The release pipeline builds the image with
  `INSTALL_KV_CONNECTORS=true` (`release-pipeline.yaml:202, 228, 253, 288`), which installs
  `requirements/kv_connectors.txt` (`nixl == 1.5.0`; `docker/Dockerfile:1220-1240`).
- No separate `ucx` package is in the pip freeze.
- The CI runs NIXL P/D on 2-GPU L4 pods (`disaggregated.yaml:221-250`).

INTERPRETATION / UNKNOWN:
- The NIXL PyPI wheel carries its own UCX libraries; nothing else needs installing. UNKNOWN until a probe lists the
  UCX transports (cuda_ipc, cuda_copy, tcp).
- Same-host without RDMA works (CI on L4 pods). With cuda_ipc, GPU-to-GPU goes over NVLink or PCIe; without it, UCX
  stages over host memory and TCP, which may be a few GB/s. A 32K-token stock 31B request is about 3.4 GB on the wire
  (10 global layers x 32K x 8 KB, plus the 1K-token window of 50 sliding layers). At 2 GB/s that is 1.7 s, small
  next to its life on D (about 100 s), but it adds to the time a request holds D blocks before it decodes.
- Modal: one container can hold several GPUs, and two processes in it share one namespace. Whether gVisor's GPU proxy
  supports CUDA IPC between processes is UNKNOWN; TCP over loopback should work. No RDMA. Runpod Pods are plain Docker
  containers, so cuda_ipc should work. Check `/dev/shm` size on Runpod: the offload tier is an mmap in `/dev/shm`
  (E-69 Section 2).
- In-process: two engines in one Python process is not a supported pattern (each engine wants device 0 of its own
  `CUDA_VISIBLE_DEVICES`). Use two processes.

## 4. Getting the compact patch into both engines

FACT:
- `vllm.general_plugins` "will be loaded in all processes (process0, engine core process and worker processes)"
  (`scratchpad/e70pin/src/vllm/plugins/__init__.py:16-18, 77-87`); call sites in the engine core
  (`R/engine_core.py:140-142`) and arg parsing (`R/arg_utils.py:930-932, 3262`). The plugin function must be
  re-entrant (`docs/design/plugin_system.md:60`). `VLLM_PLUGINS` filters by name.
- `compact_engine.install()` raises on a second call (`experiments/gemma4_kv_engine/compact_engine.py:91-92`).

INTERPRETATION:
- Simplest: no plugin. Run each engine in process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`, UniProc, as E-69) from its own
  driver; each driver calls `compact_engine.install(layout)` before it builds the engine. The driver of D also holds
  the E-69 hooks.
- If you need `vllm serve`: a tiny local package with an entry point in `vllm.general_plugins` whose function reads
  the layout from an env var and calls `install()` once (guard on `STATE["installed"]`). Install it with
  `pip install --no-deps -e` (no library change). A `.pth` file is a fallback.

## 5. Simplest valid design

FACT: `run_accuracy_test.sh` puts decoders on the prefillers' GPUs when GPUs are too few, and waits for P first
(`T/nixl_integration/run_accuracy_test.sh:222-236`; default `GPU_MEMORY_UTILIZATION` 0.2 at :82). CI comment: TP>1
configs share GPUs between P and D at 0.35 each (`disaggregated.yaml:214-219`).

INTERPRETATION:
- 12B on one 80 GB GPU: two copies of 24 GB weights plus two pools fit. Good for function (NIXL with compact, offload
  reloads, byte checks). Not good for throughput: P prefill steals SMs from D.
- 31B: about 58 GiB of weights per engine, so P and D need one 80 GB GPU each. D gets about 15 GiB of KV (E-69 used
  15 GiB), just below E-64's smallest pool (6,727 blocks, about 16.4 GiB). For E-64-sized pools, put D on an H200.
- The rehearsal should use two GPUs (2 x L40S 48 GB for the 12B), so it also tests the GPU-to-GPU transport.

## 6. Driving D3 and M2 and measuring D only

FACT (`gemma4_kv_capacity_model/e64.py:29-37`, `sweep.py:29-33`, `poolsim.py:382-400`):
- D3: prompt lognormal median 32,768, sigma 0.6, clipped [256, 131,072]; output median 2,048, sigma 0.6, [16, 16,384].
- M2: prompt median 49,152, sigma 0.6, [256, 131,072]; output median 500, sigma 0.8, [16, 4,096].
- Arrival: saturated. The waiting queue is refilled to 8 every step (`keep_waiting=8`), so the backlog never
  drains; a request whose stock full-ISL reservation does not fit the empty pool is skipped. Seeds 0 to 4, RNG
  `random.Random(f"64-{name}-{seed}")`. Window: steps 40,000 to 200,000.
- Offline params: `SamplingParams.extra_args["kv_transfer_params"]` is read into the request
  (`scratchpad/e70pin/src/request.py:119-121`). The toy proxy sends P `do_remote_decode: True`, `max_tokens: 1`,
  then sends D the `kv_transfer_params` from P's response (`toy_proxy_server.py:153-231`).

INTERPRETATION (design):
- Driver = proxy. P process: an in-process engine; it prefills with `max_tokens=1` and
  `{"do_remote_decode": True}`, and sends each finished request's `kv_transfer_params` to D over a pipe. D process:
  an in-process engine; it adds the request with `ignore_eos`, the drawn `max_tokens`, and the params. Keep 1 to 2
  requests waiting on D (FCFS admission makes only the head matter). P blocks stay held until D pulls (heartbeat
  lease, `N/base_scheduler.py:202-241`), so a large backlog fills P's pool.
- Measure on D only, over a fixed window after warm-up: generated tokens per second (from D's step outputs), local
  re-prefill tokens, preemptions (the `_preempt_request` counter), offload load bytes and re-admission hits (E-69
  hooks), NIXL bytes and seconds (`NixlKVConnectorStats`), KV pool use, and the share of steps with D's waiting queue
  non-empty. Validity: D saturated (waiting queue non-empty in at least 95% of window steps), pool-bound share at
  least 0.5 (E-64 F5), at least N natural preemptions per arm.
- The P bottleneck (rough, W-B prefill 1.0 s per 2,048 tokens on an A100): D3 needs about 0.6 to 1.0 of one P GPU
  (tight). M2 needs about 2.5 to 3 P GPUs. Options for M2: 2 to 3 P GPUs; or P with prefix caching on a small set of
  repeated prompts and a unique `cache_salt` per D request, so D's offload keys stay unique (needs a check that the
  first admission on D has zero offload hits).
- Arms (minimum): S-N and C-N (NIXL only: preempted requests prefill again on D) and S-NO and C-NO (NIXL plus
  offload). The ratio C-NO / S-NO is the E-64/E-67 claim.

## 7. Judgment: three likely failures, rehearsals, cost

1. NIXL rejects the compact page (near certain, FACT by reading), or a work-around transfers wrong bytes.
   Rehearsal: a CPU probe in the pinned image (cents). Build a real `NixlConnectorWorker` with the compact Gemma 4
   `KVCacheConfig` and the recording fake NIXL wrapper of `test_nixl_desc_geometry.py`; expect the assert; apply the
   2D-view patch; check the descriptor geometry. Then a GPU byte check (P hash equals D hash) on the 12B.
2. Transport or environment: NIXL/UCX in the container (cuda_ipc in gVisor, peer error mode, `/dev/shm`, side-channel
   ports). Rehearsal: a 2-GPU smoke test with Qwen3-0.6B and the CI script settings, 10 to 15 minutes, which logs
   the UCX transports and the GB/s.
3. The workload misses the E-64 regime: P-bound (M2), too few natural preemptions, or no offload hit after a NIXL
   arrival (for example if D's sliding blocks are not stored). Rehearsal: the 12B on 2 GPUs with scaled D3 and M2;
   VERIFY saturation share, natural preemption count, re-admission coverage of NIXL-arrived requests, P idle time.

Rough estimate (INTERPRETATION; read live prices before any plan): CPU probe under 0.1 hour. Transport smoke test
about 0.5 GPU-hours. 12B rehearsal on 2 GPUs: 1 to 1.5 hours, so 2 to 3 GPU-hours. 31B main on 2 x 80 GB: four arms x
two workloads x about 20 minutes plus four or more 31B starts, about 4 to 6 hours, so 8 to 12 GPU-hours; add about 50%
if M2 needs a second P GPU. Eager mode as E-69; CUDA graphs with the offload connector are untested.
