# E-69 feasibility: the vLLM CPU offload connector with the compact K=V cache (code reading only)

Date: 2026-11-12. Nothing ran on a GPU. No repo file changed.

## 0. Sources and their commit

- Pin: vLLM `3966924d566340e971a6da313e790d05492aa505` (the full hash is in the repo, for example in the E-57 and E-59
  result files).
- `P/` = `scratchpad/e67pin/` (the connector sources). FACT: `offloading/scheduler.py` and `offloading/worker.py` in
  `P/` are byte-identical (`cmp`) to the GitHub raw files at the pin, so `P/` is at the pin.
- `G/` = `scratchpad/pin3966/raw/`: files fetched from GitHub raw at the pin for this answer: `kv_cache_interface.py`,
  `worker/gpu/model_runner.py` (`model_runner_v2.py`), `worker/gpu/kv_connector.py` (`kv_connector_v2.py`),
  `worker/gpu/attn_utils.py`, `worker/utils.py` (`worker_utils.py`), `kv_cache_layout.py`,
  `attention/backends/utils.py` (`backends_utils.py`), `attention/backends/triton_attn.py`, `config/kv_transfer.py`,
  `config/cache.py`, `config/vllm.py` (`vllm_config.py`), `core/sched/scheduler.py`, `core/kv_cache_utils.py`,
  `core/single_type_kv_cache_manager.py`, `engine/core.py` (`engine_core.py`), `engine/arg_utils.py`,
  `entrypoints/llm.py`, `metrics/loggers.py`, `metrics/stats.py`, `device_communicators/shm_broadcast.py`.
- `scratchpad/vllm/` is commit `75be5c3919b6` (2026-11-10), NOT the pin. Its `kv_offload/` files differ from `P/`. I
  used it only to find file names.
- Repo: `experiments/gemma4_kv_engine/compact_engine.py` (CE), `experiments/gemma4_kv_engine/README.md` (ER),
  `experiments/gemma4_kv_graphs/README.md` (GR), `experiments/gemma4_kv_decode_scaling/README_E67.md` (E67).

## 1. Engine arguments

FACT. In process, pass a dict or a `KVTransferConfig`; `LLM` converts a dict (G/llm.py:235-252):

```python
llm = LLM(
    model="google/gemma-4-12B-it", ...,                     # the E-57/E-59 settings, Section 7
    kv_transfer_config={
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "cpu_bytes_to_use": N * (1 << 30),               # required, total over all workers
            "offload_prompt_only": False,                    # a Python bool, not the string "false"
        },
    },
    disable_log_stats=False,                                 # LLM sets True by default (llm.py:225-226)
    prefix_cache_retention_interval=None,                    # see Section 4; default 0
)
```

`KVTransferConfig` keys and defaults (FACT, G/kv_transfer.py:60-91, 110-124): `kv_connector` None; `engine_id` None,
set to a uuid4 in `__post_init__`; `kv_buffer_device` = the platform device type ("cuda"); `kv_role` None, required
when a connector is set; `kv_port` 14579; `kv_connector_extra_config` {}; `kv_connector_module_path` None;
`enable_permute_local_kv` False; `kv_load_failure_policy` "fail" ("recompute" is the other choice).

`kv_connector_extra_config` keys for the single CPU tier (FACT, P/docs/features/kv_offloading_usage.md:100-116 and
code):
- `spec_name` "CPUOffloadingSpec" (P/vllm/v1/kv_offload/factory.py:31).
- `cpu_bytes_to_use`: required, no default; an exception if missing (P/.../cpu/spec.py:100-104).
- `block_size`: no default (one GPU block per chunk). Do not set it for Gemma 4: it asserts unless all groups have one
  block size (P/.../offloading/config.py:137-158). `blocks_per_chunk` 1 (config.py:121-135). Not both (config.py:125).
- `eviction_policy` "lru" (spec.py:142); `cache_policy_module_path` None (spec.py:143).
- `store_threshold` 0 (spec.py:160); `max_tracker_size` 64000 (spec.py:163).
- `offload_prompt_only` True (P/.../kv_offload/base.py:754-755). It goes through `bool()`, so the string "false" gives
  True.
- `sync_load_budget_bytes` 0 (P/.../offloading/scheduler.py:323-325). 0 means every load is async.
- `self_describing_kv_events` False (base.py:746); `spec_module_path` None (factory.py:35); `canonical_layout` False
  (config.py:211).
- Multi-tier only: `secondary_tiers` [], `tier_poll_interval_s` 0.001.

Alternative (FACT, G/vllm_config.py:1166-1200): `kv_offloading_size=N` (GiB) with `kv_offloading_backend="native"`
sets the connector, `kv_role` "kv_both" and `cpu_bytes_to_use`, but not `offload_prompt_only`. If the env
`VLLM_USE_SIMPLE_KV_OFFLOAD` is set, this path selects `SimpleCPUOffloadConnector`. Use the explicit config.

## 2. The V2 model runner, the worker registration and the page bytes

- FACT. The V2 runner supports connectors. After `init_kv_cache`, it calls `get_kv_connector(vllm_config,
  kv_caches_dict)` (G/model_runner_v2.py:818-841). `ActiveKVConnector.__init__` calls
  `register_kv_caches(kv_caches_dict)` (G/kv_connector_v2.py:62-70). Loads start in `pre_forward`/`post_forward`,
  outside the model forward (kv_connector_v2.py:75-160). The V2 "unsupported" list has no connector item
  (G/vllm_config.py:3071-3105).
- FACT. `kv_caches_dict` holds the manager views, not the kernel-block views: "The manager views in `kv_caches` stay
  for KV connectors" (G/worker_utils.py:460-473). Each view is a 4D `[B, H, N, C]` view of one backing int8 buffer;
  cache groups overlay the same bytes (worker_utils.py:564-627; G/kv_cache_interface.py:418-482).
- FACT. `register_kv_caches` (P/.../offloading/worker.py:71-217): per layer, `group_kernel_blocks(cache, num_blocks)`
  (kv_cache_interface.py:348-357); `page = spec.page_size_bytes` (the PADDED page); block stride =
  `stride(0) * elem` (worker.py:100-119). Layers with the same address and strides (the group overlay) share one
  transfer tensor (worker.py:166-203). Each layer ref copies `unpadded_page_size_bytes` (worker.py:122-125, 197-202;
  copy size per op, P/.../cpu/gpu_worker.py:444-445).
- FACT. `unpadded_page_size_bytes = num_heads * num_states * state_content_size_bytes`, and
  `state_content_size_bytes = state_content_bytes` when set (kv_cache_interface.py:585-601). The padding is at the
  END of each block: the block stride is raised to the padded page (kv_cache_interface.py:394-415).
- FACT (arithmetic, ER:37-40, 106). 31B compact global: 4 x 48 x 1,280 = 245,760 B copied, block stride 262,144 B.
  31B stock global (block 32) and sliding (block 16): 262,144 B, no padding. Per block row: 15 GiB / 6,144 = 2,621,440
  B = 10 x 262,144 = `worker_kv_bytes_per_block` (P/.../offloading/config.py:175-185).
- FACT. The CPU slot of each transfer tensor is the padded page times `blocks_per_chunk` (gpu_worker.py:830-843).
  Block ids travel per group with `group_sizes` and `block_indices` (gpu_worker.py:596-666).
- FACT. `derive_canonical_mappings` runs even with `canonical_layout` off (worker.py:77). For a compact layer the view
  shape `(nb, 4, 48, 640)` is not a K/V shape, so `_attention_byte_regions` returns None and the opaque fallback with
  the unpadded page is used (P/.../canonical_mapping.py:250-265, 334-345, 443-459). The assert at worker.py:192-196
  then holds.
- INTERPRETATION. Block 48 or 32, `state_content_bytes` 1,280 and the padded page do not break the copy. The copy is
  the full unpadded page of each block, in each group, by block id.

## 3. Hybrid models (full plus sliding window groups)

- FACT. `OffloadingConnector(KVConnectorBase_V1, SupportsHMA)` (P/.../offloading_connector.py:53). So the hybrid
  manager stays on (G/vllm_config.py:2275-2298). Do not set `disable_hybrid_kv_cache_manager=True`: it turns the
  sliding layers into full attention (kv_cache_interface.py:619-624), which removes the page plan of the compact build.
- FACT. Asserts that apply: each group block size must divide by the hash block size (config.py:113-119). With a
  connector, the hash block is the GCD of the group block sizes (G/kv_cache_utils.py:860-889): GCD(16, 48) = 16, so 48
  passes; 32 passes. The scheduler block is LCM = 48 (compact) or 32 (stock). Do not set `block_size` in the extra
  config (Section 1).
- FACT. A sliding group needs a run of `cdiv(window, 16)` chunks at the hit point (config.py:61-84;
  scheduler.py:1000-1060). The hit is the minimum over all groups.
- FACT. I found no assert that rejects Gemma 4. `_max_loadable_tokens` asserts `HybridKVCacheCoordinator` only when a
  prefix-cacheable group is not offloaded (offloading_connector.py:164-181); here all groups are offloaded.

## 4. Preemption, lookup, and the counters

- FACT. `_preempt_request` frees the blocks, sets `num_computed_tokens = 0` and prepends the request to the waiting
  queue (G/scheduler.py:1799-1843). On the next admission `num_computed_tokens == 0`, so the scheduler does the local
  prefix lookup and then `connector.get_num_new_matched_tokens` (scheduler.py:1110-1142). Pending stores of the
  preempted request are flushed before its blocks are reused (P/.../scheduler.py:2180-2195; worker.py:219-244).
- FACT. A hit loads async by default: the request waits in `WAITING_FOR_REMOTE_KVS` with its blocks allocated
  (scheduler.py:1514-1523). `sync_load_budget_bytes > 0` loads inside the step.
- FACT, important. `prefix_cache_retention_interval` defaults to 0 (G/cache.py:275; arg_utils.py:602). With a value
  that is not None, the store path marks a sliding block reachable only near `num_prompt_tokens - 1` (and a shared
  prefix junction) (P/.../scheduler.py:1897-1935: "An active decode frontier is not a final request boundary"), and the
  sliding manager skips the dense branch when the interval is 0 (G/single_type_kv_cache_manager.py:1104-1146). With
  None, the segment is the alignment (48 or 32 tokens), `need` (64 blocks) >= the segment (3 or 2 blocks), so the
  mask is None and every sliding block is stored (single_type...:1110-1119).
- INTERPRETATION. At the default 0, decode-phase sliding chunks are not stored. A preempted decoding request then hits
  only up to the aligned prompt end; its generated tokens are recomputed, even with `offload_prompt_only=false`. E67
  line 18 states "returns dense"; that is true only for `prefix_cache_retention_interval=None`. Set None to test the
  E-67 `all` mode. Then each sliding group stores a 2.62 MB chunk row per 16 tokens (about 0.82 MB per token for 5
  groups on the 31B), so the CPU tier is mostly sliding KV.
- FACT, counters:
  - Hits: `SchedulerStats.connector_prefix_cache_stats` (a `PrefixCacheStats`). New requests: `requests`, `queries`,
    `hits` (tokens). Re-admitted preempted requests: `preempted_requests`, `preempted_queries`, `preempted_hits`
    (G/stats.py:121-155; scheduler.py:1487-1506). Prometheus `vllm:external_prefix_cache_queries/hits` count only the
    non-preempted part (G/loggers.py:1179-1186); `preempted_hits` is exported nowhere.
  - Loads and stores: `SchedulerStats.kv_connector_stats` (a dict) with `vllm:kv_offload_load_bytes`,
    `vllm:kv_offload_load_time` (s), `vllm:kv_offload_load_size` (histogram, one sample per job),
    `vllm:kv_offload_store_bytes`, `..._store_time`, `..._store_size`, `..._lookup_sync_delay_seconds`,
    `..._lookup_async_delay_seconds`, `vllm:kv_offload_allocation_failure` (P/.../offloading/metrics.py:39-56, 80-146;
    scheduler.py:2278-2306). CPU tier gauges (P/.../cpu/spec.py:55-84) and the info gauge `vllm:kv_offload_config_info`
    with `cpu_num_chunks` (docs:155-169).
  - Preemptions: CE `STATE["preemptions"]`; `vllm:num_preemptions` (loggers.py:738-744).
- FACT, in-process reading: `make_stats` returns None unless `log_stats` (scheduler.py:3265-3266), so set
  `disable_log_stats=False`. Worker transfer stats are merged into the scheduler stats in `update_from_output`
  (scheduler.py:2590-2603). `llm.get_metrics()` returns the Prometheus snapshot (llm.py:890-901). The offload scheduler
  `get_stats()` resets its payload on each call (P/.../scheduler.py:2356-2378), so a probe must not call it directly.
- INTERPRETATION. Read the stats with a CE-style wrapper of `Scheduler.make_stats` that sums
  `connector_prefix_cache_stats.preempted_*` and the `kv_connector_stats` dict, and returns the stats unchanged.

## 5. Constraints

- CUDA graphs. FACT: the connector has no per-layer hooks (`wait_for_layer_load`, `save_kv_layer` are no-ops,
  offloading_connector.py:122-132); transfers start outside the forward (kv_connector_v2.py:75-160). INTERPRETATION:
  FULL_DECODE_ONLY (E-59) should work. Untested.
- Async scheduling. FACT: connectors do not turn it off (G/vllm_config.py:1701-1735).
- Prefix caching. FACT: with a connector, block hashes exist even with prefix caching off (G/engine_core.py:226-233).
  INTERPRETATION: keep it off as in E-57/E-59; then every reuse after preemption comes from the CPU tier.
- KV layout. FACT: the connector asks for "LBHNC" (offloading_connector.py:235-239). The engine core uses it unless
  `VLLM_KV_CACHE_LAYOUT` is set (G/backends_utils.py:378-399); Triton declares no layouts, so the default preference
  starts with LBNHC (backends_utils.py:229-235). E-57 and E-59 logged "Using LBNHC KV cache layout" (their results).
  So the connector run silently changes the layout of the compact views.
- Host memory. FACT: the CPU tier is a shared mmap in `/dev/shm/vllm_offload_<engine_id>.mmap`, in up to 32 files
  (P/.../cpu/shared_offload_region.py:128-139; spec.py:25-27, 183-204). Startup raises if `/dev/shm` has less free space
  than the tier ("Increase /dev/shm (e.g. --shm-size or --ipc=host)", G/shm_broadcast.py:242-269). The tier is
  pre-faulted with MADV_POPULATE_WRITE, with a slow per-page fallback on kernels before 5.14
  (shared_offload_region.py:25-84). It is then registered with `cudaHostRegister`; a failure only logs a warning and
  leaves pageable copies (P/.../cpu/host_register.py:54-67; gpu_worker.py:203-268).
- INTERPRETATION. The Docker default `/dev/shm` is 64 MB, so a plain container fails at startup. Shared memory also
  counts against the container memory limit, next to the weights in the page cache. RLIMIT_MEMLOCK does not stop the
  run; it can only make copies pageable (slower). I could not verify the `/dev/shm` size of Runpod or Modal from code.

## 6. compact_engine.py patches and the overlap with the connector

- FACT (CE:93-155): it replaces `triton_attn.unified_attention` and `triton_attn.triton_reshape_and_cache_flash`
  (module attributes), `KVCacheManager.allocate_slots`, `Scheduler._preempt_request`, `Scheduler.schedule` (counters,
  pass-through), and, for "compact", the V2 `GPUModelRunner.get_kv_cache_spec`. It adds forward pre and post hooks on
  the global `Attention` modules (CE:245-253). E-59 also wrapped `TritonAttentionMetadataBuilder.__init__` and
  `GPUModelRunner.load_model` (GR:91-97).
- FACT. The spec patch feeds `kv_cache_config`, which the connector reads on both sides (offloading_connector.py:79-100).
  This is a shared input by design. The connector does not use the attention write op or the Triton views; it copies
  raw bytes per block (gpu_worker.py:407-447, 563-700). CE does not patch the allocation.
- FACT. The K/V split exists only inside the Triton backend: CE:75-83 fixes a 128-value "value cache" there. The
  connector reads the manager view `[B, H, N, 1280 B]` and copies `unpadded_page_size_bytes` = all 640 values of each
  head and token (Section 2).
- INTERPRETATION. The risk that the connector copies only the K view or the tail is low. The real overlaps are: the
  layout flip to LBHNC (Section 5), and the wrapped `allocate_slots`, which now also counts admissions with external
  tokens (its `alloc_fail` counter changes meaning).

## 7. E-57/E-59 settings and the 12B

- FACT (GR:68-90; ER:95-106). Image `vllm/vllm-openai@sha256:f255c4fad2f30ff5ee80db14d3a03afe9ae0f10d3470339864ff04745fe44cb3`
  (vLLM 0.33.0rc1.dev229+g3966924d5, torch 2.14.0+cu130, Triton 3.8.0). Not a pip setup.
- Model `google/gemma-4-31B-it` rev `842da3794eaa0b77d5f08bae87a17459d91ff475` (62.6 GB); rehearsal
  `google/gemma-4-12B-it` rev `43c9d014b231474fa7212dbb48a762566c2b255d` (23.95 GB).
- TRITON_ATTN, V2 runner, in process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`), BF16 KV, no prefix caching, seed 0,
  `max_model_len` 66,048, `max_num_batched_tokens` 2,048, `max_num_seqs` 16, `kv_cache_memory_bytes` 15 GiB, which
  overrides `gpu_memory_utilization` (G/cache.py:374-381). The READMEs do not record a `gpu_memory_utilization`.
- Pool: 6,144 blocks of 2,621,440 B (ER:106). Steady footprint per request, stock/compact: 1,349/1,008 blocks at 32K,
  2,373/1,690 at 64K (ER:335). Batches: 3 stock, 4 compact at 32K (GR:14, 427).
- E-57 eager (`enforce_eager=True`). E-59 graphs: `enforce_eager=False`, `CompilationConfig(mode=NONE,
  cudagraph_mode=FULL_DECODE_ONLY, cudagraph_capture_sizes=[1,2,3,4,8])` and the 512-wide scratch wrapper (GR:86-96).
- E-59 rehearsal: 12B on one A40, 16,384 tokens, 12 GiB pool, `max_model_len` 16,896 (GR:71, 124).
- 12B validity. FACT: 8 global layers with K=V, 1 global KV head of 512, 8 sliding KV heads of 256, window 1,024
  (ER:101-103); compact block 96, page padded to 131,072 B (ER:116, 441); EQ EXACT in the E-57 rehearsal (ER:5).
  INTERPRETATION: valid for the compact path and the connector logic (the same classes, 1.5 times the stock block,
  a padded page). It does not test the 31B numbers (4 KV heads, block 48, 10 global layers; GR:256). It needs a 48 GB
  GPU: 23.95 GB of BF16 weights do not fit a 24 GB card.

## 8. Judgment: three likely failures before a measurement, and a 20-minute 12B rehearsal

1. The engine does not start, or starts slow, in the container (`/dev/shm` too small, the memory limit, a failed
   `cudaHostRegister`). Rehearsal: print `df -h /dev/shm`, the cgroup `memory.max` and the host RAM; start the engine
   with the planned tier ratio; grep the log for "Created mmap file", "Registering full CPU offload region",
   "cudaHostRegister failed" and "stays pageable"; compute load GB/s from `load_bytes / load_time`.
2. The connector changes what the compact build reads: the layout flips from LBNHC to LBHNC, or a reloaded block is
   wrong. Rehearsal: check the "Using ... KV cache layout" line; set `VLLM_KV_CACHE_LAYOUT` to the same value in every
   arm; force preemption (a small `kv_cache_memory_bytes` or `num_gpu_blocks_override`) and compare greedy tokens of
   COMPACT+connector with EMULV+connector and with COMPACT without preemption; require CE `fix_reads` > 0.
3. The run has no hits to measure: no preemption, decode sliding chunks not stored (retention 0), or the hits sit in
   `preempted_hits`, which no log or Prometheus line shows. Rehearsal: wrap `make_stats`; require preemptions > 0,
   `preempted_hits` > 0, `load_bytes` > 0; run retention 0 against None and check that hits pass the prompt length
   only with None. Run eager first, then FULL_DECODE_ONLY graphs, for one short row each.
