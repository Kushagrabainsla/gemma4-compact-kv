# E-70 upstream question: padded attention pages and the NIXL assert

Pin: 3966924d. Main: 10a854ed6 (2026-11-13). Method: code reading only. Nothing was run.

## Answer

Yes, at the pin, by code reading. The bug is fixed on main since 2026-11-13.

## Stock trigger at the pin

Config: `--kv-cache-dtype nvfp4 --kv-cache-dtype-skip-layers sliding_window` with NixlConnector. Use any model with sliding-window layers (for example Gemma 3 or gpt-oss) on FLASHINFER.

1. `platforms/interface.py:889-938` bumps the nvfp4 block to 64 tokens. It also sets `skip_page_size_padded` to the shared page.
2. `attention/attention.py:642-672` gives each sliding-window layer `page_size_padded=shared_page`.
3. `_largest_kernel_block_within` (`attention.py:101-130`) picks 16 tokens, because FLASHINFER blocks are [16, 32, 64].
4. Per KV head, the bf16 content is 16 x 4hs = 64hs bytes. The padded page is 64 x 9hs/8 = 72hs bytes. The gap is 11 %, for every head size.
5. `kv_cache_interface.py:395-410` sets the block stride to the padded page. `create_kv_cache_views` gives `cache[0].nbytes` = content, which is less than the stride.
6. `worker_utils.py:596` puts all layers in one buffer. With the NIXL default LBHNC layout, the storage is not block-major. `page_contiguous` is False.
7. `base_worker.py:1752-1762` then tests `cache.nbytes % (num_blocks * block_stride)`. The value is num_blocks x 64hs modulo num_blocks x 72hs, which is not zero, so the assert fires.

`fp8_per_token_head` (Triton, `triton_attn.py:308-319`) plus skip layers also pads: 8192 content in an 8448 page per head (hs=128). The plain fp8 example in the docs does not pad, because the bf16/fp8 ratio is exactly 2.

## Main: fixed

Commit e5dc65c1c (PR #65685, "Kimi K3.1 Pro split 5/6", 2026-11-13) changed the check (`base_worker.py:1806-1830` on main). It now rounds the view span (sum of (size-1) x stride) up to a multiple of `block_stride`. A padded page gives exactly num_blocks x stride, so one region registers with `block_len` = padded page. The comment names DSA latent and indexer rows. The fix is generic, so it also covers the E-70 compact spec. The padded-spec code is unchanged on main (`attention.py:704`, `interface.py:943`, `kv_cache_planning.py:385`).

## Candidates that do not pad an attention view

- Hybrid Mamba (Qwen3-Next, Nemotron-H, Granite 4, Jamba, Falcon-H1): `_align_hybrid_block_size` makes the attention block bigger and pads only the MambaSpec. NIXL handles Mamba in a separate branch.
- Sliding plus full attention (Gemma 3/4, gpt-oss): the head ratios are powers of two. `unify_kv_cache_spec_page_size` (`kv_cache_planning.py:383-397`) changes the block size and does not pad. Padding applies only to a page that does not divide the maximum, for example an odd-size draft.
- DeepSeek V3.2/V4 indexer, HiddenStateCacheSpec: these are MLA specs, so they are out of scope. They use the packed or MLA paths.

## Tests, matrix, issues

- NIXL tests pad only MambaSpec (`test_nixl_desc_geometry.py:604, 1559`; `test_nixl_connector_hma.py:2209`). No NIXL test has a padded AttentionSpec or skip layers.
- The compatibility matrix does not name skip layers or padded pages.
- #63997 (open): Gemma 4 + fp8 + NIXL hits the same message. The cause is different: one kernel-block ratio for groups with different block sizes. The view is dense. The main fix does not cover it. Fix PR #64067 is open.
- #66462 (closed, not merged): host-resident padded MLA pages (DSv4.1) hit the same message.
- #59586 (open): MLA kernel rows.
- No issue reports nvfp4 skip-layer padding with NIXL.

## What would settle it

Run the pin with NIXL and nvfp4 plus sliding-window skip layers on Blackwell. A CPU unit test is cheaper: build a padded SlidingWindowSpec in `test_nixl_desc_geometry.py` at the pin and at main.
