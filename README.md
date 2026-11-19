# Gemma 4 compact K=V cache

Code and experiment records of the paper:

> Kushagra Bainsla. *Storing the Gemma 4 K=V Cache Once: Exact Global-KV Memory Savings in a Paged Serving Engine
> with Unchanged Attention Kernels.* arXiv:2611.23035, 2026. https://arxiv.org/abs/2611.23035

## What the method does

In the global attention layers of Gemma 4, the key is the value times a fixed weight, with rotary position encoding
on one quarter of its dimensions. So one 640-value row per KV head and token holds everything that the stock cache
stores in 1,024 values. An unchanged paged attention kernel reads that row through two overlapping strided views:
values 0 to 511 as its key cache and values 128 to 639 as its value cache. The paper measures the result in a vLLM
prototype.

## Where to start

| File | What it is |
|---|---|
| `experiments/gemma4_kv_engine/compact_layout.py` | The 640-value row, its two views and the query fold |
| `experiments/gemma4_kv_engine/compact_engine.py` | The vLLM prototype: the cache, the write and the read hooks |
| `experiments/gemma4_kv_disagg/nixl_compact.py` | The patch that lets NIXL move the compact page between engines |
| `experiments/gemma4_kv_capacity_model/poolsim.py` | The model of the vLLM block pool and scheduler |
| `experiments/gemma4_kv_kernel/compact_attn.py` | Reference attention, RoPE and the views helpers |

## Experiments

Each folder holds one experiment (or a small group): a README with the plan, the frozen decision rule and the
result, the code, the tests and the raw results. The plan of each experiment was written before its run.

| Folder | Experiment | Use in the paper |
|---|---|---|
| `gemma4_kv_audit` | E-49 | Ceiling of the decode-step saving, 16-bit KV (context) |
| `gemma4_kv_headroom` | E-50 | The same on an FP8 KV cache (context) |
| `gemma4_kv_kernel` | E-51 | A compact FP8 Triton kernel (exploratory) |
| `gemma4_kv_views` | E-52 | The two-view read in Triton and stock FP8 kernels (exploratory) |
| `gemma4_kv_accuracy` | E-53 | The FP8 compact row at KV scale 1.0 |
| `gemma4_kv_scaletie` | E-54 | The FP8 compact row with tied, calibrated scales |
| `gemma4_kv_model` | (E-55) | Shared code only; E-55 is not used in the paper |
| `gemma4_kv_tasks` | E-56 | Accuracy of the emulated compact cache |
| `gemma4_kv_engine` | E-57 | The vLLM prototype: exactness, accuracy, capacity, step time (main result) |
| `gemma4_kv_microbench` | E-58 | Benchmark of the global layers without weights |
| `gemma4_kv_graphs` | E-59 | The prototype with CUDA graphs (exploratory) |
| `gemma4_kv_capacity_model` | E-63, E-64 | The block-pool model |
| `gemma4_kv_decode_scaling` | E-65 to E-68 | The model with CPU offload (E-67) and attention read times (E-68) |
| `gemma4_kv_offload` | E-69 | The CPU offload connector with the compact cache |
| `gemma4_kv_disagg` | E-70 | NIXL transfer of the compact cache |
| `gemma4_kv_decode_node` | E-71, E-72, E-73 | The decode node on two A100 GPUs (main result) |
| `gemma4_kv_rowscale` | E-74 | After the paper: one FP8 scale per compact row (attention level, CPU) |

Five more folders hold only small tool files that the code imports or runs: `fp8_quant_fusion` (`analyze_e44.py`),
`nvfp4_dispatch` (`analyze_e40.py`), `safedispatch_testA2` (`analyze_a2.py`, `gsm8k.py`), `ledger` (`ledger.py`,
`compare_runs.py`) and `runpod` (`hardware_record.sh`, `pod_cmd.py`, `gpu_microbench.py`). They come from earlier
experiments of the same project.

## Run the tests

The Mac tests need no GPU and no model weights. Run them from the root of this repository, for example:

```bash
uv run --no-project --with numpy --with torch --with pytest python -B -m pytest experiments/gemma4_kv_engine/test_compact_layout.py -q
```

Test status of this release (macOS, Python 3.12, torch 2.15): 303 of the 320 Mac tests pass. 17 fail because of two
choices of this release, not because of the method code:
- 11 tests of E-56, E-70 and E-72 check the SHA-256 of `experiments/gemma4_kv_tasks/data_e56.json`, which changed
  when the private prompts were removed (see below).
- 6 stub-session tests of E-49 to E-52 copy the full folders of earlier experiments of the project, which are not
  in this release.

The GPU runs used the image `vllm/vllm-openai@sha256:f255c4fad2f30ff5ee80db14d3a03afe9ae0f10d3470339864ff04745fe44cb3`
(vLLM 0.33.0rc1.dev229+g3966924d5), the checkpoint `google/gemma-4-31B-it` at revision
`842da3794eaa0b77d5f08bae87a17459d91ff475`, and Runpod or Modal GPUs. The launch scripts are in each folder; they
need your own GPU account.

## What is not in this release

- Model weights and the captured activations (several GB); the scripts that capture them are included.
- The data, results and notes of E-55, and the token ids of two private prompts in
  `experiments/gemma4_kv_tasks/data_e56.json` (prompts P1 and C1 of the "health" check). These prompts were the
  author's own unpublished text. Their SHA-256 values stay in the file, so the E-56 and E-57 health check cannot be
  repeated exactly; the public PG-19 prompt B1 stays.
- Account records: Pod files, boot logs, the cost ledger and the Pod registry. Pod ids and Modal app ids are replaced
  by `<pod-id>` and `<modal-app-id>`.
- Compiled kernel files (`.cubin`, `.sass`, `.ptx`, `.ttgir`) of E-52.
- The internal research journal and the review write-ups. The notes still name them (for example
  `dossier/REVIEW_E-57_POSTRUN_2026-11-05.md`, `RESEARCH_JOURNAL.md` or `PROTOCOL.md`), but those files are not
  public.
- Changed files: paths of the author's computer were made relative, so some recorded SHA-256 values of code files no
  longer match the files in this release.

## How to read the notes

| Label | Meaning |
|---|---|
| E-nn | An experiment |
| R-nn | A review of a design (before a run) or of a result (after a run) |
| D-nn | A decision of the author |
| I-182 | The research idea of this paper |
| O-nn, L-nn, U-nn | A finding, a lesson, an upstream proposal (internal register) |
| A1, A2 | An amendment of a frozen plan, with the reason and the time (before or after the run) |
| S0 to S9 | The stages of the experiment protocol (plan, review, local tests, approval, rehearsal, run, analysis, review) |
| V1, V2, ... | The validity checks of a run; a failed check makes the run INVALID |
| FACT, INTERPRETATION, HYPOTHESIS | A measured or read value; a reading of facts; a prediction that is not tested |

## License and citation

Apache License 2.0 (see `LICENSE` and `NOTICE`).

```bibtex
@misc{bainsla2026gemma4kv,
  title         = {Storing the Gemma 4 K=V Cache Once: Exact Global-KV Memory Savings in a Paged Serving Engine
                   with Unchanged Attention Kernels},
  author        = {Bainsla, Kushagra},
  year          = {2026},
  eprint        = {2611.23035},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```
