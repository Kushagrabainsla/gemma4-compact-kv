"""E-74 timing test (README Sections 9 and 12): one cell at the real sizes, on synthetic random tensors.

It reads no capture file, so it shows no real result. It prints the seconds of `layer_metrics` for an 8,192-token cell
and for a 32,768-token cell (256 query rows, 32 query heads, 4 KV heads, head size 512).

Run: uv run --no-project --with numpy --with torch python -B <this file>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_e74 as e  # noqa: E402

W = 0.0615


def cell(tokens: int, q_rows: int = 256) -> dict:
    gen = torch.Generator().manual_seed(tokens)
    v = torch.randn(tokens, 4, 512, generator=gen)
    v = v / v.pow(2).mean(-1, keepdim=True).sqrt()
    pos = torch.arange(tokens, dtype=torch.float64)[:, None].expand(tokens, 4)
    w = torch.full((512,), W)
    k = e.ca.rope((w * v).double(), pos).float()
    q = torch.randn(q_rows, 32, 512, generator=gen) * 4.0
    return {"q": q.to(torch.bfloat16), "k": k.to(torch.bfloat16), "v": v.to(torch.bfloat16),
            "qpos": torch.arange(tokens - q_rows, tokens)}


def main() -> None:
    sc = {"calstd_joint": {"ks": 0.0078125, "vs": 0.125}, "tied": {"ks": W * 0.125, "vs": 0.125}}
    w_mod = e.an.bf16(torch.full((512,), W))
    print("torch", torch.__version__, "threads", torch.get_num_threads())
    for tokens in (8192, 32768):
        cap = cell(tokens)
        t0 = time.time()
        m = e.layer_metrics(cap, w_mod, sc, "cpu")
        print(f"{tokens} tokens: {time.time() - t0:.1f} s for one cell; arms {len(m['err'])} + {len(m['errk'])}")


if __name__ == "__main__":
    main()
