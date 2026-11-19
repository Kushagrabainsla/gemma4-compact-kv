import sys, json, time
from pathlib import Path
import torch

ROOT = Path("experiments")
sys.path.insert(0, str(ROOT / "gemma4_kv_rowscale"))
import analyze_e74 as e74  # noqa
an, ca = e74.an, e74.ca

CAP = ROOT / "gemma4_kv_scaletie/results/e54-h100-20261102/capture"
torch.manual_seed(0)
torch.set_num_threads(4)


def quant(x, g, top=0, perturb_top=None, rand_phase=False, gen=None):
    """per-row FP8 quantization (PTH rule) with factor g. top: keep the top-m |x| elements of each row exact.
    perturb_top: re-quantize the max element on the grid of factor perturb_top (the others stay at g)."""
    s = e74.row_scale(x, 1.0)
    if rand_phase:
        u = torch.rand(s.shape, generator=gen)
        s = (s * torch.pow(2.0, u)).float()
    else:
        s = (s * g).float()
    deq = e74.codes(x, s) * s
    if perturb_top is not None:
        idx = x.abs().argmax(-1, keepdim=True)
        s2 = (e74.row_scale(x, 1.0) * perturb_top).float()
        xm = x.gather(-1, idx)
        deq.scatter_(-1, idx, e74.codes(xm, s2) * s2)
    if top:
        idx = x.abs().topk(top, dim=-1).indices
        deq.scatter_(-1, idx, x.gather(-1, idx))
    return deq


def run(p, l):
    cap = torch.load(CAP / f"p{p}" / f"L{int(l):02d}.pt", map_location="cpu", weights_only=True)
    q, k, v = (cap[n].float() for n in ("q", "k", "v"))
    qpos = cap["qpos"].long()
    ref = an.attend(q, k, v, qpos)
    err = lambda kq, vq: an.rel(an.attend(q, kq, vq, qpos), ref)
    out = {}
    gen = torch.Generator().manual_seed(1)
    for j in (0, 3, 6):
        g = 2.0 ** (j / 8)
        out[f"j{j}"] = err(quant(k, g), quant(v, g))
        out[f"j{j}_top1exact"] = err(quant(k, g, top=1), quant(v, g, top=1))
        out[f"j{j}_top4exact"] = err(quant(k, g, top=4), quant(v, g, top=4))
        out[f"j{j}_bos_exact"] = None
    out["j0_maxperturb_j3grid"] = err(quant(k, 1.0, perturb_top=2 ** (3 / 8)), quant(v, 1.0, perturb_top=2 ** (3 / 8)))
    out["j0_maxperturb_j6grid"] = err(quant(k, 1.0, perturb_top=2 ** (6 / 8)), quant(v, 1.0, perturb_top=2 ** (6 / 8)))
    # K only and V only perturbation of the max at j=0 (j6 grid)
    out["j0_maxperturb_j6_Konly"] = err(quant(k, 1.0, perturb_top=2 ** (6 / 8)), quant(v, 1.0))
    out["j0_maxperturb_j6_Vonly"] = err(quant(k, 1.0), quant(v, 1.0, perturb_top=2 ** (6 / 8)))
    # random per-row phase (no exact max), and the same with the max exact
    for t in range(3):
        kk = quant(k, 1.0, rand_phase=True, gen=gen)
        vv = quant(v, 1.0, rand_phase=True, gen=gen)
        out[f"rand{t}"] = err(kk, vv)
        # same phases, max exact: regenerate with same seed
    gen = torch.Generator().manual_seed(1)
    for t in range(3):
        kk = quant(k, 1.0, rand_phase=True, gen=gen, top=1)
        vv = quant(v, 1.0, rand_phase=True, gen=gen, top=1)
        out[f"rand{t}_top1exact"] = err(kk, vv)
    # BOS row exact at j=6
    g = 2.0 ** (6 / 8)
    kq, vq = quant(k, g), quant(v, g)
    kq[0], vq[0] = k[0], v[0]
    out["j6_bos_exact"] = err(kq, vq)
    kq, vq = quant(k, g, top=1), quant(v, g, top=1)
    kq[0], vq[0] = k[0], v[0]
    out["j6_bos_exact_top1exact"] = err(kq, vq)
    # share of row energy in the max element (mean over rows)
    out["max_energy_share_k"] = float((k.pow(2).amax(-1) / k.pow(2).sum(-1)).mean())
    out["max_energy_share_v"] = float((v.pow(2).amax(-1) / v.pow(2).sum(-1)).mean())
    out["max_energy_share_k_bos"] = float((k[0].pow(2).amax(-1) / k[0].pow(2).sum(-1)).mean())
    out["max_energy_share_v_bos"] = float((v[0].pow(2).amax(-1) / v[0].pow(2).sum(-1)).mean())
    return {k_: v_ for k_, v_ in out.items() if v_ is not None}


res = {}
for p, l in [(0, 5), (0, 23), (4, 47), (1, 17), (5, 41)]:
    t0 = time.time()
    res[f"{p}/{l}"] = run(p, l)
    print(p, l, round(time.time() - t0, 1), flush=True)
    print(json.dumps({k_: round(v_, 4) for k_, v_ in res[f"{p}/{l}"].items()}), flush=True)
Path("/tmp/scratch").write_text(json.dumps(res, indent=1))
