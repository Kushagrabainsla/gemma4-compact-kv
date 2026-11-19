"""E-72 (README_E72.md): one engine role (P or D) of one arm. Arms: EV (stock cache with the VIEWS hooks, EMULV; probe
only) and C2 (compact cache with the VIEWS hooks; the E-71 workload, then the probe). Both roles set the NIXL
`kv_lease_duration` to LEASE_S.

C2 runs the E-71 workload code unchanged (`run_e71.run_d` and the `run_e71.run_p` loop), with `build` and
`engine_kwargs` replaced to add the lease and to keep the engine for the probe. After the window, D writes the window
record, then `probe_go`. P stops the workload, finishes its requests, and prefills the 8 probe prompts one at a time
(`q/x<k>.json`). D aborts its workload requests, waits until it has none, and decodes each probe alone (batch 1,
greedy, `max_tokens` 64, EOS allowed).

Usage: run_e72.py --role p|d --arm EV|C2 --model <dir> --dir <arm dir> --data <data_e56.json> [--stub]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path[:0] = [str(HERE), str(EXP / "gemma4_kv_disagg"), str(EXP / "gemma4_kv_offload"), str(EXP / "gemma4_kv_engine"),
                str(EXP / "gemma4_kv_model"), str(EXP / "gemma4_kv_capacity_model")]
import run_e71 as r71  # noqa: E402

ARMS = {"EV": "stock", "C2": "compact"}
LEASE_S = 600                                    # README_E72 Section 4 (R-120 F8: the default 30 s expired)
PROBES, PROBE_TOKENS = 8, 64
PLAN = {"EV": dict(r71.MAIN, warmup=0, window=0, watchdog=300),
        "C2": dict(r71.MAIN, watchdog=2_100)}
PROBE_WATCHDOG_S = 420
P_PARAMS = {"do_remote_decode": True, "do_remote_prefill": False, "remote_engine_id": None, "remote_block_ids": None,
            "remote_host": None, "remote_port": None}
STASH: dict = {}


def engine_kwargs(role: str, plan: dict) -> dict:
    """The E-71 settings plus the NIXL lease on P and on the NIXL child of D's MultiConnector."""
    kw = r71.engine_kwargs_e71(role, plan)
    cfg = kw["kv_transfer_config"]
    nixl = cfg if cfg["kv_connector"] == "NixlConnector" else cfg["kv_connector_extra_config"]["connectors"][0]
    nixl["kv_connector_extra_config"] = {**nixl["kv_connector_extra_config"], "kv_lease_duration": LEASE_S}
    return kw


def build(layout: str, role: str, model: str, plan: dict, rec: dict):
    """`run_e71.build` with the VIEWS hooks on both layouts (C2 as E-71 C; EV is EMULV) and the engine kept."""
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"
    os.environ["VLLM_KV_CACHE_LAYOUT"] = "LBHNC"
    os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] = "127.0.0.1"
    os.environ["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(r71.PORT[role])
    os.environ.setdefault("UCX_NET_DEVICES", "all")
    import torch
    import compact_engine as ce
    import nixl_compact as nx
    ce.install(layout)
    nx.install_capture_and_patch(layout == "compact")
    if role == "d":
        import run_e69
        from run_e70 import MultiStatsSum
        run_e69.HOOK["stats"] = MultiStatsSum(run_e69.HOOK["stats"])
        run_e69.HOOK["phase"] = "e71"
        run_e69.install_hooks(conn=True, corrupt=False)
    from vllm import LLM
    torch.backends.cuda.matmul.allow_tf32 = False
    rec["gpu"] = torch.cuda.get_device_name(0)
    rec["engine_kwargs"] = engine_kwargs(role, plan)
    t0 = time.time()
    try:
        llm = LLM(model=model, tokenizer=model, **rec["engine_kwargs"])
    finally:
        rec["spec"] = ce.STATE["spec"]
        rec["patched"] = len(nx.STATE["patched"])
    rec["engine_seconds"] = round(time.time() - t0, 1)
    rec["kv"] = ce.kv_record(llm)
    splice = ce.ViewsSplice()
    rec["hooks"] = llm.apply_model(lambda m: ce.register(m, splice))[0]
    splice.set_mode("VIEWS")
    rec["mode"] = splice.mode                         # A1 (R-121 F4): read back, not hard-coded
    STASH.update(llm=llm, ce=ce)
    return llm, ce


def install() -> None:
    """Route the E-71 workload code to the E-72 build and settings."""
    if not hasattr(r71, "engine_kwargs_e71"):
        r71.engine_kwargs_e71 = r71.engine_kwargs
    r71.engine_kwargs = engine_kwargs
    r71.build = build
    r71.ARMS.update(ARMS)


def load_probes(data: str) -> list[dict]:
    d = json.loads(Path(data).read_text())
    return [{"id": x["id"], "ids": x["ids"], "gold": str(x["gold"])} for x in d["needle"][:PROBES]]


def run_p(args, plan: dict, rec: dict) -> None:
    llm, _ = build(ARMS[args.arm], "p", args.model, plan, rec)
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    d = Path(args.dir)
    q = d / "q"
    q.mkdir(exist_ok=True)
    eng = llm.llm_engine
    sp = dict(max_tokens=1, temperature=0.0, seed=0, ignore_eos=True, detokenize=False,
              extra_args={"kv_transfer_params": dict(P_PARAMS)})
    reqs = r71.requests(args.n, plan["max_model_len"]) if plan["window"] else []
    nxt, produced = 0, 0
    t0 = time.time()
    while not (d / "probe_go").exists() and not (d / "d_done").exists():          # the E-71 workload loop
        consumed = int((d / "consumed.txt").read_text() or 0) if (d / "consumed.txt").exists() else 0
        if nxt - consumed < r71.BUF and nxt < len(reqs):
            eng.add_request(f"p{nxt}", TokensPrompt(prompt_token_ids=r71.prompt_ids(nxt, reqs[nxt][0])),
                            SamplingParams(**sp))
            nxt += 1
        for o in eng.step() if eng.has_unfinished_requests() else []:
            if o.finished:
                i = int(o.request_id.split("-")[0][1:])
                r71.write_json(q / f"{i}.json", {"i": i, "params": o.kv_transfer_params})
                produced += 1
        if not eng.has_unfinished_requests():
            eng.step()
            time.sleep(0.01)
    rec.update(produced=produced, workload_seconds=round(time.time() - t0, 1))
    while eng.has_unfinished_requests() and not (d / "d_done").exists():          # finish what P has
        eng.step()
    probes = load_probes(args.data)
    rec["probe_p"] = []
    for k, it in enumerate(probes):
        if (d / "d_done").exists():
            break
        eng.add_request(f"x{k}", TokensPrompt(prompt_token_ids=it["ids"]), SamplingParams(**sp))
        done = None
        while done is None and not (d / "d_done").exists():                       # A1 (R-121 F1)
            for o in eng.step():
                if o.finished and o.request_id.split("-")[0] == f"x{k}":
                    done = o
        if done is None:
            break
        r71.write_json(q / f"x{k}.json", {"k": k, "params": done.kv_transfer_params})
        rec["probe_p"].append({"k": k, "first_token": list(done.outputs[0].token_ids)})
    while not (d / "d_done").exists():                                             # serve D's reads
        eng.step()
        time.sleep(0.01)


def probe_d(args, rec: dict, n_added: int) -> None:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    d = Path(args.dir)
    q = d / "q"
    eng = STASH["llm"].llm_engine
    sch = STASH["ce"].engine_core(STASH["llm"]).scheduler
    (d / "probe_go").write_text(r71.now() + "\n")
    if n_added:
        eng.abort_request([f"d{i}" for i in range(n_added)])
    # A1 (R-121 F1): step until the scheduler holds no request, so the frees and the release notices reach P
    t_end = time.time() + 60
    while (eng.has_unfinished_requests() or sch.requests) and time.time() < t_end:
        eng.step()
        time.sleep(0.005)
    rec["probe_left_after_abort"] = len(sch.requests)
    if eng.has_unfinished_requests():
        raise RuntimeError("probe: workload requests did not leave D within 60 s")
    rec["probe"] = []
    for k, it in enumerate(load_probes(args.data)):
        f = q / f"x{k}.json"
        t_end = time.time() + 300
        while not f.exists() and time.time() < t_end:
            eng.step()                                 # A1 (R-121 F1): keep D's connector work going while waiting
            time.sleep(0.2)
        if not f.exists():
            raise RuntimeError(f"probe: P did not deliver x{k} within 300 s")
        params = json.loads(f.read_text())["params"]
        # A1 (R-121 B2): ignore_eos, so each probe compares 64 tokens; the sampled-token logprobs are recorded too
        sp = SamplingParams(max_tokens=PROBE_TOKENS, temperature=0.0, seed=0, detokenize=True, ignore_eos=True,
                            logprobs=1, extra_args={"kv_transfer_params": params})
        eng.add_request(f"y{k}", TokensPrompt(prompt_token_ids=it["ids"]), sp)
        done, t0 = None, time.time()
        while done is None:
            for o in eng.step():
                if o.finished and o.request_id.split("-")[0] == f"y{k}":
                    done = o
            if time.time() - t0 > 120:
                raise RuntimeError(f"probe: y{k} did not finish within 120 s")
        c = done.outputs[0]
        lps = [float(step[t].logprob) if step and t in step else None
               for step, t in zip(c.logprobs or [], c.token_ids)]
        rec["probe"].append({"k": k, "id": it["id"], "tokens": list(c.token_ids), "logprobs": lps,
                             "text": (c.text or "")[:300], "finish_reason": c.finish_reason,
                             "gold_in_text": it["gold"].lower() in (c.text or "").lower(),
                             "num_cached_tokens": getattr(done, "num_cached_tokens", None),
                             "seconds": round(time.time() - t0, 2)})


def run_d(args, plan: dict, rec: dict) -> None:
    d = Path(args.dir)
    if plan["window"]:
        r71.run_d(args, plan, rec)                     # the E-71 workload loop; it builds through `build`
        rec["status_window"] = "ok"
        r71.write_json(d / "d_run.json", rec)          # the window record survives a probe failure
    else:
        build(ARMS[args.arm], "d", args.model, plan, rec)
    timer = threading.Timer(PROBE_WATCHDOG_S, lambda: (rec.update(status="failed", error="probe watchdog"),
                                                       r71.write_json(d / "d_run.json", rec),
                                                       (d / "d_done").write_text(r71.now() + "\n"), os._exit(1)))
    timer.daemon = True
    timer.start()
    probe_d(args, rec, (rec.get("window") or {}).get("added", 0))
    timer.cancel()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=("p", "d"))
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=2_000)
    ap.add_argument("--stub", action="store_true")
    args = ap.parse_args()
    install()
    plan = PLAN[args.arm]
    d = Path(args.dir)
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{args.role}_run.json"
    rec: dict = {"role": args.role, "arm": args.arm, "layout_kind": ARMS[args.arm], "status": "started",
                 "started_at": r71.now(), "plan": plan, "lease_s": LEASE_S,
                 "manifest_ok": os.environ.get("E72_MANIFEST_OK") == "1"}
    r71.write_json(out, rec)
    try:
        if args.stub:
            import stub_e72
            stub_e72.run(args, plan, rec)
        elif args.role == "p":
            run_p(args, plan, rec)
        else:
            run_d(args, plan, rec)
        rec["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        rec["status"] = "failed"
        rec["error"] = f"{type(e).__name__}: {e}"[:3_000]
        rec["traceback"] = traceback.format_exc()[-8_000:]
    rec["finished_at"] = r71.now()
    r71.write_json(out, rec)
    if args.role == "d":
        (d / "d_done").write_text(r71.now() + "\n")
    print(f"E-72 {args.arm} {args.role}: {rec['status']} | {json.dumps(rec.get('window'))} | "
          f"probes {len(rec.get('probe') or rec.get('probe_p') or [])} | {rec.get('error', '')[:300]}", flush=True)
    sys.stdout.flush()
    os._exit(0 if rec["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
