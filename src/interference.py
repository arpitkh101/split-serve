"""Prefill-decode interference, measured directly with two processes and no shared state.
A decoder holds B in-flight requests on GPU D and logs the wall time of every decode step;
a prefiller runs T-token prefills on the same GPU or the other one, either as fast as it can
or at a fixed arrival rate (--rate). Conditions: idle, colocated 1.7B, colocated 4B,
disaggregated 4B, disaggregated 4B + stitch. Interference shows up as the decoder's
time-between-tokens distribution (p50/p99) moving.
"""
import sys, os, json, time, argparse, warnings, torch
import torch.multiprocessing as mp
warnings.filterwarnings("ignore")
import common as C


def load(name, dev):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa").to(dev).eval()


def decoder(dev, batch, prompt_len, steps, ready, go, out_q):
    import common as C
    torch.cuda.set_device(dev)
    m = load(C.DECODE_MODEL, dev)
    ids = torch.randint(0, 150000, (batch, prompt_len), device=dev)
    with torch.inference_mode():
        o = m(input_ids=ids, use_cache=True); past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
        for _ in range(8):                                     # warm
            o = m(input_ids=nxt, past_key_values=past, use_cache=True); past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
        ready.set(); go.wait()
        tbt = []
        for _ in range(steps):
            torch.cuda.synchronize(dev); t = time.perf_counter()
            o = m(input_ids=nxt, past_key_values=past, use_cache=True)
            past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize(dev); tbt.append(1000 * (time.perf_counter() - t))
    out_q.put(tbt)


def prefiller(dev, model, prompt_len, use_stitch, ready, go, stop, out_q, graph=False, rate=0.0):
    """rate > 0: issue prefills at that many per second (open loop) instead of back-to-back."""
    import common as C
    torch.cuda.set_device(dev)
    def pace(t_start, n):
        if rate > 0:
            slack = t_start + n / rate - time.perf_counter()
            if slack > 0: time.sleep(slack)
    if model.startswith("partial"):                    # "partial20": 1.7B layers 20..27 only
        import residual_cut as HS; HS.DEV = dev
        cut = int(model[len("partial"):]); m = load(C.DECODE_MODEL, dev)
        from transformers import DynamicCache
        h = torch.randn(1, prompt_len, m.config.hidden_size, device=dev, dtype=torch.bfloat16)
        n = 0
        with torch.inference_mode():
            for _ in range(2): HS.run_last_layers(m, h, cut, DynamicCache(), prompt_len)
            torch.cuda.synchronize(dev); ready.set(); go.wait(); t_start = time.perf_counter()
            while not stop.is_set():
                pace(t_start, n)
                HS.run_last_layers(m, h, cut, DynamicCache(), prompt_len); torch.cuda.synchronize(dev); n += 1
        out_q.put(n); return
    m = load(C.PREFILL_MODEL if model == "4b" else C.DECODE_MODEL, dev)
    br = None
    if use_stitch:
        from kv_stitch import BridgeV2
        ck = torch.load(f"{C.WORK}/{use_stitch}.pt")
        Lt = len(ck["src_map"]); k = ck["k"]
        W = torch.zeros(Lt, 8, k * 256, 256); b = torch.zeros(Lt, 8, 256)
        from kv_stitch import parse_hidden
        br = BridgeV2(ck["src_map"], W, b, ck["rank"], parse_hidden(ck["hidden"], Lt)).to(dev)
        br.load_state_dict(ck["state"]); br = br.to(torch.bfloat16).eval()
    ids = torch.randint(0, 150000, (1, prompt_len), device=dev)
    gb = None
    if br is not None and graph:
        from latency_kv_graph import GraphedBridge
        m17_rope = load(C.DECODE_MODEL, dev)              # only for its rotary tables
        gb = GraphedBridge(br, m, m17_rope, prompt_len)
    n = 0
    with torch.inference_mode():
        for _ in range(2): m(input_ids=ids, use_cache=True)
        ready.set(); go.wait(); t_start = time.perf_counter()
        while not stop.is_set():
            pace(t_start, n)
            if br is None:
                m(input_ids=ids, use_cache=True)
            elif gb is not None:
                gb(C._cache_layers(m(input_ids=ids, use_cache=True).past_key_values))
            else:
                kv = C.capture_kv(m, ids, strip=True)
                kv = [(a.bfloat16(), v.bfloat16()) for a, v in kv]
                cos, sin = C.rope_cos_sin(m, prompt_len, dev)
                for lt in range(len(br.src_map)):
                    y = br(kv, lt); D = y.shape[-1] // 2
                    C.apply_rope(y[..., :D].permute(1, 0, 2).unsqueeze(0), cos, sin)
            torch.cuda.synchronize(dev); n += 1
    out_q.put(n)


def run(cond, dec_dev, pre_dev, pre_model, use_stitch, batch, plen, steps, graph=False, rate=0.0):
    ctx = mp.get_context("spawn")
    r1, r2, go, stop = ctx.Event(), ctx.Event(), ctx.Event(), ctx.Event()
    q1, q2 = ctx.Queue(), ctx.Queue()
    pd = ctx.Process(target=decoder, args=(dec_dev, batch, plen, steps, r1, go, q1)); pd.start()
    pp = None
    if pre_model:
        pp = ctx.Process(target=prefiller, args=(pre_dev, pre_model, plen, use_stitch, r2, go, stop, q2, graph, rate)); pp.start()
    r1.wait(); (r2.wait() if pp else None)
    t0 = time.perf_counter(); go.set()
    tbt = q1.get(); dur = time.perf_counter() - t0
    stop.set(); n_pre = q2.get() if pp else 0
    pd.join(); (pp.join() if pp else None)
    tbt = sorted(tbt); p = lambda q: tbt[min(len(tbt) - 1, int(q * len(tbt)))]
    r = dict(cond=cond, batch=batch, prompt_len=plen, p50=p(0.5), p90=p(0.9), p99=p(0.99),
             max=tbt[-1], mean=sum(tbt) / len(tbt), prefills_per_s=n_pre / dur,
             decode_tok_s=batch * len(tbt) / (sum(tbt) / 1000))
    print(f"  {cond:14s} TBT p50 {r['p50']:6.1f}  p90 {r['p90']:6.1f}  p99 {r['p99']:6.1f}  "
          f"max {r['max']:7.1f} ms   decode {r['decode_tok_s']:6.1f} tok/s   "
          f"prefills {r['prefills_per_s']:5.2f}/s", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--stitch", default="", help="checkpoint tag in the work dir, e.g. v2_k8_full_h0")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--only-stitch", action="store_true", help="run just the stitch conditions (eager + graph)")
    ap.add_argument("--partial-cuts", default="", help="e.g. 20,14: also run colocated PARTIAL 1.7B prefills (layers >= cut)")
    ap.add_argument("--rate", type=float, default=0.0, help="prefill arrivals per second (0 = saturate)")
    a = ap.parse_args()
    D, P = [f"cuda:{g}" for g in a.gpus.split(",")]
    print(f"decoder: Qwen3-1.7B, batch {a.batch}, {a.prompt_len}-token contexts, {a.steps} steps on {D}")
    res = []
    R = a.rate
    if not a.only_stitch:
        res += [run("idle", D, None, None, None, a.batch, a.prompt_len, a.steps),
                run("colocated-17b", D, D, "17b", None, a.batch, a.prompt_len, a.steps, rate=R),
                run("colocated-4b", D, D, "4b", None, a.batch, a.prompt_len, a.steps, rate=R),
                run("disagg-4b", D, P, "4b", None, a.batch, a.prompt_len, a.steps, rate=R)]
    if a.stitch:
        res.append(run("disagg+stitch", D, P, "4b", a.stitch, a.batch, a.prompt_len, a.steps))
        res.append(run("disagg+graph", D, P, "4b", a.stitch, a.batch, a.prompt_len, a.steps, graph=True))
    for cut in [int(c) for c in a.partial_cuts.split(",") if c]:
        res.append(run(f"coloc-partial{cut}", D, D, f"partial{cut}", None, a.batch, a.prompt_len, a.steps, rate=R))
    suffix = ("_stitchonly" if a.only_stitch else "") + ("_partial" if a.partial_cuts else "") + (f"_rate{a.rate:g}" if a.rate else "")
    json.dump(res, open(f"{C.RESULTS}/interference_b{a.batch}_p{a.prompt_len}{suffix}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
