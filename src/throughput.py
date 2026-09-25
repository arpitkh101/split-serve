"""Throughput under load on two GPUs, three ways on the same hardware: two independent 1.7B
replicas; 1.7B prefill on GPU P shipping its cache to a 1.7B decoder on GPU D; 4B + stitch
prefill on P, 1.7B decode on D. Batches flow P -> D through a multiprocessing queue. CUDA IPC
was blocked on the test machine, so the cache is staged through host shared memory; both
disaggregated conditions pay that round trip and the replicated baseline does not.
"""
import sys, os, json, time, argparse, warnings, torch
import torch.multiprocessing as mp
warnings.filterwarnings("ignore")
import common as C
mp.set_sharing_strategy("file_system")


def load(name, dev):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa").to(dev).eval()


@torch.inference_mode()
def decode_steps(m, past, nxt, G):
    for _ in range(G):
        o = m(input_ids=nxt, past_key_values=past, use_cache=True)
        past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
    return past


# ---------------------------------------------------------------- replicated

def replica(dev, B, P, G, n_batches, ready, go, out_q):
    import common as C
    torch.cuda.set_device(dev); m = load(C.DECODE_MODEL, dev)
    ids = torch.randint(0, 150000, (B, P), device=dev)
    with torch.inference_mode():
        o = m(input_ids=ids, use_cache=True); decode_steps(m, o.past_key_values, o.logits[:, -1:].argmax(-1), 4)
        torch.cuda.synchronize(); ready.set(); go.wait(); t0 = time.perf_counter()
        for _ in range(n_batches):
            o = m(input_ids=ids, use_cache=True)
            decode_steps(m, o.past_key_values, o.logits[:, -1:].argmax(-1), G)
        torch.cuda.synchronize()
    out_q.put(time.perf_counter() - t0)


# --------------------------------------------------------------- disaggregated

def prefill_worker(dev, model, stitch, B, P, n_batches, q, ready, go):
    import common as C
    torch.cuda.set_device(dev)
    m = load(C.PREFILL_MODEL if model == "4b" else C.DECODE_MODEL, dev)
    gb = None
    if stitch:
        from latency_kv import load_bridge
        from latency_kv_graph import GraphedBridge
        import latency_kv; latency_kv.DEV = dev
        br = load_bridge(stitch).to(dev)
        m17 = load(C.DECODE_MODEL, dev)
        gb = GraphedBridge(br, m, m17, P)
        del m17
    ids = torch.randint(0, 150000, (B, P), device=dev)

    def one():
        o = m(input_ids=ids, use_cache=True)
        layers = C._cache_layers(o.past_key_values)
        if gb is not None:
            # graph was captured for batch 1; run per row (B small) and stack
            outs = [gb([(k[i:i+1], v[i:i+1]) for k, v in layers]) for i in range(B)]
            layers = [(torch.cat([outs[i][l][0] for i in range(B)]), torch.cat([outs[i][l][1] for i in range(B)]))
                      for l in range(len(outs[0]))]
        nxt = o.logits[:, -1:].argmax(-1)
        # stage through host shared memory (CUDA IPC is not permitted here)
        return [(k.cpu().share_memory_(), v.cpu().share_memory_()) for k, v in layers], nxt.cpu()

    with torch.inference_mode():
        one(); torch.cuda.synchronize(); ready.set(); go.wait()
        for _ in range(n_batches):
            layers, nxt = one(); torch.cuda.synchronize()
            q.put((layers, nxt))
    q.put(None)


def decode_worker(dev, B, P, G, q, ready, go, out_q):
    import common as C
    from transformers import DynamicCache
    torch.cuda.set_device(dev); m = load(C.DECODE_MODEL, dev)
    ids = torch.randint(0, 150000, (B, P), device=dev)
    with torch.inference_mode():
        o = m(input_ids=ids, use_cache=True); decode_steps(m, o.past_key_values, o.logits[:, -1:].argmax(-1), 4)
        torch.cuda.synchronize(); ready.set(); go.wait(); t0 = time.perf_counter(); wait_t = 0.0
        while True:
            tw = time.perf_counter(); item = q.get(); wait_t += time.perf_counter() - tw
            if item is None: break
            layers, nxt = item
            cache = DynamicCache()
            for l, (k, v) in enumerate(layers):
                cache.update(k.to(dev, non_blocking=True), v.to(dev, non_blocking=True), l, {})
            decode_steps(m, cache, nxt.to(dev), G)
            torch.cuda.synchronize()
            del layers, cache
        torch.cuda.synchronize()
    out_q.put((time.perf_counter() - t0, wait_t))


def run(cond, D, P_dev, B, P, G, N, stitch=None):
    ctx = mp.get_context("spawn")
    go = ctx.Event(); outq = ctx.Queue()
    if cond == "replicated":
        r1, r2 = ctx.Event(), ctx.Event()
        half = [N // 2, N - N // 2]
        ps = [ctx.Process(target=replica, args=(d, B, P, G, n, r, go, outq)) for d, n, r in ((D, half[0], r1), (P_dev, half[1], r2))]
        for p in ps: p.start()
        r1.wait(); r2.wait(); t0 = time.perf_counter(); go.set()
        ts = [outq.get(), outq.get()]; wall = max(ts); wait_t = None
        for p in ps: p.join()
    else:
        r1, r2 = ctx.Event(), ctx.Event(); q = ctx.Queue(maxsize=2)
        pw = ctx.Process(target=prefill_worker, args=(P_dev, "4b" if cond == "disagg-4b" else "17b", stitch, B, P, N, q, r1, go))
        dw = ctx.Process(target=decode_worker, args=(D, B, P, G, q, r2, go, outq))
        pw.start(); dw.start(); r1.wait(); r2.wait(); go.set()
        wall, wait_t = outq.get(); pw.join(); dw.join()
    toks = N * B * G
    r = dict(cond=cond, batches=N, batch=B, prompt=P, gen=G, wall_s=wall, gen_tok_s=toks / wall,
             req_s=N * B / wall, decode_idle_s=wait_t)
    extra = f"   decoder waited {wait_t:5.1f}s for prefill" if wait_t is not None else ""
    print(f"  {cond:12s} wall {wall:6.1f}s   {r['gen_tok_s']:7.1f} gen tok/s   {r['req_s']:5.2f} req/s{extra}", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--prompt", type=int, default=1024)
    ap.add_argument("--gen", type=int, default=64); ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--stitch", default="v2_k1_full_h0"); ap.add_argument("--gpus", default="0,1")
    a = ap.parse_args()
    D, Pd = [f"cuda:{g}" for g in a.gpus.split(",")]
    print(f"{a.batches} batches x {a.batch} req x ({a.prompt} prompt + {a.gen} gen)   two GPUs")
    res = [run("replicated", D, Pd, a.batch, a.prompt, a.gen, a.batches),
           run("disagg-17b", D, Pd, a.batch, a.prompt, a.gen, a.batches),
           run("disagg-4b", D, Pd, a.batch, a.prompt, a.gen, a.batches, a.stitch)]
    json.dump(res, open(f"{C.RESULTS}/throughput_b{a.batch}_p{a.prompt}_g{a.gen}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
