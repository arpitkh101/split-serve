"""Fair prefill-side timing, 2k to 32k tokens, batch 1, bf16: the 1.7B re-prefill on its best attention
path against the bridge alone (maps plus the 1.7B's own k/v projections, one CUDA graph), with the 4B
prefill for reference.  Re-prefill writes the KV cache and computes only the last position's logits,
which is what a server does; the older sdpa full-logits number is kept for comparison.

  CUDA_VISIBLE_DEVICES=4 python3 src/latency_fair.py --tag qamix_4b_c28
"""
import json, time, warnings, argparse, torch
warnings.filterwarnings("ignore")
import common as C
import residual_stitch as R
from latency import timed
from transformers import AutoModelForCausalLM, DynamicCache

DEV = "cuda:0"


def load(name, attn):
    return AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, attn_implementation=attn).to(DEV).eval()


def source_residuals(m4, ids, need):
    """4B residual stream entering each layer the maps read (index n_layers = the pre-norm final residual),
    collected with hooks so only those layers are kept; last-position logits only."""
    H, n = {}, len(m4.model.layers); hooks = []
    for j in need:
        if j < n: hooks.append(m4.model.layers[j].register_forward_pre_hook(lambda m, a, j=j: H.__setitem__(j, a[0].contiguous())))
        else: hooks.append(m4.model.norm.register_forward_pre_hook(lambda m, a, j=j: H.__setitem__(j, a[0].contiguous())))
    try: m4(input_ids=ids, use_cache=False, logits_to_keep=1)
    finally:
        for h in hooks: h.remove()
    return H


class GraphedBridge:
    """Maps plus native k/v projections for a fixed T as one CUDA graph; reads H in place (no clone) so 32k fits."""
    def __init__(self, fwd, m17, H, Lc, T):
        self.fwd, self.m17, self.H, self.Lc = fwd, m17, H, Lc
        pos = torch.arange(T, device=DEV).unsqueeze(0)
        self.pe = m17.model.rotary_emb(torch.zeros(1, T, 1, device=DEV, dtype=m17.dtype), pos)
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): self.out = self._run()
        torch.cuda.current_stream().wait_stream(s)
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g): self.out = self._run()

    def _run(self):
        ks, vs = [], []
        for l in range(self.Lc):
            k, v = R.native_kv(self.m17, self.fwd(self.H, l).unsqueeze(0), l, self.pe); ks.append(k); vs.append(v)
        return ks, vs, self.fwd(self.H, self.Lc).unsqueeze(0)

    def __call__(self):
        self.g.replay(); return self.out


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qamix_4b_c28"); ap.add_argument("--lengths", default="2048,4096,8192,16384,32768")
    ap.add_argument("--reps", type=int, default=10); ap.add_argument("--no-compile", action="store_true")
    a = ap.parse_args()
    try: import flash_attn; attn = "flash_attention_2"; print(f"flash_attn {flash_attn.__version__}", flush=True)
    except ImportError: attn = "sdpa"; print("flash_attn not importable, using sdpa", flush=True)
    m17 = load(C.DECODE_MODEL, attn); m17s = load(C.DECODE_MODEL, "sdpa"); m4 = load(C.PREFILL_MODEL, attn)
    R.DEV = DEV
    maps, Lc, _ = R.load_maps(a.tag); maps = maps.to(torch.bfloat16)
    def fwd(H, l):
        m = maps.maps[l]; return m(torch.cat([H[j][0] for j in m.src], -1))
    need = sorted({j for m in maps.maps for j in m.src})
    T = lambda fn: timed(fn, reps=a.reps, warm=3)

    comp = None
    if not a.no_compile:
        try:
            comp = torch.compile(m17, dynamic=True)
            ids = torch.randint(0, 150000, (1, 2048), device=DEV); t0 = time.time()
            comp(input_ids=ids, use_cache=True, logits_to_keep=1); torch.cuda.synchronize()
            print(f"torch.compile warm-up at 2048: {time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            print(f"torch.compile failed, skipped: {str(e)[:200]}", flush=True); comp = None

    def meas(r, key, fn):
        """One timing; an out-of-memory path is recorded as None instead of ending the run."""
        try: r[key] = T(fn)
        except torch.OutOfMemoryError: r[key] = None; print(f"  {key} at {r['T']}: out of memory, skipped", flush=True)
        torch.cuda.empty_cache()
    path = f"{C.RESULTS}/latency_fair.json"
    try: rows = json.load(open(path))
    except FileNotFoundError: rows = []
    for L in [int(x) for x in a.lengths.split(",")]:
        ids = torch.randint(0, 150000, (1, L), device=DEV); r = {"T": L, "attn": attn}
        if L > 16384 and m17s is not None: del m17s; m17s = None; torch.cuda.empty_cache()   # headroom for 32k
        if m17s is not None: meas(r, "prefill_17b_sdpa_full_logits", lambda: m17s(input_ids=ids, use_cache=True))
        else: r["prefill_17b_sdpa_full_logits"] = None
        meas(r, "prefill_17b_fair", lambda: m17(input_ids=ids, use_cache=True, logits_to_keep=1))
        if comp is not None:
            try: r["prefill_17b_fair_compiled"] = T(lambda: comp(input_ids=ids, use_cache=True, logits_to_keep=1))
            except Exception as e: print(f"  compiled path failed at {L}: {str(e)[:120]}", flush=True)
        meas(r, "prefill_4b", lambda: m4(input_ids=ids, use_cache=False, logits_to_keep=1))
        H = source_residuals(m4, ids, need)
        pos = torch.arange(L, device=DEV).unsqueeze(0)
        pe = m17.model.rotary_emb(torch.zeros(1, L, 1, device=DEV, dtype=m17.dtype), pos)
        def bridge_eager():
            cache = DynamicCache()
            for l in range(Lc):
                k, v = R.native_kv(m17, fwd(H, l).unsqueeze(0), l, pe); cache.update(k, v, l, {})
            return cache, fwd(H, Lc)
        r["bridge_eager"] = T(bridge_eager)
        gb = GraphedBridge(fwd, m17, H, Lc, L); r["bridge_graph"] = T(gb); del gb
        r["ratio_fair"] = r["prefill_17b_fair"] / r["bridge_graph"]
        rows = [x for x in rows if x["T"] != L] + [r]; rows.sort(key=lambda x: x["T"]); del H; torch.cuda.empty_cache()
        f = lambda v: f"{v:7.1f}" if v is not None else "   none"
        print(f"  T={L:6d}  1.7B re-prefill: sdpa+full logits {f(r['prefill_17b_sdpa_full_logits'])}  {attn}+last logit {f(r['prefill_17b_fair'])}"
              + (f"  compiled {r['prefill_17b_fair_compiled']:7.1f}" if "prefill_17b_fair_compiled" in r else "")
              + f"  | bridge eager {r['bridge_eager']:6.1f} graph {r['bridge_graph']:6.1f}  | 4B prefill {f(r['prefill_4b'])} ms"
              + f"  | fair ratio {r['ratio_fair']:.2f}x", flush=True)
        json.dump(rows, open(path, "w"), indent=2)


if __name__ == "__main__":
    main()
