"""Prefill-side latency of the residual-space pipeline, decomposed: 4B prefill, the bridge
(maps + the 1.7B's own k/v projections) eager and as one CUDA graph, and any native 1.7B tail.
"""
import sys, json, warnings, argparse, torch
warnings.filterwarnings("ignore")
import common as C
import residual_stitch as R
from residual_cut import run_last_layers
from latency import timed, load
from transformers import DynamicCache

DEV = "cuda:0"


class GraphedBridge:
    """Capture maps + native k/v projections for a fixed T as one CUDA graph."""
    def __init__(self, maps, m17, H, Lc, T):
        self.maps, self.m17, self.Lc, self.T = maps, m17, Lc, T
        self.H = [h.clone() for h in H]
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
            h = self.maps(self.H, l).unsqueeze(0).to(self.m17.dtype)
            k, v = R.native_kv(self.m17, h, l, self.pe); ks.append(k); vs.append(v)
        hc = self.maps(self.H, self.Lc).unsqueeze(0).to(self.m17.dtype)
        return ks, vs, hc

    def __call__(self, H):
        for dst, src in zip(self.H, H): dst.copy_(src)
        self.g.replay(); return self.out


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="reskv_4b"); a = ap.parse_args()
    m4, m17 = load(C.PREFILL_MODEL), load(C.DECODE_MODEL)
    maps, Lc, _ = R.load_maps(a.tag); maps = maps.to(torch.bfloat16)
    R.DEV = DEV
    # maps.forward casts inputs to float; in bf16 keep everything bf16
    def fwd(H, l):
        m = maps.maps[l]; x = torch.cat([H[j][0] for j in m.src], -1); return m(x)
    maps.forward = fwd
    rows = []
    for T in (256, 1024, 4096):
        ids = torch.randint(0, 150000, (1, T), device=DEV)
        pre4 = timed(lambda: m4(input_ids=ids, output_hidden_states=True, use_cache=False))
        pre17 = timed(lambda: m17(input_ids=ids, use_cache=True))
        H = m4(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
        pos = torch.arange(T, device=DEV).unsqueeze(0)
        pe = m17.model.rotary_emb(torch.zeros(1, T, 1, device=DEV, dtype=m17.dtype), pos)
        def bridge_eager():
            cache = DynamicCache()
            for l in range(Lc):
                k, v = R.native_kv(m17, fwd(H, l).unsqueeze(0), l, pe); cache.update(k, v, l, {})
            return cache, fwd(H, Lc).unsqueeze(0)
        t_eager = timed(bridge_eager)
        gb = GraphedBridge(maps, m17, H, Lc, T)
        t_graph = timed(lambda: gb(H))
        cache, hc = bridge_eager()
        def tail():
            c2 = DynamicCache()
            for l in range(Lc): c2.update(cache.key_cache[l], cache.value_cache[l], l, {})
            return run_last_layers(m17, hc, Lc, c2, T)
        t_tail = timed(tail)
        tot = pre4 + t_graph + t_tail
        rows.append(dict(T=T, prefill_4b=pre4, bridge_eager=t_eager, bridge_graph=t_graph, tail_17b=t_tail, total=tot, prefill_17b=pre17))
        print(f"  T={T:5d}  4B prefill {pre4:6.1f} + bridge eager {t_eager:5.1f} / graph {t_graph:5.1f} + 1.7B layers {Lc}-27 {t_tail:5.1f}"
              f" = {tot:6.1f} ms  | 1.7B full prefill {pre17:6.1f} ms ({tot/pre17:.2f}x)   bridge share of 4B prefill {t_graph/pre4*100:.0f}%", flush=True)
        del gb; torch.cuda.empty_cache()
    json.dump(rows, open(f"{C.RESULTS}/latency_{a.tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
