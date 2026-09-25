"""Which 4B-sourced component loses task accuracy in the residual-cut pipeline: the early-layer
KV stitch or the residual map at the cut? Swap each for the 1.7B's own at eval time, one at a
time. The own-KV variants are diagnostics only (they cost a full 1.7B prefill).
"""
import sys, json, argparse, torch
import common as C
from transformers import DynamicCache
from residual_cut import load_sdpa, HiddenStitch, run_last_layers, DEV, OUT
from kv_stitch import BridgeV2, parse_hidden
from task_eval import Cond, load_task, run_task


def load_maps(tag):
    ci = torch.load(f"{OUT}/hidden_{tag}.pt")
    ck = torch.load(f"{OUT}/v2_k1_full_h0{'_self' if 'self' in tag else ''}.pt"); Lt = len(ck["src_map"])
    kvbr = BridgeV2(ck["src_map"], torch.zeros(Lt, 8, 256, 256), torch.zeros(Lt, 8, 256), 0, parse_hidden("0", Lt))
    kvbr.load_state_dict(ci["kv"]); hbr = HiddenStitch(ci["h"]["W"], ci["h"]["b"])
    return kvbr.to(DEV).float(), hbr.to(DEV).float(), ci["cut"], ci["src_layer"]


class Hybrid(Cond):
    def __init__(self, name, m4, m17, tok, kv_src, h_src):
        super().__init__(name, m4, m17, tok)
        self.kv_src, self.h_src = kv_src, h_src
        self.kv4, self.h4, self.Lc, self.js = load_maps("cut20_4b")
        self.kv17, self.h17, _, _ = load_maps("cut20_self")

    @torch.no_grad()
    def prefill(self, ids):
        T = ids.shape[1]; Lc = self.Lc
        cos17, sin17 = C.rope_cos_sin(self.m17, T, DEV); cache = DynamicCache()
        o4 = self.m4(input_ids=ids, output_hidden_states=True, use_cache=True)
        o17 = self.m17(input_ids=ids, output_hidden_states=True, use_cache=True)
        if self.kv_src == "4b":
            cos4, sin4 = C.rope_cos_sin(self.m4, T, DEV)
            kv = [(C.strip_rope(K.float(), cos4, sin4), V.float()) for K, V in C._cache_layers(o4.past_key_values)]; br = self.kv4
        else:
            kv = [(C.strip_rope(K.float(), cos17, sin17), V.float()) for K, V in C._cache_layers(o17.past_key_values)]; br = self.kv17
        for lt in range(Lc):
            y = br(kv, lt); D = y.shape[-1] // 2
            K = C.apply_rope(y[..., :D].permute(1, 0, 2).unsqueeze(0), cos17, sin17)
            cache.update(K.to(self.m17.dtype), y[..., D:].permute(1, 0, 2).unsqueeze(0).to(self.m17.dtype), lt, {})
        if self.h_src == "4b": h = self.h4(o4.hidden_states[self.js + 1][0].float())
        else: h = self.h17(o17.hidden_states[Lc][0].float())
        h = h.unsqueeze(0).to(self.m17.dtype)
        return cache, run_last_layers(self.m17, h, Lc, cache, T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="arc_challenge,lambada"); ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    m4, tok = load_sdpa(C.PREFILL_MODEL); m17, _ = load_sdpa(C.DECODE_MODEL)
    conds = [Hybrid("both4b", m4, m17, tok, "4b", "4b"), Hybrid("kv4b", m4, m17, tok, "4b", "own"),
             Hybrid("h4b", m4, m17, tok, "own", "4b"), Hybrid("own", m4, m17, tok, "own", "own")]
    res = {}
    for t in a.tasks.split(","):
        kind, shots, exs = load_task(t, a.n, 0); res[t] = {}
        print(f"[{t}] {len(exs)} examples", flush=True)
        for c in conds:
            res[t][c.name] = run_task(c, tok, kind, shots, exs)
            print(f"  {c.name:8s} " + "  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                                for k, v in res[t][c.name].items() if k != "per"), flush=True)
        json.dump(res, open(f"{C.RESULTS}/task_diag.json", "w"), indent=1)


if __name__ == "__main__":
    main()
