"""The KV stitch is launch-bound below a few thousand tokens (~5 small kernels x 28 layers), so
capture it as one CUDA graph per prompt length and time the replay.
"""
import sys, json, warnings, argparse, torch
warnings.filterwarnings("ignore")
import common as C
from latency import load, timed
from latency_kv import load_bridge

DEV = "cuda:0"


class GraphedBridge:
    """Static-shape bridge: inputs copied into fixed buffers, graph replayed."""
    def __init__(self, br, m4, m17, T):
        self.br = br
        dev = next(br.parameters()).device          # follow the stitch, not a global
        Ls = 36; H, D = 8, 128
        self.kin = [torch.zeros(1, H, T, D, device=dev, dtype=torch.bfloat16) for _ in range(Ls)]
        self.vin = [torch.zeros(1, H, T, D, device=dev, dtype=torch.bfloat16) for _ in range(Ls)]
        self.cos4, self.sin4 = C.rope_cos_sin(m4, T, dev, dtype=torch.bfloat16)
        self.cos17, self.sin17 = C.rope_cos_sin(m17, T, dev, dtype=torch.bfloat16)
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): self._run()
        torch.cuda.current_stream().wait_stream(s)
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):
            self.out = self._run()

    def _run(self):
        kv = [(C.strip_rope(k, self.cos4, self.sin4), v) for k, v in zip(self.kin, self.vin)]
        outs = []
        for lt in range(len(self.br.src_map)):
            y = self.br(kv, lt); Dh = y.shape[-1] // 2
            K = C.apply_rope(y[..., :Dh].permute(1, 0, 2).unsqueeze(0), self.cos17, self.sin17)
            outs.append((K, y[..., Dh:].permute(1, 0, 2).unsqueeze(0)))
        return outs

    def __call__(self, layers):
        for (k, v), kb, vb in zip(layers, self.kin, self.vin):
            kb.copy_(k); vb.copy_(v)
        self.g.replay()
        return self.out


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); a = ap.parse_args()
    m4, m17 = load(C.PREFILL_MODEL), load(C.DECODE_MODEL)
    br = load_bridge(a.tag)
    from transformers import DynamicCache
    rows = []
    for T in (256, 1024, 4096):
        ids = torch.randint(0, 150000, (1, T), device=DEV)
        layers = C._cache_layers(m4(input_ids=ids, use_cache=True).past_key_values)
        gb = GraphedBridge(br, m4, m17, T)
        def full():
            outs = gb(layers); c = DynamicCache()
            for lt, (K, V) in enumerate(outs): c.update(K, V, lt, {})
            return c
        t_graph = timed(full, reps=9, warm=3)
        pre4 = timed(lambda: m4(input_ids=ids, use_cache=True))
        rows.append(dict(T=T, bridge_graph=t_graph, prefill_4b=pre4))
        print(f"  T={T:5d}  bridge (CUDA graph) {t_graph:6.1f} ms  = {100*t_graph/pre4:4.1f}% of 4B prefill ({pre4:.0f} ms)", flush=True)
        del gb; torch.cuda.empty_cache()
    json.dump(rows, open(f"{C.RESULTS}/latency_graph_{a.tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
