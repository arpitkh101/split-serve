"""Inference-time cost of a trained KV stitch (RoPE strip, 28 layers of maps, RoPE re-apply,
cache assembly), eager vs torch.compile, next to the 4B prefill it follows.
"""
import sys, json, warnings, argparse, torch
warnings.filterwarnings("ignore")
import common as C
from kv_stitch import BridgeV2, parse_hidden
from latency import load, timed
from transformers import DynamicCache

DEV = "cuda:0"


def load_bridge(tag):
    ck = torch.load(f"{C.WORK}/{tag}.pt")
    Lt, k = len(ck["src_map"]), ck["k"]
    W = torch.zeros(Lt, 8, k * 256, 256); b = torch.zeros(Lt, 8, 256)
    br = BridgeV2(ck["src_map"], W, b, ck["rank"], parse_hidden(ck["hidden"], Lt),
                  ck.get("share_heads", False))
    br.load_state_dict(ck["state"])
    return br.to(DEV).to(torch.bfloat16).eval()


def bridge_path(br, m4, m17, ids):
    """Everything between '4B prefill finished' and '1.7B can decode'."""
    out = m4(input_ids=ids, use_cache=True)
    T = ids.shape[1]
    cos4, sin4 = C.rope_cos_sin(m4, T, DEV, dtype=torch.bfloat16)
    cos17, sin17 = C.rope_cos_sin(m17, T, DEV, dtype=torch.bfloat16)
    layers = C._cache_layers(out.past_key_values)
    torch.cuda.synchronize(); s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
    kv = [(C.strip_rope(K, cos4, sin4), V) for K, V in layers]
    cache = DynamicCache()
    for lt in range(len(br.src_map)):
        y = br(kv, lt); D = y.shape[-1] // 2
        K = C.apply_rope(y[..., :D].permute(1, 0, 2).unsqueeze(0), cos17, sin17)
        cache.update(K, y[..., D:].permute(1, 0, 2).unsqueeze(0), lt, {})
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); a = ap.parse_args()
    m4, m17 = load(C.PREFILL_MODEL), load(C.DECODE_MODEL)
    br = load_bridge(a.tag)
    n = sum(p.numel() for p in br.parameters())
    print(f"{a.tag}: {n/1e6:.1f}M params, {n*2/1e6:.0f} MB bf16")
    rows = []
    for T in (256, 1024, 4096):
        ids = torch.randint(0, 150000, (1, T), device=DEV)
        pre4 = timed(lambda: m4(input_ids=ids, use_cache=True))
        pre17 = timed(lambda: m17(input_ids=ids, use_cache=True))
        for _ in range(2): bridge_path(br, m4, m17, ids)
        bt = sorted(bridge_path(br, m4, m17, ids) for _ in range(5))[2]
        rows.append(dict(T=T, prefill_4b=pre4, prefill_17b=pre17, bridge=bt))
        print(f"  T={T:5d}  4B prefill {pre4:7.1f} ms | bridge {bt:6.1f} ms ({100*bt/pre4:4.1f}% of 4B prefill)"
              f" | 1.7B prefill {pre17:6.1f} ms | split pre-side {pre4+bt:7.1f} ms", flush=True)
    json.dump(rows, open(f"{C.RESULTS}/latency_{a.tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
