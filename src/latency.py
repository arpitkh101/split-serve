"""Single-request wall clock at batch 1: prefill time for both models at several prompt lengths,
bridge time (map + RoPE + cache assembly) in fp32 and bf16, decode throughput off a prefilled
cache. Stitch weights are random here; cost depends on shapes, not values.
"""
import sys, json, copy, warnings, torch
warnings.filterwarnings("ignore")
import common as C
from kv_stitch_mlp import Bridge, make_cache

DEV, K, HIDDEN, N_DEC = "cuda:0", 8, 1024, 128
LENS = (256, 1024, 4096)


def load(name):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()


def ev():
    return torch.cuda.Event(enable_timing=True)


def timed(fn, reps=5, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        s, e = ev(), ev(); s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return sorted(ts)[len(ts) // 2]


def decode_ms(m, ids, reps=4):
    """Median ms for N_DEC greedy tokens off a freshly prefilled cache."""
    ts = []
    for r in range(reps + 1):
        out = m(input_ids=ids, use_cache=True)
        past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize(); s, e = ev(), ev(); s.record()
        for _ in range(N_DEC):
            o = m(input_ids=nxt, past_key_values=past, use_cache=True)
            past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
        e.record(); torch.cuda.synchronize()
        if r:                                       # rep 0 is warm-up
            ts.append(s.elapsed_time(e))
        del past, out
    return sorted(ts)[len(ts) // 2]


@torch.inference_mode()
def main():
    m4, m17 = load(C.PREFILL_MODEL), load(C.DECODE_MODEL)
    Ls, Lt = m4.config.num_hidden_layers, m17.config.num_hidden_layers
    H, F = 8, 256
    src_map = [list(range(min(j, Ls - K), min(j, Ls - K) + K)) for j in range(Lt)]
    W = torch.randn(Lt, H, K * F, F) * 0.01; b = torch.zeros(Lt, H, F)
    br32 = Bridge(src_map, W, b, HIDDEN).to(DEV).float().eval()
    br16 = copy.deepcopy(br32).to(torch.bfloat16)
    n_par = sum(p.numel() for p in br32.parameters())

    wgb = lambda m: sum(p.numel() * p.element_size() for p in m.parameters()) / 1e9
    print(f"resident weights: 4B {wgb(m4):.2f} GB | 1.7B {wgb(m17):.2f} GB | "
          f"stitch {n_par/1e6:.1f}M params = {n_par*4/1e9:.2f} GB fp32 / "
          f"{n_par*2/1e9:.2f} GB bf16")
    print(f"  single-model 1.7B: {wgb(m17):.2f} GB   split system: "
          f"{wgb(m4)+wgb(m17)+n_par*2/1e9:.2f} GB  "
          f"(+{100*(wgb(m4)+n_par*2/1e9)/wgb(m17):.0f}% vs 1.7B alone)\n", flush=True)

    rows = []
    for T in LENS:
        ids = torch.randint(0, 150000, (1, T), device=DEV)
        pre4 = timed(lambda: m4(input_ids=ids, use_cache=True))
        pre17 = timed(lambda: m17(input_ids=ids, use_cache=True))
        raw = C.capture_kv(m4, ids, strip=False)
        cos, sin = C.rope_cos_sin(m4, T, DEV)
        strip = timed(lambda: [C.strip_rope(k, cos, sin) for k, _ in raw])
        kv32 = C.capture_kv(m4, ids, strip=True)
        kv16 = [(k.bfloat16(), v.bfloat16()) for k, v in kv32]
        b32 = timed(lambda: make_cache(br32, kv32, m17, T))
        b16 = timed(lambda: make_cache(br16, kv16, m17, T))
        d4, d17 = decode_ms(m4, ids), decode_ms(m17, ids)
        r = dict(T=T, prefill_4b=pre4, prefill_17b=pre17, rope_strip=strip,
                 bridge_fp32=b32, bridge_bf16=b16,
                 dec_tps_4b=1000 * N_DEC / d4, dec_tps_17b=1000 * N_DEC / d17,
                 e2e_17b=pre17 + d17, e2e_4b=pre4 + d4,
                 e2e_split=pre4 + strip + b16 + d17)
        rows.append(r)
        print(f"T={T:5d}  prefill 4B {pre4:7.1f} ms | 1.7B {pre17:7.1f} ms | "
              f"strip {strip:5.1f} | bridge fp32 {b32:7.1f} / bf16 {b16:7.1f} ms | "
              f"decode 4B {r['dec_tps_4b']:5.1f} tok/s, 1.7B {r['dec_tps_17b']:5.1f} tok/s",
              flush=True)
        del raw, kv32, kv16; torch.cuda.empty_cache()

    print("\n" + "=" * 86)
    print(f"END-TO-END, one request, {N_DEC} generated tokens (ms)")
    print("=" * 86)
    print(f"  {'T':>5s} {'1.7B alone':>11s} {'4B alone':>10s} {'split(bf16)':>12s}   "
          f"{'split pre-side':>15s} {'1.7B prefill':>13s}  bridge/4B-prefill")
    for r in rows:
        pre_side = r["prefill_4b"] + r["rope_strip"] + r["bridge_bf16"]
        print(f"  {r['T']:5d} {r['e2e_17b']:11.1f} {r['e2e_4b']:10.1f} "
              f"{r['e2e_split']:12.1f}   {pre_side:15.1f} {r['prefill_17b']:13.1f}"
              f"  {100*r['bridge_bf16']/r['prefill_4b']:6.1f}%")
    json.dump(rows, open(f"{C.RESULTS}/latency.json", "w"), indent=2)


if __name__ == "__main__":
    main()
