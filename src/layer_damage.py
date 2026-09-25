"""Where does the ridge bridge lose quality? Re-baseline on the 64-chunk eval set, extend the
source-count sweep to k=24, then bridge only a prefix or suffix of the 1.7B's layers from the
4B and keep the rest as its own cache. Also per-layer R2, to see whether damage tracks fit.
"""
import sys, json, warnings, torch
warnings.filterwarnings("ignore")
import common as C
from probe import continuation_nll
from ridge_sweep import fit_ridge_blocked
from transformers import DynamicCache

DEV, OUT = "cuda:0", C.WORK
CHUNK, PROMPT, N_EVAL = 512, 256, 64


@torch.no_grad()
def hybrid_eval(m4, m17, chunks, src_map, W, b, mapped_layers):
    """Layers in `mapped_layers` come from the 4B via the bridge; the rest are
    the 1.7B's own real KV."""
    nlls = []
    for ch in chunks:
        ids = ch.unsqueeze(0).to(DEV)
        p = ids[:, :PROMPT]
        kv4 = C.capture_kv(m4, p, strip=True)
        kv17 = C.capture_kv(m17, p, strip=True)
        cos, sin = C.rope_cos_sin(m17, PROMPT, DEV)
        cache = DynamicCache()
        for lt in range(len(src_map)):
            if lt in mapped_layers:
                X = torch.cat([torch.cat([kv4[s][0][0], kv4[s][1][0]], -1)
                               .permute(1, 0, 2) for s in src_map[lt]], -1).float()
                Y = torch.einsum('thf,hfg->thg', X, W[lt]) + b[lt]
                D = Y.shape[-1] // 2
                K = Y[..., :D].permute(1, 0, 2).unsqueeze(0)
                V = Y[..., D:].permute(1, 0, 2).unsqueeze(0)
            else:
                K, V = kv17[lt]
            cache.update(C.apply_rope(K, cos, sin).to(m17.dtype),
                         V.to(m17.dtype), lt, {})
        nlls.append(continuation_nll(m17, ch, PROMPT, cache=cache))
        del kv4, kv17, cache
    return torch.tensor(nlls).mean().exp().item()


def main():
    aff = torch.load(f"{OUT}/affinity.pt"); R2 = aff["R2"]; Ls, Lt = R2.shape
    Xcpu = torch.load(f"{OUT}/kv_4b.pt")["kv"]
    Ycpu = torch.load(f"{OUT}/kv_17b.pt")["kv"]
    m4, tok = C.load_model(C.PREFILL_MODEL, DEV)
    m17, _ = C.load_model(C.DECODE_MODEL, DEV)
    chunks = C.load_text_chunks(tok, N_EVAL, CHUNK, split="test", seed=0)

    print("=" * 78); print(f"RE-BASELINED ON THE EVAL SET ({N_EVAL} chunks)")
    print("=" * 78)
    b4 = torch.tensor([continuation_nll(m4, c, PROMPT) for c in chunks]).mean().exp().item()
    b17 = torch.tensor([continuation_nll(m17, c, PROMPT) for c in chunks]).mean().exp().item()
    print(f"  Qwen3-4B   standalone   ppl = {b4:8.3f}")
    print(f"  Qwen3-1.7B standalone   ppl = {b17:8.3f}     gap = {b17-b4:.3f}")
    json.dump({"Qwen3-4B": b4, "Qwen3-1.7B": b17},
              open(f"{C.RESULTS}/baselines_eval64.json", "w"), indent=2)

    print("\n" + "=" * 78); print("SOURCE-COUNT SWEEP, EXTENDED"); print("=" * 78)
    from ridge_sweep import run
    sweep = json.load(open(f"{C.RESULTS}/sweep_k.json"))
    for k in (24,):
        sm = [R2[:, j].topk(k).indices.tolist() for j in range(Lt)]
        W, b, r2 = fit_ridge_blocked(sm, Xcpu, Ycpu)
        ppl = run(f"k={k:2d}  (fit R2 = {r2:.4f})", m4, sm, W, b, m17, chunks)
        sweep[str(k)] = {"ppl": ppl, "r2": r2}
        del W, b; torch.cuda.empty_cache()
    json.dump(sweep, open(f"{C.RESULTS}/sweep_k.json", "w"), indent=2)
    print("\n   k    fit R2      ppl")
    for k in sorted(sweep, key=int):
        print(f"  {int(k):3d}    {sweep[k]['r2']:.4f}   {sweep[k]['ppl']:9.3f}")

    print("\n" + "=" * 78); print("LAYER-DAMAGE PROFILE (k=8)"); print("=" * 78)
    K = 8
    sm = [R2[:, j].topk(K).indices.tolist() for j in range(Lt)]
    W, b, _ = fit_ridge_blocked(sm, Xcpu, Ycpu)
    print("  bridge only the FIRST n layers, rest = 1.7B's own real KV:")
    for n in (0, 4, 8, 14, 20, 28):
        ppl = hybrid_eval(m4, m17, chunks, sm, W, b, set(range(n)))
        print(f"    first {n:2d} layers bridged   ppl = {ppl:9.3f}", flush=True)
    print("  bridge only the LAST n layers:")
    for n in (4, 8, 14, 20):
        ppl = hybrid_eval(m4, m17, chunks, sm, W, b, set(range(Lt - n, Lt)))
        print(f"    last  {n:2d} layers bridged   ppl = {ppl:9.3f}", flush=True)

    print("\n" + "=" * 78); print("PER-LAYER FIT QUALITY (k=8)"); print("=" * 78)
    per = []
    for lt in range(Lt):
        _, _, r = fit_ridge_blocked([sm[lt]], Xcpu, Ycpu[lt:lt+1])
        per.append(r)
    print("  tgt layer :  " + " ".join(f"{i:5d}" for i in range(0, Lt, 2)))
    print("  R2        :  " + " ".join(f"{per[i]:5.2f}" for i in range(0, Lt, 2)))
    json.dump(per, open(f"{C.RESULTS}/per_layer_r2.json", "w"), indent=2)


if __name__ == "__main__":
    main()