"""Fit the per-head ridge KV bridge on calibration data and evaluate it: the 4B prefills, the
map runs in RoPE-free space, the 1.7B's RoPE is re-applied, and the 1.7B decodes the
continuation off the mapped cache. Scored on the same continuation tokens as the baselines.
"""
import sys, json, time, argparse, torch
import common as C
from probe import continuation_nll
from transformers import DynamicCache

DEV, OUT, LAM = "cuda:0", C.WORK, 1e-2
CHUNK, PROMPT, N_EVAL = 512, 256, 128


# ------------------------------------------------------------------ fitting

def fit_ridge(src_map, X_all, Y_all, lam=LAM):
    """src_map[lt] = list of 4B source layers feeding target layer lt.

    Returns W [Lt,H,k*F,F] and b [Lt,H,F] such that  Y ~= X @ W + b.
    """
    Lt, H, F = Y_all.shape[0], Y_all.shape[2], Y_all.shape[3]
    Ws, bs = [], []
    for lt in range(Lt):
        srcs = src_map[lt]
        X = torch.cat([X_all[s].float() for s in srcs], dim=-1)   # [N,H,k*F]
        Y = Y_all[lt].float()                                     # [N,H,F]
        mx, my = X.mean(0), Y.mean(0)
        Xc, Yc = X - mx, Y - my
        G = torch.einsum('nhf,nhg->hfg', Xc, Xc)
        Gxy = torch.einsum('nhf,nhg->hfg', Xc, Yc)
        eye = torch.eye(G.shape[-1], device=X.device) * lam
        W = torch.linalg.solve(G + eye, Gxy)                      # [H,kF,F]
        b = my - torch.einsum('hf,hfg->hg', mx, W)
        Ws.append(W); bs.append(b)
        del X, Y, Xc, Yc, G, Gxy
    return torch.stack(Ws), torch.stack(bs)


# ------------------------------------------------------------------ bridge

@torch.no_grad()
def bridge_cache(kv4, src_map, W, b, m17, dtype, identity=False):
    """4B stripped KV -> mapped, RoPE'd DynamicCache for the 1.7B."""
    T = kv4[0][0].shape[2]
    cos, sin = C.rope_cos_sin(m17, T, DEV)
    cache = DynamicCache()
    for lt in range(len(src_map)):
        srcs = src_map[lt]
        # [1,H,T,2D] -> [T,H,k*2D]
        X = torch.cat([torch.cat([kv4[s][0][0], kv4[s][1][0]], -1).permute(1, 0, 2)
                       for s in srcs], dim=-1).float()
        if identity:
            Y = X[..., :W.shape[-1]]                # pass the 4B KV straight through
        else:
            Y = torch.einsum('thf,hfg->thg', X, W[lt]) + b[lt]
        D = Y.shape[-1] // 2
        K = Y[..., :D].permute(1, 0, 2).unsqueeze(0)            # [1,H,T,D]
        V = Y[..., D:].permute(1, 0, 2).unsqueeze(0)
        cache.update(C.apply_rope(K, cos, sin).to(dtype), V.to(dtype), lt, {})
    return cache


@torch.no_grad()
def evaluate(name, src_map, W, b, m4, m17, chunks, identity=False):
    nlls, t0 = [], time.time()
    for ch in chunks:
        ids = ch.unsqueeze(0).to(DEV)
        kv4 = C.capture_kv(m4, ids[:, :PROMPT], strip=True)
        cache = bridge_cache(kv4, src_map, W, b, m17, m17.dtype, identity)
        nlls.append(continuation_nll(m17, ch, PROMPT, cache=cache))
        del kv4, cache
    nll = torch.tensor(nlls)
    ppl = nll.mean().exp().item()
    se = (nll.std() / len(nll) ** 0.5).item()
    print(f"  {name:34s} ppl = {ppl:7.3f}   (nll {nll.mean():.4f} +/- {se:.4f})"
          f"   [{time.time()-t0:.0f}s]", flush=True)
    return ppl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=1)
    args = ap.parse_args()

    aff = torch.load(f"{OUT}/affinity.pt")
    R2, KL = aff["R2"], aff["KL"]
    Ls, Lt = R2.shape

    a = torch.load(f"{OUT}/kv_4b.pt"); bdump = torch.load(f"{OUT}/kv_17b.pt")
    X_all, Y_all = a["kv"].to(DEV), bdump["kv"].to(DEV)

    def topk_map(score, k, largest):
        return [score[:, j].topk(k, largest=largest).indices.tolist()
                for j in range(Lt)]

    maps = {
        "uniform":  [[round(j * (Ls - 1) / (Lt - 1))] for j in range(Lt)],
        "KL-min":   topk_map(KL, args.topk, largest=False),
        "R2-max":   topk_map(R2, args.topk, largest=True),
    }

    m4, tok = C.load_model(C.PREFILL_MODEL, DEV)
    m17, _ = C.load_model(C.DECODE_MODEL, DEV)
    chunks = C.load_text_chunks(tok, N_EVAL, CHUNK, split="test", seed=0)

    print("=" * 78)
    print(f"BRIDGE EVALUATION  (top-k = {args.topk}, {N_EVAL} test chunks, "
          f"{CHUNK-PROMPT-1} scored tokens each)")
    print("=" * 78)
    base = json.load(open(f"{C.RESULTS}/baselines.json"))
    print(f"  {'Qwen3-4B standalone (upper bound)':34s} ppl = {base['Qwen3-4B']:7.3f}")
    print(f"  {'Qwen3-1.7B standalone (baseline)':34s} ppl = {base['Qwen3-1.7B']:7.3f}")
    print("  " + "-" * 74)

    res = {}
    ident = [[round(j * (Ls - 1) / (Lt - 1))] for j in range(Lt)]
    Wd = torch.zeros(Lt, 8, 256, 256, device=DEV)
    res["naive (uniform, no map)"] = evaluate(
        "naive: uniform layers, NO map", ident, Wd, Wd[:, :, 0], m4, m17,
        chunks, identity=True)

    for nm, sm in maps.items():
        W, b = fit_ridge(sm, X_all, Y_all)
        res[f"ridge {nm}"] = evaluate(f"ridge, {nm} layer selection",
                                      sm, W, b, m4, m17, chunks)
        del W, b; torch.cuda.empty_cache()

    res["baseline_1.7B"] = base["Qwen3-1.7B"]; res["upper_4B"] = base["Qwen3-4B"]
    json.dump(res, open(f"{C.RESULTS}/results_k{args.topk}.json", "w"),
              indent=2)

    gap = base["Qwen3-1.7B"] - base["Qwen3-4B"]
    print("  " + "-" * 74)
    print(f"  gap between standalone models: {gap:.3f} ppl")
    for k, v in res.items():
        if k.startswith(("ridge", "naive")):
            closed = 100 * (base["Qwen3-1.7B"] - v) / gap
            print(f"    {k:34s} closes {closed:+6.1f}% of the gap")


if __name__ == "__main__":
    main()
