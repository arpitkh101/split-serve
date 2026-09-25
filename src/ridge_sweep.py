"""Sweep the number of 4B source layers feeding each 1.7B layer (ridge maps, R2-selected),
with two controls: the same machinery fitting the 1.7B to itself (must land at identity) and
injecting the 1.7B's real cache (must equal the baseline). Gram matrices are accumulated in
token blocks so large k fits in memory.
"""
import sys, json, time, torch
import common as C
from probe import continuation_nll
from ridge_fit import bridge_cache
from transformers import DynamicCache

DEV, OUT, LAM = "cuda:0", C.WORK, 1e-2
CHUNK, PROMPT, N_EVAL, BLK = 512, 256, 64, 4096


def fit_ridge_blocked(src_map, Xcpu, Ycpu, lam=LAM):
    """Accumulate X^T X and X^T Y in token blocks; solve per head."""
    Lt, N, H, F = Ycpu.shape[0], Ycpu.shape[1], Ycpu.shape[2], Ycpu.shape[3]
    Ws, bs, r2s = [], [], []
    for lt in range(Lt):
        srcs = src_map[lt]; kF = len(srcs) * F
        G = torch.zeros(H, kF, kF, device=DEV, dtype=torch.float64)
        Gxy = torch.zeros(H, kF, F, device=DEV, dtype=torch.float64)
        sx = torch.zeros(H, kF, device=DEV, dtype=torch.float64)
        sy = torch.zeros(H, F, device=DEV, dtype=torch.float64)
        syy = torch.zeros(H, F, device=DEV, dtype=torch.float64)
        for s in range(0, N, BLK):
            e = min(s + BLK, N)
            X = torch.cat([Xcpu[l, s:e].to(DEV).float() for l in srcs], -1)
            Y = Ycpu[lt, s:e].to(DEV).float()
            G += torch.einsum('nhf,nhg->hfg', X, X).double()
            Gxy += torch.einsum('nhf,nhg->hfg', X, Y).double()
            sx += X.sum(0).double(); sy += Y.sum(0).double()
            syy += (Y ** 2).sum(0).double()
            del X, Y
        # centre via the accumulated sums
        Gc = G - torch.einsum('hf,hg->hfg', sx, sx) / N
        Gxyc = Gxy - torch.einsum('hf,hg->hfg', sx, sy) / N
        Syy = syy - sy ** 2 / N                                    # [H,F]
        eye = torch.eye(kF, device=DEV, dtype=torch.float64) * lam
        # solve one head at a time: the batched call peaks at 8x the memory
        W = torch.stack([torch.linalg.solve(Gc[h] + eye, Gxyc[h])
                         for h in range(H)])
        ssres = (Syy.sum(-1)
                 - 2 * torch.einsum('hfg,hfg->h', W, Gxyc)
                 + torch.einsum('hfg,hfg->h', W, Gc @ W))
        r2s.append((1 - ssres / Syy.sum(-1)).mean().item())
        b = sy / N - torch.einsum('hf,hfg->hg', sx / N, W)
        Ws.append(W.float()); bs.append(b.float())
        del G, Gxy, Gc, Gxyc
        torch.cuda.empty_cache()
    return torch.stack(Ws), torch.stack(bs), sum(r2s) / len(r2s)


@torch.no_grad()
def run(name, src_model, src_map, W, b, m17, chunks, identity=False):
    nlls = []
    for ch in chunks:
        ids = ch.unsqueeze(0).to(DEV)
        kv = C.capture_kv(src_model, ids[:, :PROMPT], strip=True)
        cache = bridge_cache(kv, src_map, W, b, m17, m17.dtype, identity)
        nlls.append(continuation_nll(m17, ch, PROMPT, cache=cache))
        del kv, cache
    ppl = torch.tensor(nlls).mean().exp().item()
    print(f"  {name:40s} ppl = {ppl:9.3f}", flush=True)
    return ppl


def main():
    aff = torch.load(f"{OUT}/affinity.pt"); R2 = aff["R2"]; Ls, Lt = R2.shape
    Xcpu = torch.load(f"{OUT}/kv_4b.pt")["kv"]
    Ycpu = torch.load(f"{OUT}/kv_17b.pt")["kv"]
    m4, tok = C.load_model(C.PREFILL_MODEL, DEV)
    m17, _ = C.load_model(C.DECODE_MODEL, DEV)
    chunks = C.load_text_chunks(tok, N_EVAL, CHUNK, split="test", seed=0)
    base = json.load(open(f"{C.RESULTS}/baselines.json"))

    print("=" * 78); print("CONTROLS"); print("=" * 78)
    print(f"  {'Qwen3-1.7B standalone':40s} ppl = {base['Qwen3-1.7B']:9.3f}")
    print(f"  {'Qwen3-4B standalone':40s} ppl = {base['Qwen3-4B']:9.3f}")
    ident = [[j] for j in range(Lt)]
    Wz = torch.zeros(Lt, 8, 256, 256, device=DEV)
    run("B oracle: 1.7B's real KV injected", m17, ident, Wz, Wz[:, :, 0],
        m17, chunks, identity=True)
    Ws, bs, r2 = fit_ridge_blocked(ident, Ycpu, Ycpu)
    print(f"      (self-map fit R2 = {r2:.4f}, should be ~1.000)")
    run("A self-map: 1.7B -> ridge -> 1.7B", m17, ident, Ws, bs, m17, chunks)
    del Ws, bs; torch.cuda.empty_cache()

    print("\n" + "=" * 78); print("4B -> 1.7B, SOURCE-LAYER COUNT SWEEP (R2 selection)")
    print("=" * 78)
    res = {}
    for k in (1, 2, 4, 8, 16):
        sm = [R2[:, j].topk(k).indices.tolist() for j in range(Lt)]
        t0 = time.time()
        W, b, r2 = fit_ridge_blocked(sm, Xcpu, Ycpu)
        ppl = run(f"k={k:2d}  (fit R2 = {r2:.4f})", m4, sm, W, b, m17, chunks)
        res[k] = {"ppl": ppl, "r2": r2}
        print(f"      fit+eval {time.time()-t0:.0f}s", flush=True)
        del W, b; torch.cuda.empty_cache()
    json.dump(res, open(f"{C.RESULTS}/sweep_k.json", "w"), indent=2)


if __name__ == "__main__":
    main()
