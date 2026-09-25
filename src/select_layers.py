"""Source-to-target layer affinity between the 4B and the 1.7B, two ways: R2 of a ridge map
between the two caches, and KL between logit-lens next-token distributions. Used to decide
which 4B layer feeds each 1.7B layer.
"""
import sys, json, time, torch
import common as C

DEV, OUT, LAM = "cuda:0", C.WORK, 1e-2


# ------------------------------------------------------------------ R2

def r2_matrix():
    a = torch.load(f"{OUT}/kv_4b.pt"); b = torch.load(f"{OUT}/kv_17b.pt")
    X_all, Y_all = a["kv"].to(DEV), b["kv"].to(DEV)         # [L,N,H,2D] fp16
    Ls, Lt, N, H, F = X_all.shape[0], Y_all.shape[0], X_all.shape[1], \
                      X_all.shape[2], X_all.shape[3]
    print(f"  R2: source {Ls} layers, target {Lt} layers, "
          f"{N} tokens, {H} heads, {F} features/head", flush=True)

    # Keep the dumps in fp16 on device and cast one layer at a time -- the
    # full fp32 copy is ~10 GB and does not fit alongside the fp16 originals.
    mu_x = torch.stack([X_all[l].float().mean(0) for l in range(Ls)])   # [Ls,H,F]
    mu_y = torch.stack([Y_all[l].float().mean(0) for l in range(Lt)])   # [Lt,H,F]
    cx = lambda l: X_all[l].float() - mu_x[l]
    cy = lambda l: Y_all[l].float() - mu_y[l]

    eye = torch.eye(F, device=DEV) * LAM
    Gxx = torch.stack([torch.einsum('nhf,nhg->hfg', cx(l), cx(l)) for l in range(Ls)])
    Syy = torch.stack([(cy(l) ** 2).sum(dim=(0, 2)) for l in range(Lt)])   # [Lt,H]

    R2 = torch.zeros(Ls, Lt)
    t0 = time.time()
    for ls in range(Ls):
        Ainv = torch.linalg.inv(Gxx[ls] + eye)               # [H,F,F]
        Xl = cx(ls)
        for lt in range(Lt):
            Gxy = torch.einsum('nhf,nhg->hfg', Xl, cy(lt))
            W = Ainv @ Gxy
            ssres = (Syy[lt]
                     - 2 * torch.einsum('hfg,hfg->h', W, Gxy)
                     + torch.einsum('hfg,hfg->h', W, Gxx[ls] @ W))
            R2[ls, lt] = (1 - ssres / Syy[lt]).mean().cpu()
        if ls % 9 == 0:
            print(f"    src layer {ls}/{Ls}  {time.time()-t0:.0f}s", flush=True)
    del X_all, Y_all, Gxx; torch.cuda.empty_cache()
    return R2


# ------------------------------------------------------------------ KL

@torch.no_grad()
def kl_matrix(n_chunks=8, micro=128):
    ids_all = torch.load(f"{OUT}/calib_ids.pt")[:n_chunks]
    m4, _ = C.load_model(C.PREFILL_MODEL, DEV)
    m17, _ = C.load_model(C.DECODE_MODEL, DEV)
    Ls = m4.config.num_hidden_layers; Lt = m17.config.num_hidden_layers
    acc = torch.zeros(Ls, Lt, device=DEV, dtype=torch.float64); ntok = 0
    t0 = time.time()

    def lens(model, ids):
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        return out.hidden_states[1:], model.model.norm, model.lm_head

    for ci, ch in enumerate(ids_all):
        ids = ch.unsqueeze(0).to(DEV)
        h4, n4, hd4 = lens(m4, ids)
        h17, n17, hd17 = lens(m17, ids)
        T = ids.shape[1]
        for s in range(0, T, micro):
            e = min(s + micro, T)
            P = torch.stack([torch.softmax(hd4(n4(h[0, s:e])).float(), -1)
                             for h in h4]).half()                 # [Ls,t,V]
            lQ = torch.stack([torch.log_softmax(hd17(n17(h[0, s:e])).float(), -1)
                              for h in h17]).half()               # [Lt,t,V]
            ent = (P.float() * torch.log(P.float().clamp_min(1e-12))).sum(-1)  # [Ls,t]
            cross = torch.einsum('itv,jtv->ijt', P.float(), lQ.float())        # [Ls,Lt,t]
            acc += (ent.unsqueeze(1) - cross).sum(-1).double()
            ntok += e - s
            del P, lQ, cross
        del h4, h17; torch.cuda.empty_cache()
        print(f"    KL chunk {ci}/{len(ids_all)}  {time.time()-t0:.0f}s", flush=True)
    return (acc / ntok).float().cpu(), ntok


def main():
    print("=" * 72); print("LAYER AFFINITY MATRICES"); print("=" * 72)
    R2 = r2_matrix()
    KL, ntok = kl_matrix()
    print(f"  KL computed over {ntok} tokens")
    torch.save({"R2": R2, "KL": KL}, f"{OUT}/affinity.pt")

    Lt = R2.shape[1]
    sel_r2 = R2.argmax(0).tolist()
    sel_kl = KL.argmin(0).tolist()
    sel_uni = [round(i * (R2.shape[0] - 1) / (Lt - 1)) for i in range(Lt)]
    print("\n  target layer -> chosen 4B source layer")
    print(f"    {'tgt':>4s} {'R2':>5s} {'KL':>5s} {'unif':>5s}   "
          f"{'bestR2':>7s} {'minKL':>7s}")
    for j in range(Lt):
        print(f"    {j:4d} {sel_r2[j]:5d} {sel_kl[j]:5d} {sel_uni[j]:5d}   "
              f"{R2[sel_r2[j], j]:7.3f} {KL[sel_kl[j], j]:7.3f}")
    mono = lambda s: all(s[i] <= s[i+1] for i in range(len(s)-1))
    print(f"\n  monotonic?  R2={mono(sel_r2)}  KL={mono(sel_kl)}")
    print(f"  distinct sources used:  R2={len(set(sel_r2))}  KL={len(set(sel_kl))}"
          f"  (out of {Lt} targets)")
    print(f"  agreement R2 vs KL: {sum(a==b for a,b in zip(sel_r2,sel_kl))}/{Lt} exact,"
          f" mean |diff| = {sum(abs(a-b) for a,b in zip(sel_r2,sel_kl))/Lt:.2f} layers")
    json.dump({"r2": sel_r2, "kl": sel_kl, "uniform": sel_uni},
              open(f"{C.RESULTS}/layer_maps.json", "w"), indent=2)


if __name__ == "__main__":
    main()
