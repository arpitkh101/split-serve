"""The first trained stitch: per-head linear map initialised at the ridge solution plus a
zero-initialised MLP correction, trained either on MSE to the 1.7B's own cache (stage A) or on
next-token loss through the frozen 1.7B (stage B). Training starts exactly at the linear
solution, so any movement is attributable to the objective and the nonlinearity.
"""
import sys, json, math, time, warnings, argparse, torch
import torch.nn as nn
warnings.filterwarnings("ignore")
import common as C
from probe import continuation_nll
from ridge_sweep import fit_ridge_blocked
from transformers import DynamicCache

DEV, OUT = "cuda:0", C.WORK
CHUNK, PROMPT, N_EVAL = 512, 256, 64


class Stitch(nn.Module):
    """One per target layer.

    The linear part stays PER HEAD so it reproduces the ridge solution exactly
    (averaging the per-head weights, as an earlier version did, throws the fit
    away).  Only the nonlinear correction is shared across heads, which is where
    the parameter savings come from.  Its output layer is zero-initialised, so
    the module starts life numerically identical to ridge.
    """
    def __init__(self, kF, F, hidden, W, b):
        super().__init__()
        self.W = nn.Parameter(W.clone())            # [H, kF, F]
        self.b = nn.Parameter(b.clone())            # [H, F]
        self.mlp = nn.Sequential(
            nn.Linear(kF, hidden, bias=False), nn.SiLU(),
            nn.RMSNorm(hidden), nn.Linear(hidden, F, bias=False))
        nn.init.zeros_(self.mlp[3].weight)

    def forward(self, x):                           # x: [T, H, kF]
        return torch.einsum('thf,hfg->thg', x, self.W) + self.b + self.mlp(x)


class Bridge(nn.Module):
    def __init__(self, src_map, W, b, hidden):
        super().__init__()
        self.src_map = src_map
        F = W.shape[-1]
        self.layers = nn.ModuleList(
            [Stitch(W.shape[-2], F, hidden, W[i], b[i]) for i in range(len(src_map))])

    def forward(self, kv4, lt):
        """kv4: list of (K,V) [1,H,T,D] stripped -> mapped [T,H,F] for layer lt."""
        x = torch.cat([torch.cat([kv4[s][0][0], kv4[s][1][0]], -1).permute(1, 0, 2)
                       for s in self.src_map[lt]], dim=-1)
        return self.layers[lt](x)


def make_cache(bridge, kv4, m17, T, grad=False):
    cos, sin = C.rope_cos_sin(m17, T, DEV)
    cache = DynamicCache()
    for lt in range(len(bridge.src_map)):
        Y = bridge(kv4, lt)
        D = Y.shape[-1] // 2
        K = Y[..., :D].permute(1, 0, 2).unsqueeze(0)
        V = Y[..., D:].permute(1, 0, 2).unsqueeze(0)
        K = C.apply_rope(K, cos, sin).to(m17.dtype)
        V = V.to(m17.dtype)
        cache.update(K if grad else K.detach(), V if grad else V.detach(), lt, {})
    return cache


@torch.no_grad()
def eval_bridge(bridge, m4, m17, chunks):
    bridge.eval()
    nlls = []
    for ch in chunks:
        ids = ch.unsqueeze(0).to(DEV)
        kv4 = C.capture_kv(m4, ids[:, :PROMPT], strip=True)
        cache = make_cache(bridge, kv4, m17, PROMPT)
        nlls.append(continuation_nll(m17, ch, PROMPT, cache=cache))
        del kv4, cache
    bridge.train()
    return torch.tensor(nlls).mean().exp().item()


def lm_loss(bridge, m17, kv4, ids, prompt_len):
    """Next-token loss on the continuation, gradients flowing into the stitch."""
    cache = make_cache(bridge, kv4, m17, prompt_len, grad=True)
    cont = ids[:, prompt_len:]
    pos = torch.arange(prompt_len, ids.shape[1], device=DEV).unsqueeze(0)
    out = m17(input_ids=cont, past_key_values=cache,
              position_ids=pos, cache_position=pos[0], use_cache=True)
    logits = out.logits[0, :-1].float()
    tgt = ids[0, prompt_len + 1:]
    return nn.functional.cross_entropy(logits, tgt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--train-chunks", type=int, default=3000)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--source", default="4b", choices=["4b", "17b"],
                    help="17b = self-bridge control: the stitch reads the decode "
                         "model's OWN cache, so any gain is adaptation, not transfer")
    ap.add_argument("--tag", default="")
    ap.add_argument("--steps-a", type=int, default=400)
    ap.add_argument("--steps-b", type=int, default=4000)
    a = ap.parse_args()

    R2 = torch.load(f"{OUT}/affinity.pt")["R2"]; Ls, Lt = R2.shape
    Ycpu = torch.load(f"{OUT}/kv_17b.pt")["kv"]
    if a.source == "4b":
        Xcpu = torch.load(f"{OUT}/kv_4b.pt")["kv"]
        src_map = [R2[:, j].topk(a.k).indices.tolist() for j in range(Lt)]
    else:
        # same input width as the 4B bridge: a window of k of the 1.7B's own
        # layers, always containing lt itself, so ridge starts at ~identity
        Xcpu = Ycpu
        st = lambda j: max(0, min(Lt - a.k, j - a.k // 2))
        src_map = [list(range(st(j), st(j) + a.k)) for j in range(Lt)]
    print(f"fitting ridge init (k={a.k}) ...", flush=True)
    W, b, r2 = fit_ridge_blocked(src_map, Xcpu, Ycpu)
    print(f"  ridge fit R2 = {r2:.4f}")

    if a.source == "4b":
        m4, tok = C.load_model(C.PREFILL_MODEL, DEV)
        m17, _ = C.load_model(C.DECODE_MODEL, DEV)
    else:
        m17, tok = C.load_model(C.DECODE_MODEL, DEV)
        m4 = m17                      # "source" model is the decode model itself
    for p in m17.parameters(): p.requires_grad_(False)
    eval_chunks = C.load_text_chunks(tok, N_EVAL, CHUNK, split="test", seed=0)
    all_tr = C.load_text_chunks(tok, a.train_chunks + 32, CHUNK, split="train", seed=777)
    train_chunks, val_chunks = all_tr[:-32], all_tr[-32:]   # held out from training
    base = json.load(open(f"{C.RESULTS}/baselines_eval64.json"))

    bridge = Bridge(src_map, W.float(), b.float(), a.hidden).to(DEV).float()
    n_par = sum(p.numel() for p in bridge.parameters())
    print(f"stitch params: {n_par/1e6:.1f}M  "
          f"({100*n_par/sum(p.numel() for p in m17.parameters()):.1f}% of the 1.7B)")

    print("\n" + "=" * 78)
    print(f"  {'Qwen3-4B standalone':38s} ppl = {base['Qwen3-4B']:8.3f}")
    print(f"  {'Qwen3-1.7B standalone':38s} ppl = {base['Qwen3-1.7B']:8.3f}")
    p0 = eval_bridge(bridge, m4, m17, eval_chunks)
    print(f"  {'ridge init (stage 0)':38s} ppl = {p0:8.3f}")
    print("=" * 78, flush=True)

    # ---- Stage A: MSE to the 1.7B's own KV
    opt = torch.optim.AdamW(bridge.parameters(), lr=3e-4, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps_a)
    N = Ycpu.shape[1]; t0 = time.time()
    for step in range(a.steps_a):
        idx = torch.randint(0, N - 1024, (1,)).item()
        sl = slice(idx, idx + 1024)
        loss = 0.0
        for lt in range(Lt):
            x = torch.cat([Xcpu[s, sl].to(DEV).float() for s in src_map[lt]], -1)
            y = Ycpu[lt, sl].to(DEV).float()
            loss = loss + nn.functional.mse_loss(bridge.layers[lt](x), y)
        loss = loss / Lt
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 100 == 0:
            print(f"  [A {step:4d}] mse={loss.item():.4f}  {time.time()-t0:.0f}s", flush=True)
    pA = eval_bridge(bridge, m4, m17, eval_chunks)
    print(f"\n  {'after stage A (MSE)':38s} ppl = {pA:8.3f}", flush=True)

    # ---- Stage B: LM loss through the frozen decode model
    opt = torch.optim.AdamW(bridge.parameters(), lr=1e-4, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps_b)
    t0 = time.time(); run = None
    for step in range(a.steps_b):
        ch = train_chunks[torch.randint(0, len(train_chunks), (1,)).item()]
        ids = ch.unsqueeze(0).to(DEV)
        with torch.no_grad():
            kv4 = C.capture_kv(m4, ids[:, :PROMPT], strip=True)
        loss = lm_loss(bridge, m17, kv4, ids, PROMPT)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        opt.step(); sched.step()
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if step % 100 == 0:
            print(f"  [B {step:4d}] lm={loss.item():.4f} ema={run:.4f} "
                  f"train-ppl~{math.exp(run):6.2f}  {time.time()-t0:.0f}s", flush=True)
        if a.val_every and step > 0 and step % a.val_every == 0:
            pv = eval_bridge(bridge, m4, m17, val_chunks)
            pt = eval_bridge(bridge, m4, m17, eval_chunks[:16])
            print(f"        -> held-out train ppl={pv:8.3f}   test ppl={pt:8.3f}"
                  f"   (1.7B standalone {base['Qwen3-1.7B']:.3f})", flush=True)
        del kv4
    pB = eval_bridge(bridge, m4, m17, eval_chunks)

    print("\n" + "=" * 78); print("RESULT"); print("=" * 78)
    gap = base["Qwen3-1.7B"] - base["Qwen3-4B"]
    for nm, p in (("Qwen3-4B standalone", base["Qwen3-4B"]),
                  ("Qwen3-1.7B standalone", base["Qwen3-1.7B"]),
                  ("ridge only", p0), ("+ MLP, stage A (MSE)", pA),
                  ("+ MLP, stage B (LM loss)", pB)):
        print(f"  {nm:38s} ppl = {p:8.3f}")
    print(f"\n  gap between standalone models: {gap:.3f} ppl")
    print(f"  stage B vs 1.7B standalone:    {pB - base['Qwen3-1.7B']:+.3f} ppl")
    json.dump({"ridge": p0, "stageA": pA, "stageB": pB, **base},
              open(f"{C.RESULTS}/stitch_k{a.k}{a.tag}.json", "w"), indent=2)
    torch.save(bridge.state_dict(), f"{OUT}/bridge_k{a.k}{a.tag}.pt")


if __name__ == "__main__":
    main()
