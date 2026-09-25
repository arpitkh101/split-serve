"""Compact KV stitch: per-head linear map, ridge-initialised, factorised as U_h @ V_h so its
rank can be swept (U, V from the truncated SVD of the ridge solution), optional MLP branch.
Data, loss, steps and eval match kv_stitch_mlp.py; --source 17b gives the paired control that
feeds the 1.7B's own cache through the identical stitch.
"""
import sys, json, math, time, warnings, argparse, torch
import torch.nn as nn
warnings.filterwarnings("ignore")
import common as C
from probe import continuation_nll
from ridge_sweep import fit_ridge_blocked
from kv_stitch_mlp import make_cache, eval_bridge, lm_loss

DEV, OUT = "cuda:0", C.WORK
CHUNK, PROMPT, N_EVAL = 512, 256, 64


class StitchV2(nn.Module):
    def __init__(self, W, b, rank, hidden, share_heads=False):
        super().__init__()
        H, kF, F = W.shape
        self.share = share_heads
        if share_heads:
            # one [kF,F] map for all heads + per-head gain/bias.  the first stitch showed the
            # plain head-average is a bad *init* (ppl 55 vs 36); the question is
            # whether LM training recovers a shared map.  8x fewer linear params.
            self.W = nn.Parameter(W.mean(0).clone())
            self.g = nn.Parameter(torch.ones(H, F))
            self.U = self.V = None
        elif rank and rank < min(kF, F):
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)     # per head
            s = S[:, :rank].sqrt()
            self.U = nn.Parameter((U[:, :, :rank] * s[:, None, :]).contiguous())
            self.V = nn.Parameter((s[:, :, None] * Vt[:, :rank, :]).contiguous())
            self.W = None
        else:
            self.W = nn.Parameter(W.clone()); self.U = self.V = None
        self.b = nn.Parameter(b.clone())
        if hidden:
            self.mlp = nn.Sequential(nn.Linear(kF, hidden, bias=False), nn.SiLU(),
                                     nn.RMSNorm(hidden), nn.Linear(hidden, F, bias=False))
            nn.init.zeros_(self.mlp[3].weight)
        else:
            self.mlp = None

    def forward(self, x):                                   # [T, H, kF]
        if self.share:
            y = (x @ self.W) * self.g
        elif self.W is not None:
            y = torch.einsum('thf,hfg->thg', x, self.W)
        else:
            y = torch.einsum('thr,hrg->thg', torch.einsum('thf,hfr->thr', x, self.U), self.V)
        y = y + self.b
        return y + self.mlp(x) if self.mlp is not None else y


class BridgeV2(nn.Module):
    def __init__(self, src_map, W, b, rank, hidden_fn, share_heads=False):
        super().__init__()
        self.src_map = src_map
        self.layers = nn.ModuleList(
            [StitchV2(W[i], b[i], rank, hidden_fn(i), share_heads) for i in range(len(src_map))])

    def forward(self, kv4, lt):
        x = torch.cat([torch.cat([kv4[s][0][0], kv4[s][1][0]], -1).permute(1, 0, 2)
                       for s in self.src_map[lt]], dim=-1)
        return self.layers[lt](x)


def parse_hidden(spec, Lt):
    """'256' -> constant; '128:1024@20' -> 128 for layers < 20, 1024 from 20 on."""
    if "@" in spec:
        lohi, at = spec.split("@"); lo, hi = map(int, lohi.split(":")); at = int(at)
        return lambda i: lo if i < at else hi
    v = int(spec); return lambda i: v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--rank", type=int, default=32, help="0 = full rank")
    ap.add_argument("--hidden", default="256", help="MLP width, 0 = none, or lo:hi@layer")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--train-chunks", type=int, default=3000)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--source", default="4b", choices=["4b", "17b"])
    ap.add_argument("--share-heads", action="store_true")
    ap.add_argument("--uniform", action="store_true",
                    help="ablation: evenly spaced source layers instead of R2-selected")
    ap.add_argument("--random-init", action="store_true",
                    help="ablation: ignore the ridge solution, start from small random weights")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    R2 = torch.load(f"{OUT}/affinity.pt")["R2"]; Ls, Lt = R2.shape
    Ycpu = torch.load(f"{OUT}/kv_17b.pt")["kv"]
    if a.source == "4b":
        Xcpu = torch.load(f"{OUT}/kv_4b.pt")["kv"]
        if a.uniform:
            st = lambda j: max(0, min(Ls - a.k, round(j * (Ls - 1) / (Lt - 1)) - a.k // 2))
            src_map = [list(range(st(j), st(j) + a.k)) for j in range(Lt)]
        else:
            src_map = [R2[:, j].topk(a.k).indices.tolist() for j in range(Lt)]
    else:
        Xcpu = Ycpu
        st = lambda j: max(0, min(Lt - a.k, j - a.k // 2))
        src_map = [list(range(st(j), st(j) + a.k)) for j in range(Lt)]
    W, b, r2 = fit_ridge_blocked(src_map, Xcpu, Ycpu)
    del Xcpu, Ycpu
    if a.random_init:
        W = torch.randn_like(W) / W.shape[-2] ** 0.5; b = torch.zeros_like(b); r2 = float("nan")

    hidden_fn = parse_hidden(a.hidden, Lt)
    bridge = BridgeV2(src_map, W.float(), b.float(), a.rank, hidden_fn, a.share_heads).to(DEV).float()
    n_par = sum(p.numel() for p in bridge.parameters())

    if a.source == "4b":
        m4, tok = C.load_model(C.PREFILL_MODEL, DEV); m17, _ = C.load_model(C.DECODE_MODEL, DEV)
    else:
        m17, tok = C.load_model(C.DECODE_MODEL, DEV); m4 = m17
    for p in m17.parameters(): p.requires_grad_(False)
    eval_chunks = C.load_text_chunks(tok, N_EVAL, CHUNK, split="test", seed=0)
    all_tr = C.load_text_chunks(tok, a.train_chunks + 32, CHUNK, split="train", seed=777)
    train_chunks, val_chunks = all_tr[:-32], all_tr[-32:]
    base = json.load(open(f"{C.RESULTS}/baselines_eval64.json"))

    cfg = f"k={a.k} rank={a.rank} hidden={a.hidden} share={a.share_heads} rand={a.random_init} unif={a.uniform} src={a.source}"
    print(f"[{a.tag}] {cfg}  ridge R2={r2:.4f}  params={n_par/1e6:.2f}M "
          f"({100*n_par/1.72e9:.2f}% of the 1.7B)", flush=True)
    p0 = eval_bridge(bridge, m4, m17, eval_chunks)
    print(f"  step 0 (rank-{a.rank or 'full'} ridge)   test ppl = {p0:8.3f}", flush=True)

    torch.manual_seed(a.seed)                     # chunk sampling + MLP init order
    opt = torch.optim.AdamW(bridge.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    t0, run = time.time(), None
    for step in range(a.steps):
        ch = train_chunks[torch.randint(0, len(train_chunks), (1,)).item()]
        ids = ch.unsqueeze(0).to(DEV)
        with torch.no_grad():
            kv4 = C.capture_kv(m4, ids[:, :PROMPT], strip=True)
        loss = lm_loss(bridge, m17, kv4, ids, PROMPT)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        opt.step(); sched.step()
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if step % 250 == 0:
            print(f"  [{step:4d}] ema={run:.4f} train-ppl~{math.exp(run):6.2f}  {time.time()-t0:.0f}s", flush=True)
        if a.val_every and step > 0 and step % a.val_every == 0:
            print(f"        -> held-out train {eval_bridge(bridge, m4, m17, val_chunks):.3f}"
                  f"   test(16) {eval_bridge(bridge, m4, m17, eval_chunks[:16]):.3f}", flush=True)
        del kv4
    pB = eval_bridge(bridge, m4, m17, eval_chunks)
    print(f"\n[{a.tag}] RESULT  {cfg}  params={n_par/1e6:.2f}M  "
          f"step0={p0:.3f}  final={pB:.3f}  (1.7B {base['Qwen3-1.7B']:.3f}, 4B {base['Qwen3-4B']:.3f})")
    json.dump({"tag": a.tag, "seed": a.seed, "k": a.k, "rank": a.rank, "hidden": a.hidden, "source": a.source,
               "share_heads": a.share_heads, "random_init": a.random_init, "uniform": a.uniform,
               "steps": a.steps, "params_M": n_par / 1e6, "ridge_r2": r2, "step0": p0, "final": pB},
              open(f"{C.RESULTS}/v2_{a.tag}.json", "w"), indent=2)
    torch.save({"state": bridge.state_dict(), "src_map": src_map, "rank": a.rank,
                "hidden": a.hidden, "k": a.k, "share_heads": a.share_heads}, f"{OUT}/v2_{a.tag}.pt")


if __name__ == "__main__":
    main()
