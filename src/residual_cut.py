"""Residual-stream stitch with a native tail. Layers below a cut take KV from the 4B through the
per-layer KV stitch; at the cut, the 4B's residual stream at its best-matching layer is mapped
by one 2560 -> 2048 linear into the 1.7B's residual; the 1.7B then runs its remaining layers
over the prompt itself, writing native KV for them. Both stitches train on LM loss through the
frozen 1.7B. Control: the same pipeline fed the 1.7B's own KV and residual (identity init).
"""
import sys, json, math, time, warnings, argparse, torch
import torch.nn as nn
warnings.filterwarnings("ignore")
import common as C
from transformers import DynamicCache
from kv_stitch import BridgeV2, parse_hidden

DEV, OUT = "cuda:0", C.WORK
CHUNK, PROMPT, N_EVAL = 512, 256, 64


def load_sdpa(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16,
                                             attn_implementation="sdpa").to(DEV).eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m, AutoTokenizer.from_pretrained(name)


# ------------------------------------------------------------ calibration

@torch.no_grad()
def collect_hidden(m4, m17, ids_all, Lc, n_chunks):
    """4B hidden states at every layer and the 1.7B's residual stream entering layer Lc."""
    H4, H17 = [], []
    for ch in ids_all[:n_chunks]:
        ids = ch.unsqueeze(0).to(DEV)
        o4 = m4(input_ids=ids, output_hidden_states=True, use_cache=False)
        o17 = m17(input_ids=ids, output_hidden_states=True, use_cache=False)
        H4.append(torch.stack([h[0] for h in o4.hidden_states[1:]]).half().cpu())   # [L4, T, 2560]
        H17.append(o17.hidden_states[Lc][0].half().cpu())                            # [T, 2048]
    return torch.cat(H4, 1), torch.cat(H17, 0)


def ridge_fit(X, Y, lam=1e-2):
    """Y ~= X W + b, fp64, centred.  X [N,dx] Y [N,dy].  Returns W, b, R2."""
    X, Y = X.double().to(DEV), Y.double().to(DEV)
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my
    G = Xc.T @ Xc; W = torch.linalg.solve(G + lam * torch.eye(G.shape[0], device=DEV, dtype=torch.float64), Xc.T @ Yc)
    res = ((Yc - Xc @ W) ** 2).sum(); tot = (Yc ** 2).sum()
    return W.float(), (my - mx @ W).float(), (1 - res / tot).item()


# --------------------------------------------------------------- pipeline

class HiddenStitch(nn.Module):
    def __init__(self, W, b):
        super().__init__(); self.W = nn.Parameter(W.clone()); self.b = nn.Parameter(b.clone())
    def forward(self, h): return h @ self.W + self.b


def run_last_layers(m17, h, Lc, cache, T):
    """Run the 1.7B's layers Lc..end over hidden state h [1,T,2048], writing native KV."""
    pos = torch.arange(T, device=DEV).unsqueeze(0)
    pe = m17.model.rotary_emb(h, pos)
    for li in range(Lc, m17.config.num_hidden_layers):
        out = m17.model.layers[li](h, attention_mask=None, position_ids=pos, past_key_value=cache,
                                   use_cache=True, cache_position=pos[0], position_embeddings=pe)
        h = out[0] if isinstance(out, tuple) else out
    return m17.lm_head(m17.model.norm(h))                       # [1,T,V]


def prefill_pipeline(m4, m17, kvbr, hbr, ids_p, Lc, js, source):
    """Returns (cache with all 28 layers, logits over the prompt)."""
    T = ids_p.shape[1]
    cos17, sin17 = C.rope_cos_sin(m17, T, DEV)
    cache = DynamicCache()
    if source == "4b":
        o4 = m4(input_ids=ids_p, output_hidden_states=True, use_cache=True)
        cos4, sin4 = C.rope_cos_sin(m4, T, DEV)
        kv4 = [(C.strip_rope(K.float(), cos4, sin4), V.float()) for K, V in C._cache_layers(o4.past_key_values)]
        for lt in range(Lc):
            y = kvbr(kv4, lt); D = y.shape[-1] // 2
            K = C.apply_rope(y[..., :D].permute(1, 0, 2).unsqueeze(0), cos17, sin17)
            cache.update(K.to(m17.dtype), y[..., D:].permute(1, 0, 2).unsqueeze(0).to(m17.dtype), lt, {})
        h = hbr(o4.hidden_states[js + 1][0].float()).unsqueeze(0).to(m17.dtype)
    else:                                                        # control: the 1.7B's own
        o17 = m17(input_ids=ids_p, output_hidden_states=True, use_cache=True)
        kv17 = [(C.strip_rope(K.float(), cos17, sin17), V.float()) for K, V in C._cache_layers(o17.past_key_values)]
        for lt in range(Lc):
            y = kvbr(kv17, lt); D = y.shape[-1] // 2
            K = C.apply_rope(y[..., :D].permute(1, 0, 2).unsqueeze(0), cos17, sin17)
            cache.update(K.to(m17.dtype), y[..., D:].permute(1, 0, 2).unsqueeze(0).to(m17.dtype), lt, {})
        h = hbr(o17.hidden_states[Lc][0].float()).unsqueeze(0).to(m17.dtype)
    logits_p = run_last_layers(m17, h, Lc, cache, T)
    return cache, logits_p


def continuation_logits(m17, cache, ids, T):
    cont = ids[:, T:]; pos = torch.arange(T, ids.shape[1], device=DEV).unsqueeze(0)
    return m17(input_ids=cont, past_key_values=cache, position_ids=pos, cache_position=pos[0],
               use_cache=True).logits[0, :-1].float()


def nll_of(logits, ids):
    tgt = ids[0, PROMPT + 1:]
    return -torch.log_softmax(logits, -1)[torch.arange(tgt.numel()), tgt].mean()


@torch.no_grad()
def evaluate(m4, m17, kvbr, hbr, chunks, Lc, js, source):
    nll = []
    for ch in chunks:
        ids = ch.unsqueeze(0).to(DEV)
        cache, _ = prefill_pipeline(m4, m17, kvbr, hbr, ids[:, :PROMPT], Lc, js, source)
        nll.append(nll_of(continuation_logits(m17, cache, ids, PROMPT), ids).item())
    return torch.tensor(nll).mean().exp().item()


@torch.no_grad()
def baseline_nll(m, chunks):
    nll = []
    for ch in chunks:
        ids = ch.unsqueeze(0).to(DEV)
        lg = m(input_ids=ids, use_cache=False).logits[0, PROMPT:-1].float()
        nll.append(nll_of(lg, ids).item())
    return torch.tensor(nll).mean().exp().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", type=int, default=20, help="1.7B layer where the residual stream is stitched in")
    ap.add_argument("--source", default="4b", choices=["4b", "17b"])
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--calib-chunks", type=int, default=64); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-eval", type=int, default=N_EVAL); ap.add_argument("--tag", required=True)
    a = ap.parse_args(); Lc = a.cut

    m4, tok = load_sdpa(C.PREFILL_MODEL); m17, _ = load_sdpa(C.DECODE_MODEL)
    calib = torch.load(f"{OUT}/calib_ids.pt")
    eval_chunks = C.load_text_chunks(tok, a.n_eval, CHUNK, split="test", seed=0)
    all_tr = C.load_text_chunks(tok, 3032, CHUNK, split="train", seed=777)
    train_chunks, val_chunks = all_tr[:-32], all_tr[-32:]

    # --- hidden-state map: pick the 4B layer whose residual best predicts the 1.7B's at Lc
    print(f"[{a.tag}] collecting hidden states on {a.calib_chunks} chunks ...", flush=True)
    H4, H17 = collect_hidden(m4, m17, calib, Lc, a.calib_chunks)
    if a.source == "4b":
        best = (-1, None, None, -1e9)
        for j in range(H4.shape[0]):
            W, b, r2 = ridge_fit(H4[j], H17)
            if r2 > best[3]: best = (j, W, b, r2)
        js, Wh, bh, r2h = best
        print(f"  residual map: 4B layer {js} -> 1.7B layer {Lc} input, R2 = {r2h:.4f}", flush=True)
    else:
        js, Wh, bh, r2h = Lc - 1, torch.eye(H17.shape[1], device=DEV), torch.zeros(H17.shape[1], device=DEV), 1.0
        print(f"  control: identity map on the 1.7B's own layer-{Lc} residual", flush=True)
    del H4, H17
    hbr = HiddenStitch(Wh, bh).to(DEV).float()

    # --- KV stitch for layers < Lc: the trained k=1 stitch, trimmed
    ck = torch.load(f"{OUT}/v2_k1_full_h0{'' if a.source == '4b' else '_self'}.pt")
    Lt = len(ck["src_map"]); W0 = torch.zeros(Lt, 8, 256, 256); b0 = torch.zeros(Lt, 8, 256)
    kvbr = BridgeV2(ck["src_map"], W0, b0, 0, parse_hidden("0", Lt)); kvbr.load_state_dict(ck["state"])
    kvbr = kvbr.to(DEV).float()
    n_par = sum(p.numel() for p in hbr.parameters()) + sum(p.numel() for p in kvbr.layers[:Lc].parameters())

    p17, p4 = baseline_nll(m17, eval_chunks), baseline_nll(m4, eval_chunks)
    p0 = evaluate(m4, m17, kvbr, hbr, eval_chunks, Lc, js, a.source)
    print(f"  baselines (sdpa): 1.7B {p17:.3f}  4B {p4:.3f}   |  step 0 pipeline ppl {p0:.3f}   "
          f"trainable {n_par/1e6:.1f}M   1.7B prefill share {(28-Lc)/28:.2f}", flush=True)

    params = list(hbr.parameters()) + list(kvbr.layers[:Lc].parameters())
    torch.manual_seed(a.seed)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, a.steps))
    t0, run = time.time(), None
    for step in range(a.steps):
        ids = train_chunks[torch.randint(0, len(train_chunks), (1,)).item()].unsqueeze(0).to(DEV)
        cache, _ = prefill_pipeline(m4, m17, kvbr, hbr, ids[:, :PROMPT], Lc, js, a.source)
        loss = nll_of(continuation_logits(m17, cache, ids, PROMPT), ids)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if step % 250 == 0:
            print(f"  [{step:4d}] lm ema={run:.4f} train-ppl~{math.exp(run):6.2f}  {time.time()-t0:.0f}s", flush=True)
        if step > 0 and step % 1000 == 0:
            print(f"        -> held-out train ppl {evaluate(m4, m17, kvbr, hbr, val_chunks, Lc, js, a.source):.3f}", flush=True)
        del cache
    pB = evaluate(m4, m17, kvbr, hbr, eval_chunks, Lc, js, a.source) if a.steps else p0
    print(f"\n[{a.tag}] RESULT cut={Lc} source={a.source} src_layer={js} R2={r2h:.3f}  "
          f"step0 {p0:.3f} -> final {pB:.3f}   (1.7B {p17:.3f}, 4B {p4:.3f})")
    json.dump({"tag": a.tag, "cut": Lc, "source": a.source, "src_layer": js, "r2_hidden": r2h,
               "params_M": n_par / 1e6, "step0": p0, "final": pB, "base_17b": p17, "base_4b": p4,
               "steps": a.steps}, open(f"{C.RESULTS}/hidden_{a.tag}.json", "w"), indent=2)
    torch.save({"h": hbr.state_dict(), "kv": kvbr.state_dict(), "cut": Lc, "src_layer": js},
               f"{OUT}/hidden_{a.tag}.pt")


if __name__ == "__main__":
    main()
