"""Residual-space stitch at every layer: the decoder's own projections build the cache.

For each 1.7B layer l below the cut, one linear map (2560 -> 2048, ridge init) carries the
4B's residual stream at its best-matching layer into the 1.7B's residual at l; the 1.7B's own
input_layernorm, k_proj, k_norm, v_proj and RoPE then produce that layer's cache. No KV-to-KV
map anywhere. At the cut the mapped residual enters the 1.7B, which runs any remaining layers
itself; with --cut 28 there are none and the 1.7B does no prompt work.

Control: identity-initialised maps on the 1.7B's own residuals, same training.
Loss: next-token loss of the continuation decoded off the stitched cache; --prompt-w adds the
LM loss over prompt positions from the prefill logits (needed at cut 28, where the final map
otherwise gets no gradient). --attn-heads adds a zero-initialised causal attention branch over
the source positions to every map. --fit-only caches the ridge fits per layer.
"""
import sys, os, json, math, time, warnings, argparse, torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings("ignore")
import common as C
from transformers import DynamicCache
from residual_cut import load_sdpa, run_last_layers, OUT
from residual_cut_train import nll_of, ppl, load_corpus_chunks, N_TRAIN, CONT, N_EVAL

DEV = "cuda:0"


# ------------------------------------------------------------------ maps

class LinMap(nn.Module):
    """x_cat (sum of src widths) -> 2048, optional zero-init MLP branch."""
    def __init__(self, src, W, b, mlp_hidden=0):
        super().__init__()
        self.src = list(src); self.W = nn.Parameter(W.clone()); self.b = nn.Parameter(b.clone())
        self.mlp = None
        if mlp_hidden:
            self.mlp = nn.Sequential(nn.Linear(W.shape[0], mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, W.shape[1]))
            nn.init.zeros_(self.mlp[2].weight); nn.init.zeros_(self.mlp[2].bias)
    def forward(self, x):
        y = x @ self.W + self.b
        return y + self.mlp(x) if self.mlp is not None else y


_ROPE = {}
def rope_tables(T, dh, device):
    key = (T, dh, str(device))
    if key not in _ROPE:
        inv = 1.0 / (1e6 ** (torch.arange(0, dh, 2, device=device).float() / dh))
        f = torch.outer(torch.arange(T, device=device).float(), inv); e = torch.cat([f, f], -1)
        _ROPE[key] = (e.cos(), e.sin())
    return _ROPE[key]

def rope(x, cos, sin):                                   # x [T, h, dh]
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return x * cos[:, None] + torch.cat([-x2, x1], -1) * sin[:, None]


class AttnMap(LinMap):
    """LinMap + a zero-initialised causal self-attention branch over the SOURCE positions:
    position t may read the source residual at any position <= t, the cross-position
    mixing a per-position map cannot do.  RoPE on q/k, LayerNorm on the source (Qwen
    residual norms grow ~100x with depth), output projection starts at zero so step 0
    equals the linear map exactly."""
    def __init__(self, src, W, b, heads=8, dh=64, mlp_hidden=0):
        super().__init__(src, W, b, mlp_hidden)
        din, dout = W.shape; self.heads, self.dh = heads, dh
        self.norm = nn.LayerNorm(din)
        self.qkv = nn.Linear(din, 3 * heads * dh, bias=False)
        self.o = nn.Linear(heads * dh, dout, bias=False); nn.init.zeros_(self.o.weight)
    def forward(self, x):
        y = super().forward(x); T = x.shape[0]
        q, k, v = self.qkv(self.norm(x)).view(T, 3, self.heads, self.dh).unbind(1)
        cos, sin = rope_tables(T, self.dh, x.device); q, k = rope(q, cos, sin), rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], is_causal=True)[0]
        return y + self.o(a.transpose(0, 1).reshape(T, -1))


class ResMaps(nn.Module):
    def __init__(self, maps):
        super().__init__(); self.maps = nn.ModuleList(maps)
    def forward(self, H, l):
        m = self.maps[l]; x = torch.cat([H[j][0].float() for j in m.src], -1)
        return m(x)

    def spec(self):
        return [{"src": m.src, "in": m.W.shape[0], "mlp": (m.mlp[0].out_features if m.mlp is not None else 0),
                 "attn": ([m.heads, m.dh] if isinstance(m, AttnMap) else 0)} for m in self.maps]


def build_from_spec(spec):
    def mk(s):
        if s.get("attn"): return AttnMap(s["src"], torch.zeros(s["in"], 2048), torch.zeros(2048), s["attn"][0], s["attn"][1], s["mlp"])
        return LinMap(s["src"], torch.zeros(s["in"], 2048), torch.zeros(2048), s["mlp"])
    return ResMaps([mk(s) for s in spec])


def load_maps(tag):
    """Reconstruct a saved ResMaps (old single-source format or new spec format)."""
    ck = torch.load(f"{OUT}/reskv_{tag}.pt")
    if "spec" in ck:
        maps = build_from_spec(ck["spec"]); maps.load_state_dict(ck["maps"])
    else:                                                    # older format: W.i / b.i + js
        Ws = [ck["maps"][f"W.{i}"] for i in range(len(ck["js"]))]; bs = [ck["maps"][f"b.{i}"] for i in range(len(ck["js"]))]
        maps = ResMaps([LinMap([j], W, b) for j, W, b in zip(ck["js"], Ws, bs)])
    return maps.to(DEV).float(), ck["cut"], ck["source"]


# ------------------------------------------------------------ ridge fits

def hidden_states(model, ids):
    """Residual stream entering each layer, plus the final residual BEFORE the final norm.
    HF's hidden_states[-1] is post-norm; a cut at the last layer must target the pre-norm residual,
    otherwise run_last_layers applies the norm twice."""
    box = {}
    hk = model.model.norm.register_forward_hook(lambda m, i, o: box.__setitem__("pre", i[0]))
    try: hs = model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
    finally: hk.remove()
    return tuple(hs[:-1]) + (box["pre"],)


@torch.no_grad()
def collect(m4, m17, chunks, layers):
    """All 4B residuals (37 = embeddings + 36 layers) and 1.7B residuals entering the given layers."""
    H4, H17 = [], []
    for ch in chunks:
        ids = ch.unsqueeze(0).to(DEV)
        H4.append(torch.stack([h[0] for h in hidden_states(m4, ids)]).half().cpu())
        hs = hidden_states(m17, ids)
        H17.append(torch.stack([hs[l][0] for l in layers]).half().cpu())
    return torch.cat(H4, 1), torch.cat(H17, 1)


def ridge_fit(X, Y, lam=1e-2):
    X, Y = X.double().to(DEV), Y.double().to(DEV)
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my
    G = Xc.T @ Xc; W = torch.linalg.solve(G + lam * torch.eye(G.shape[0], device=DEV, dtype=torch.float64), Xc.T @ Yc)
    r2 = 1 - ((Yc - Xc @ W) ** 2).sum() / (Yc ** 2).sum()
    return W.float().cpu(), (my - mx @ W).float().cpu(), r2.item()


def ridge_path(l): return f"{OUT}/reskv_ridge_l{l}.pt"


def fit_layers(m4, m17, calib, layers):
    """Fit and cache: for each 1.7B layer, the best single 4B layer + r2 over all candidates."""
    H4, H17 = collect(m4, m17, calib, layers)
    for i, l in enumerate(layers):
        t0 = time.time(); best = None; r2s = []
        for j in range(H4.shape[0]):
            W, b, r2 = ridge_fit(H4[j], H17[i]); r2s.append(r2)
            if best is None or r2 > best[3]: best = (j, W, b, r2)
        torch.save({"j": best[0], "W": best[1], "b": best[2], "r2": best[3], "r2_all": r2s}, ridge_path(l))
        print(f"  layer {l:2d}: best 4B layer {best[0]:2d}  R2 {best[3]:.3f}   ({time.time()-t0:.0f}s)", flush=True)
    return H4, H17


# --------------------------------------------------------------- pipeline

def native_kv(m17, h, l, pos_emb):
    """The 1.7B's own K (with RoPE) and V for layer l from residual h [1,T,2048]."""
    layer = m17.model.layers[l]; attn = layer.self_attn
    x = layer.input_layernorm(h); T = h.shape[1]
    k = attn.k_norm(attn.k_proj(x).view(1, T, -1, attn.head_dim)).transpose(1, 2)
    v = attn.v_proj(x).view(1, T, -1, attn.head_dim).transpose(1, 2)
    cos, sin = pos_emb
    return C.apply_rope(k, cos, sin), v


def prefill(m4, m17, maps, ids, Lc, source, H=None):
    T = ids.shape[1]; cache = DynamicCache()
    pos = torch.arange(T, device=DEV).unsqueeze(0)
    pe = m17.model.rotary_emb(torch.zeros(1, T, 1, device=DEV, dtype=m17.dtype), pos)
    if H is None:
        src_model = m4 if source == "4b" else m17
        H = hidden_states(src_model, ids)
    for l in range(Lc):
        h = maps(H, l).unsqueeze(0).to(m17.dtype)
        k, v = native_kv(m17, h, l, pe)
        cache.update(k, v, l, {})
    h = maps(H, Lc).unsqueeze(0).to(m17.dtype)
    return cache, run_last_layers(m17, h, Lc, cache, T)


def continuation_logits(m17, cache, ids, T):
    cont = ids[:, T:]; pos = torch.arange(T, ids.shape[1], device=DEV).unsqueeze(0)
    return m17(input_ids=cont, past_key_values=cache, position_ids=pos, cache_position=pos[0], use_cache=True).logits[0, :-1].float()


def pipeline_nll(m4, m17, maps, ids, P, Lc, source, first_tok=False, prompt_w=0.0):
    """Mean NLL of the continuation (tokens P+1..), decoded natively off the stitched cache.
    first_tok : also score token P from the prompt's last-position logit (1 of 257 tokens).
    prompt_w  : add prompt_w * mean NLL of tokens 1..P from the prompt logits.  At cut 28 the
                prompt logits come straight from the layer-28 map, so this is the only dense
                signal that map gets; without it, it had no gradient at all (LAMBADA 8 %)."""
    cache, plg = prefill(m4, m17, maps, ids[:, :P], Lc, source)
    lg = continuation_logits(m17, cache, ids, P)
    if first_tok:
        lg = torch.cat([plg[0, -1:].float(), lg], 0); tgt = ids[0, P:]
        loss = -torch.log_softmax(lg, -1)[torch.arange(tgt.numel()), tgt].mean()
    else:
        loss = nll_of(lg, ids, P)
    if prompt_w:
        # bf16 cross-entropy: a float copy of a 2048 x 152k logit matrix plus its grad OOMs a 24 GB card
        loss = loss + prompt_w * F.cross_entropy(plg[0, :P - 1], ids[0, 1:P])
    return loss


@torch.no_grad()
def evaluate(m4, m17, maps, chunks, P, Lc, source):
    return [pipeline_nll(m4, m17, maps, ch.unsqueeze(0).to(DEV), P, Lc, source).item() for ch in chunks]


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", type=int, default=20); ap.add_argument("--source", default="4b", choices=["4b", "17b"])
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lengths", default="256,512,1024,2048"); ap.add_argument("--corpus", default="wt103")
    ap.add_argument("--calib-chunks", type=int, default=64); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cut-src", type=int, default=1, help="top-k 4B layers concatenated into the cut map")
    ap.add_argument("--cut-mlp", type=int, default=0, help="hidden width of a zero-init MLP branch on the cut map")
    ap.add_argument("--attn-heads", type=int, default=0, help="add a zero-init causal attention branch (heads) to every map")
    ap.add_argument("--attn-dh", type=int, default=64); ap.add_argument("--attn-lr", type=float, default=3e-4)
    ap.add_argument("--first-tok", action="store_true", help="include the prompt's last-position logit (token P) in the training loss")
    ap.add_argument("--prompt-w", type=float, default=0.0, help="weight of the prompt-position LM loss (tokens 1..P from the prefill logits)")
    ap.add_argument("--fit-only", action="store_true"); ap.add_argument("--layers", default="0-28")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args(); Lc = a.cut
    m4, tok = load_sdpa(C.PREFILL_MODEL); m17, _ = load_sdpa(C.DECODE_MODEL)
    calib = torch.load(f"{OUT}/calib_ids.pt")[:a.calib_chunks]

    if a.fit_only:
        lo, hi = [int(x) for x in a.layers.split("-")]
        fit_layers(m4, m17, calib, list(range(lo, hi + 1))); return

    # --- maps: cached ridge fits for 4B source, identity for the control
    maps = []
    if a.source == "4b":
        missing = [l for l in range(Lc + 1) if not os.path.exists(ridge_path(l))]
        if missing:
            print(f"[{a.tag}] fitting uncached layers {missing} ...", flush=True); fit_layers(m4, m17, calib, missing)
        for l in range(Lc):
            r = torch.load(ridge_path(l)); maps.append(LinMap([r["j"]], r["W"], r["b"]))
        r = torch.load(ridge_path(Lc))
        if a.cut_src > 1:
            top = sorted(range(len(r["r2_all"])), key=lambda j: -r["r2_all"][j])[:a.cut_src]; top.sort()
            H4, H17 = collect(m4, m17, calib, [Lc])
            W, b, r2 = ridge_fit(torch.cat([H4[j] for j in top], -1), H17[0]); del H4, H17
            print(f"  cut map: concat of 4B layers {top} -> 1.7B {Lc}, R2 {r2:.3f} (single best {r['r2']:.3f})", flush=True)
            maps.append(LinMap(top, W, b, a.cut_mlp))
        else:
            maps.append(LinMap([r["j"]], r["W"], r["b"], a.cut_mlp))
        print(f"  src layers {[m.src for m in maps]}", flush=True)
    else:
        for l in range(Lc + 1):
            maps.append(LinMap([l], torch.eye(2048), torch.zeros(2048), a.cut_mlp if l == Lc else 0))
    if a.attn_heads:
        maps = [AttnMap(m.src, m.W.data, m.b.data, a.attn_heads, a.attn_dh, a.cut_mlp if i == Lc else 0) for i, m in enumerate(maps)]
    maps = ResMaps(maps).to(DEV).float(); n_par = sum(p.numel() for p in maps.parameters())

    lens = [int(x) for x in a.lengths.split(",")]; elens = [256, 512, 1024, 2048]
    ev = {P: C.load_text_chunks(tok, N_EVAL, P + CONT, split="test", seed=0) for P in elens}
    tr = {P: load_corpus_chunks(tok, a.corpus, N_TRAIN[P], P + CONT, seed=777) for P in lens}

    def full_eval(label):
        res = {P: (lambda n: {"ppl": ppl(n), "nll": n})(evaluate(m4, m17, maps, ev[P], P, Lc, a.source)) for P in elens}
        print(f"  {label}: " + "  ".join(f"P{P} {res[P]['ppl']:.3f}" for P in elens), flush=True); return res
    r0 = full_eval("step 0"); print(f"  params {n_par/1e6:.1f}M", flush=True)

    torch.manual_seed(a.seed)
    lin = [p for n, p in maps.named_parameters() if not any(k in n for k in (".qkv.", ".o.", ".norm."))]
    att = [p for n, p in maps.named_parameters() if any(k in n for k in (".qkv.", ".o.", ".norm."))]
    groups = [{"params": lin, "lr": a.lr}] + ([{"params": att, "lr": a.attn_lr}] if att else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, a.steps)); t0, run = time.time(), None
    for step in range(a.steps):
        P = lens[torch.randint(0, len(lens), (1,)).item()]
        ids = tr[P][torch.randint(0, len(tr[P]), (1,)).item()].unsqueeze(0).to(DEV)
        loss = pipeline_nll(m4, m17, maps, ids, P, Lc, a.source, a.first_tok, a.prompt_w)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(maps.parameters(), 1.0); opt.step(); sched.step()
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if step % 250 == 0:
            print(f"  [{step:4d}] lm ema={run:.4f} train-ppl~{math.exp(run):6.2f}  {time.time()-t0:.0f}s", flush=True)
    rF = full_eval("final") if a.steps else r0
    print(f"\n[{a.tag}] RESULT cut={Lc} source={a.source} cut_src={a.cut_src} cut_mlp={a.cut_mlp} attn={a.attn_heads}x{a.attn_dh} first_tok={a.first_tok} prompt_w={a.prompt_w} seed={a.seed} steps={a.steps}  " +
          "  ".join(f"P{P}: {r0[P]['ppl']:.3f}->{rF[P]['ppl']:.3f}" for P in elens), flush=True)
    json.dump({"tag": a.tag, "cut": Lc, "source": a.source, "spec": maps.spec(), "params_M": n_par / 1e6,
               "steps": a.steps, "seed": a.seed, "corpus": a.corpus, "train_lengths": lens,
               "cut_src": a.cut_src, "cut_mlp": a.cut_mlp, "attn": [a.attn_heads, a.attn_dh], "attn_lr": a.attn_lr, "first_tok": a.first_tok, "prompt_w": a.prompt_w,
               "step0": {P: r0[P]["ppl"] for P in elens}, "final": {P: rF[P]["ppl"] for P in elens},
               "nll_final": {P: rF[P]["nll"] for P in elens}},
              open(f"{C.RESULTS}/hv2_{a.tag}.json", "w"), indent=2)
    torch.save({"maps": maps.state_dict(), "spec": maps.spec(), "cut": Lc, "source": a.source}, f"{OUT}/reskv_{a.tag}.pt")


if __name__ == "__main__":
    main()
