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


def prefill(m4, m17, maps, ids, Lc, source, H=None, last_only=False, ckpt=False):
    """ckpt=True recomputes each layer's map + projections in the backward pass instead of keeping
    their activations; needed to train on 8k-token prompts within 24 GB."""
    T = ids.shape[1]; cache = DynamicCache()
    pos = torch.arange(T, device=DEV).unsqueeze(0)
    pe = m17.model.rotary_emb(torch.zeros(1, T, 1, device=DEV, dtype=m17.dtype), pos)
    if H is None:
        src_model = m4 if source == "4b" else m17
        H = hidden_states(src_model, ids)
    def layer_kv(l, *Hs):
        return native_kv(m17, maps(Hs, l).unsqueeze(0).to(m17.dtype), l, pe)
    for l in range(Lc):
        if ckpt and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            k, v = checkpoint(layer_kv, l, *H, use_reentrant=False)
        else:
            k, v = layer_kv(l, *H)
        cache.update(k, v, l, {})
    h = maps(H, Lc).unsqueeze(0).to(m17.dtype)
    if last_only and Lc == m17.config.num_hidden_layers:          # no native tail: skip the lm_head over T positions
        return cache, m17.lm_head(m17.model.norm(h[:, -1:]))
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


def _kl_chunk(ls, lt):
    """Sum over positions of KL(teacher || student), both given as bf16 logits; float inside."""
    lt = torch.log_softmax(lt.float(), -1); ls = torch.log_softmax(ls.float(), -1)
    return (lt.exp() * (lt - ls)).sum()


def pipeline_kl(m4, m17, maps, ids, P, Lc, source, prompt_w=1.0, chunk=256):
    """Receiver-matching objective: KL(native 1.7B || pipeline), with the native 1.7B doing its
    own prefill as the teacher.  Scored densely: every prompt position from the prefill logits,
    and every continuation token decoded off the stitched cache vs off the native cache.  One
    extra no-grad 1.7B forward per step.  Targets parity with the receiver's own prefill directly,
    instead of next-token loss on text, which rewards adaptation as much as transfer.
    The KL is computed in position chunks under activation checkpointing: a float log-softmax of
    a 2k x 152k logit matrix, times four, does not fit next to the models on a 24 GB card."""
    from torch.utils.checkpoint import checkpoint
    with torch.no_grad():
        t = m17(input_ids=ids, use_cache=False).logits[0]           # bf16 [T, V]
    cache, plg = prefill(m4, m17, maps, ids[:, :P], Lc, source)
    lg = continuation_logits(m17, cache, ids, P)                     # float [cont-1, V], positions P..end-1
    sc = torch.cat([plg[0, -1:].float(), lg], 0).to(t.dtype)         # positions P-1..end-1
    sp = plg[0, :P - 1]                                               # positions 0..P-2
    tc, tp = t[P - 1:-1], t[:P - 1]
    def total(ls, lt):
        acc = 0.0
        for i in range(0, ls.shape[0], chunk):
            acc = acc + checkpoint(_kl_chunk, ls[i:i + chunk], lt[i:i + chunk], use_reentrant=False)
        return acc / ls.shape[0]
    return total(sc, tc) + prompt_w * total(sp, tp)


class AttnTap:
    """Capture every layer's attention output (the input to o_proj) during a forward."""
    def __init__(self, m17):
        self.out = {}; self.h = [l.self_attn.o_proj.register_forward_pre_hook(self._mk(i)) for i, l in enumerate(m17.model.layers)]
    def _mk(self, i):
        def hook(mod, args): self.out[i] = args[0]
        return hook
    def remove(self):
        for h in self.h: h.remove()


def prompt_attn_out(m17, maps, H, cache, l, pe, idx):
    """Student attention output (the o_proj input) at prompt positions idx for layer l: queries from
    the mapped residual, keys and values from the stitched cache, causal over the prompt."""
    layer = m17.model.layers[l]; attn = layer.self_attn; cos, sin = pe; S = idx.numel()
    x = layer.input_layernorm(maps(H, l).unsqueeze(0).to(m17.dtype)[:, idx])
    q = C.apply_rope(attn.q_norm(attn.q_proj(x).view(1, S, -1, attn.head_dim)).transpose(1, 2), cos[:, idx], sin[:, idx])
    k, v = cache.key_cache[l], cache.value_cache[l]; full = S == k.shape[2]; g = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)       # GQA by hand: enable_gqa falls back to the math kernel (8k x 8k scores)
    mask = None if full else (torch.arange(k.shape[2], device=DEV)[None, :] <= idx[:, None])
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=full)
    return o.transpose(1, 2).reshape(1, S, -1)


def pipeline_attn(m4, m17, maps, ids, P, Lc, source, prompt_w=1.0, kl_w=1.0, prompt_kl=True, attn_w=1.0, prompt_attn=False, pattn_pos=8192):
    """Attention-matching objective.  The 1.7B's queries at the continuation positions must
    produce the same per-layer attention outputs over the stitched cache as over the 1.7B's own
    cache: that is what retrieval into the context needs, and what next-token or logit matching
    on local text does not enforce.  Loss = mean over layers of normalised MSE between attention
    outputs (student: decode off the stitched cache; teacher: the native 1.7B on the same text),
    plus kl_w * the receiver-matching logit KL of pipeline_kl.  prompt_attn: also match the
    attention outputs at the prompt positions (all of them, or a random pattn_pos of them), the
    student's queries coming from the mapped residuals; attn_w scales both attention terms."""
    from torch.utils.checkpoint import checkpoint
    tap = AttnTap(m17); H = None
    try:
        with torch.no_grad():
            keep = 0 if prompt_kl else ids.shape[1] - P + 1                   # 0 = all positions
            t = m17(input_ids=ids, use_cache=False, logits_to_keep=keep).logits[0]
            T_attn = {i: a[0, P:-1].float() for i, a in tap.out.items()}    # continuation positions P..end-1
            if prompt_attn:
                full = P <= pattn_pos
                idx = torch.arange(P, device=DEV) if full else torch.randperm(P, device=DEV)[:pattn_pos].sort().values
                T_p = {i: (a[0, :P] if full else a[0, idx]) for i, a in tap.out.items()}
                H = hidden_states(m4 if source == "4b" else m17, ids[:, :P])
        tap.out = {}
        cache, plg = prefill(m4, m17, maps, ids[:, :P], Lc, source, H=H, last_only=not prompt_kl, ckpt=not prompt_kl)
        lg = continuation_logits(m17, cache, ids, P)
        S_attn = {i: a[0, :-1].float() for i, a in tap.out.items()}        # same positions (last input token dropped to align with lg)
    finally:
        tap.remove()
    attn = sum(((S_attn[i] - T_attn[i]) ** 2).mean() / (T_attn[i].var() + 1e-6) for i in T_attn) / len(T_attn)
    if prompt_attn:
        pe = m17.model.rotary_emb(torch.zeros(1, P, 1, device=DEV, dtype=m17.dtype), torch.arange(P, device=DEV).unsqueeze(0))
        def pa(l):                                                           # checkpointed per layer: only the scalar survives the forward
            def f(*Hs):
                S_p = prompt_attn_out(m17, maps, Hs, cache, l, pe, idx)[0].float(); T = T_p[l].float()
                return ((S_p - T) ** 2).mean() / (T.var() + 1e-6)
            return checkpoint(f, *H, use_reentrant=False)
        attn = attn + sum(pa(l) for l in T_p) / len(T_p)
    sc = torch.cat([plg[0, -1:].float(), lg], 0).to(t.dtype); tc = t[P - 1:-1] if prompt_kl else t[:-1]
    def total(ls, lt, chunk=256):
        acc = 0.0
        for i in range(0, ls.shape[0], chunk):
            acc = acc + checkpoint(_kl_chunk, ls[i:i + chunk], lt[i:i + chunk], use_reentrant=False)
        return acc / ls.shape[0]
    loss = attn_w * attn + kl_w * total(sc, tc)
    if prompt_kl: loss = loss + kl_w * prompt_w * total(plg[0, :P - 1], t[:P - 1])
    return loss


@torch.no_grad()
def evaluate(m4, m17, maps, chunks, P, Lc, source):
    return [pipeline_nll(m4, m17, maps, ch.unsqueeze(0).to(DEV), P, Lc, source).item() for ch in chunks]


def load_qa_examples(tok, n, seed, lengths=(512, 1024, 2048)):
    """Multi-document QA training examples from the HotpotQA TRAIN split (evaluation uses
    validation): the same prompt format as longctx_eval, padded with distractor documents to a
    length drawn from `lengths`, followed by the gold answer and a newline.  Returns a list of
    (ids [1, T], P) with P the prompt length, so the loss falls on the answer tokens only: the
    retrieval signal that wikitext continuation never provides."""
    import random
    from datasets import load_dataset
    import longctx_eval as LC
    ds = list(load_dataset("hotpotqa/hotpot_qa", "distractor")["train"])
    rng = random.Random(seed); rng.shuffle(ds); ds = ds[:n + 3000]
    cache = {}
    def ntok(t):
        if t not in cache: cache[t] = len(tok(t, add_special_tokens=False).input_ids)
        return cache[t]
    def doc(t, ss): return f"Document ({t}): {''.join(ss).strip()}\n\n"
    pool = [doc(t, ss) for ex in ds[n:] for t, ss in zip(ex["context"]["title"], ex["context"]["sentences"])]
    out = []
    for ex in ds[:n]:
        own = [doc(t, ss) for t, ss in zip(ex["context"]["title"], ex["context"]["sentences"])]
        q = f"Question: {ex['question']}\nAnswer:"
        L = rng.choice(lengths); budget = L - ntok(LC.HEADER) - ntok(q) - sum(ntok(d) for d in own) - 8
        docs = list(own)
        for d in rng.sample(pool, 200):
            if ntok(d) > budget: continue
            docs.append(d); budget -= ntok(d)
            if budget < 40: break
        rng.shuffle(docs)
        prompt = LC.HEADER + "".join(docs) + q
        pid = tok(prompt, return_tensors="pt").input_ids
        aid = tok(" " + ex["answer"].strip() + "\n", return_tensors="pt", add_special_tokens=False).input_ids
        if aid.shape[1] < 2: continue
        out.append((torch.cat([pid, aid], 1), pid.shape[1]))
    print(f"  qa examples: {len(out)}, prompt tokens {min(p for _, p in out)}-{max(p for _, p in out)}", flush=True)
    return out


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", type=int, default=20); ap.add_argument("--source", default="4b", choices=["4b", "17b"])
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lengths", default="256,512,1024,2048")
    ap.add_argument("--corpus", default="wt103", help="wt103 | hotpot | qamix: alternate QA examples (loss on the answer) with wt103 chunks; qamix = hotpot, drop, musique, coqa")
    ap.add_argument("--qa-lengths", default="512,1024,2048", help="target prompt lengths for the QA examples")
    ap.add_argument("--calib-chunks", type=int, default=64); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cut-src", type=int, default=1, help="top-k 4B layers concatenated into the cut map")
    ap.add_argument("--multi-src", type=int, default=1, help="top-k 4B layers concatenated into the maps of --multi-layers")
    ap.add_argument("--multi-layers", default="", help="band of 1.7B layers lo-hi (hi exclusive) that get --multi-src source layers")
    ap.add_argument("--cut-mlp", type=int, default=0, help="hidden width of a zero-init MLP branch on the cut map")
    ap.add_argument("--attn-heads", type=int, default=0, help="add a zero-init causal attention branch (heads) to every map")
    ap.add_argument("--attn-dh", type=int, default=64); ap.add_argument("--attn-lr", type=float, default=3e-4)
    ap.add_argument("--first-tok", action="store_true", help="include the prompt's last-position logit (token P) in the training loss")
    ap.add_argument("--prompt-w", type=float, default=0.0, help="weight of the prompt-position LM loss (tokens 1..P from the prefill logits)")
    ap.add_argument("--teacher", default="", choices=["", "17b", "attn"], help="17b: KL to the native 1.7B's own-prefill distribution; attn: that plus per-layer attention-output matching at the continuation positions")
    ap.add_argument("--attn-w", type=float, default=1.0, help="weight of the attention-matching terms (teacher attn)")
    ap.add_argument("--prompt-attn", action="store_true", help="also match attention outputs at the prompt positions (teacher attn)")
    ap.add_argument("--pattn-pos", type=int, default=8192, help="prompt positions scored by --prompt-attn; a random subset above this")
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
        ml = set(range(*[int(x) for x in a.multi_layers.split("-")])) if a.multi_src > 1 and a.multi_layers else set()
        ml = {l for l in ml if l < Lc}
        if ml:                                                       # two or more 4B source layers for a chosen band of maps
            H4, H17 = collect(m4, m17, calib, sorted(ml)); multi = {}
            for i, l in enumerate(sorted(ml)):
                r = torch.load(ridge_path(l)); top = sorted(range(len(r["r2_all"])), key=lambda j: -r["r2_all"][j])[:a.multi_src]; top.sort()
                W, b, r2 = ridge_fit(torch.cat([H4[j] for j in top], -1), H17[i]); multi[l] = LinMap(top, W, b)
                print(f"  layer {l:2d}: concat of 4B layers {top}, R2 {r2:.3f} (single {r['r2']:.3f})", flush=True)
            del H4, H17
        for l in range(Lc):
            if l in ml: maps.append(multi[l]); continue
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
    qlens = tuple(int(x) for x in a.qa_lengths.split(","))
    if a.corpus == "hotpot": qa = load_qa_examples(tok, 4000, seed=777, lengths=qlens)
    elif a.corpus == "qamix": import qa_data; qa = qa_data.train_mix(tok, 4000, 777, qlens)
    else: qa = None
    tr = {P: load_corpus_chunks(tok, "wt103" if qa else a.corpus, N_TRAIN[P], P + CONT, seed=777) for P in lens}

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
        if qa and step % 2 == 0:                                     # alternate: QA example, wikitext chunk
            ids, P = qa[torch.randint(0, len(qa), (1,)).item()]; ids = ids.to(DEV)
        else:
            P = lens[torch.randint(0, len(lens), (1,)).item()]
            ids = tr[P][torch.randint(0, len(tr[P]), (1,)).item()].unsqueeze(0).to(DEV)
        if a.teacher == "attn": loss = pipeline_attn(m4, m17, maps, ids, P, Lc, a.source, a.prompt_w or 1.0, prompt_kl=(P <= 2048), attn_w=a.attn_w, prompt_attn=a.prompt_attn, pattn_pos=a.pattn_pos)
        elif a.teacher: loss = pipeline_kl(m4, m17, maps, ids, P, Lc, a.source, a.prompt_w or 1.0)
        else: loss = pipeline_nll(m4, m17, maps, ids, P, Lc, a.source, a.first_tok, a.prompt_w)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(maps.parameters(), 1.0); opt.step(); sched.step()
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if step % 250 == 0:
            print(f"  [{step:4d}] lm ema={run:.4f} train-ppl~{math.exp(run):6.2f}  {time.time()-t0:.0f}s", flush=True)
    rF = full_eval("final") if a.steps else r0
    print(f"\n[{a.tag}] RESULT cut={Lc} source={a.source} cut_src={a.cut_src} cut_mlp={a.cut_mlp} attn={a.attn_heads}x{a.attn_dh} first_tok={a.first_tok} prompt_w={a.prompt_w} teacher={a.teacher or 'none'} seed={a.seed} steps={a.steps}  " +
          "  ".join(f"P{P}: {r0[P]['ppl']:.3f}->{rF[P]['ppl']:.3f}" for P in elens), flush=True)
    json.dump({"tag": a.tag, "cut": Lc, "source": a.source, "spec": maps.spec(), "params_M": n_par / 1e6,
               "steps": a.steps, "seed": a.seed, "corpus": a.corpus, "train_lengths": lens,
               "cut_src": a.cut_src, "multi_src": a.multi_src, "multi_layers": a.multi_layers, "cut_mlp": a.cut_mlp, "attn": [a.attn_heads, a.attn_dh], "attn_lr": a.attn_lr, "first_tok": a.first_tok, "prompt_w": a.prompt_w, "teacher": a.teacher, "attn_w": a.attn_w, "prompt_attn": a.prompt_attn,
               "step0": {P: r0[P]["ppl"] for P in elens}, "final": {P: rF[P]["ppl"] for P in elens},
               "nll_final": {P: rF[P]["nll"] for P in elens}},
              open(f"{C.RESULTS}/hv2_{a.tag}.json", "w"), indent=2)
    torch.save({"maps": maps.state_dict(), "spec": maps.spec(), "cut": Lc, "source": a.source}, f"{OUT}/reskv_{a.tag}.pt")


if __name__ == "__main__":
    main()
