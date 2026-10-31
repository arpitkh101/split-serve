"""longctx_eval for 16k and 32k prompts, same flags.  Three changes so the evaluation fits in 24 GB: model
forwards keep only the last position's logits (the harness reads nothing else), both models run flash
attention (sdpa materialises a T x T causal mask, 2 GiB at 32k), and the stitched conditions collect only
the source-model residuals the maps read, park them on the CPU, and bring each to the GPU for the map that
uses it.  Quality is otherwise the harness's; the bf16 kernel change moves greedy answers by 2 to 3 EM at
32k with n=200, so always compare conditions run in the same process.

  python src/longctx_long.py --conds 17b,reskv:qamix_4b_c28@4b,reskv:qamix_self_c28 --lengths 32768 --n 200 --tag handoff_qamix_long32
"""
import sys, time, torch
import residual_cut as RC


def load_flash(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(RC.DEV).eval()
    for q in m.parameters(): q.requires_grad_(False)
    f = m.forward
    def fwd(*a, **k):
        k.setdefault("logits_to_keep", 1); return f(*a, **k)
    m.forward = fwd
    return m, AutoTokenizer.from_pretrained(name)
RC.load_sdpa = load_flash                  # before longctx_eval binds it
import longctx_eval as LC, task_eval as TE, residual_stitch as R
from residual_cut import DEV
from transformers import DynamicCache


@torch.no_grad()
def collect_needed(model, ids, need, want_logits):
    """Residuals entering layers j in need (index n_layers = pre-norm final residual), as a list with
    None elsewhere; plus the last-position logits when want_logits."""
    n = model.config.num_hidden_layers; H = [None] * (n + 1); hooks = []
    for j in need:
        if j < n: hooks.append(model.model.layers[j].register_forward_pre_hook(lambda m, a, j=j: H.__setitem__(j, a[0].cpu())))
        else: hooks.append(model.model.norm.register_forward_hook(lambda m, i, o: H.__setitem__(n, i[0].cpu())))
    try: o = model(input_ids=ids, use_cache=False, logits_to_keep=1)
    finally:
        for h in hooks: h.remove()
    return H, (o.logits if want_logits else None)


@torch.no_grad()
def prefill_free(m17, maps, H, Lc, T):
    """R.prefill without the native tail (cut == n_layers); the source residuals live on the CPU and
    each map's inputs are brought to the GPU only while that map runs."""
    cache = DynamicCache(); pos = torch.arange(T, device=DEV).unsqueeze(0)
    pe = m17.model.rotary_emb(torch.zeros(1, T, 1, device=DEV, dtype=m17.dtype), pos)
    def on_gpu(l):
        Hl = [None] * len(H)
        for j in maps.maps[l].src: Hl[j] = H[j].to(DEV, non_blocking=True)
        return Hl
    for l in range(Lc):
        k, v = R.native_kv(m17, maps(on_gpu(l), l).unsqueeze(0).to(m17.dtype), l, pe); cache.update(k, v, l, {})
    h = maps(on_gpu(Lc), Lc).unsqueeze(0).to(m17.dtype)
    return cache, m17.lm_head(m17.model.norm(h[:, -1:]))


orig_prefill = TE.Cond.prefill
@torch.no_grad()
def prefill(self, ids):
    if not self.reskv: return orig_prefill(self, ids)
    Rm, maps, Lc, source = self.reskv; T = ids.shape[1]
    assert Lc == self.m17.config.num_hidden_layers and self.suffix == 0 and not self.first17b, "32k wrapper: cut 28, no suffix/oracle"
    src = self.m4 if source == "4b" else self.m17
    need = sorted({j for m in maps.maps for j in m.src})
    H, lg4 = collect_needed(src, ids, need, self.first4b)
    cache, lg = prefill_free(self.m17, maps, H, Lc, T)
    return cache, (lg4 if self.first4b else lg)
TE.Cond.prefill = prefill



def prefill_timed(cond, ids):
    """The harness's timed path calls the 4B with all hidden states kept; route the stitched conditions
    through the memory-fixed prefill instead, timing the source forward and the bridge apart."""
    if not cond.reskv: return orig_timed(cond, ids)
    Rm, maps, Lc, source = cond.reskv; T = ids.shape[1]
    src = cond.m4 if source == "4b" else cond.m17; need = sorted({j for m in maps.maps for j in m.src})
    LC.sync(); t0 = time.perf_counter()
    H, lg4 = collect_needed(src, ids, need, cond.first4b)
    LC.sync(); t1 = time.perf_counter()
    cache, lg = prefill_free(cond.m17, maps, H, Lc, T)
    LC.sync(); t2 = time.perf_counter()
    return cache, (lg4 if cond.first4b else lg), ({"prefill_4b": t1 - t0, "bridge": t2 - t1} if cond.first4b else {"prefill": t2 - t0})
orig_timed = LC.prefill_timed; LC.prefill_timed = prefill_timed

if __name__ == "__main__":
    LC.main()
