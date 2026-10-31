"""Layer patching, no training: on held-out SQuAD questions the stitched pipeline gets wrong and the
1.7B re-prefill gets right, swap the native 1.7B keys and values into the stitched cache one block
of four layers at a time and count how many answers recover.  The first decode token comes from
the 4B, as in the @4b condition, so the first-token effect is held fixed.  The reverse rows patch
every block except one.  Writes out/layer_patch_squad4k.json.

  CUDA_VISIBLE_DEVICES=5 python3 src/layer_patch.py --n 100 --length 4096
"""
import sys, json, time, warnings, argparse, torch
warnings.filterwarnings("ignore")
import common as C
from residual_cut import load_sdpa, DEV
from task_eval import Cond
import longctx_eval as LC
import qa_data
from transformers import DynamicCache


def kv_lists(cache):
    """(keys, values) lists of per-layer tensors for either DynamicCache layout."""
    if hasattr(cache, "layers"): return [l.keys for l in cache.layers], [l.values for l in cache.layers]
    return list(cache.key_cache), list(cache.value_cache)


def make_cache(ks, vs):
    return DynamicCache.from_legacy_cache(tuple((k, v) for k, v in zip(ks, vs)))


@torch.no_grad()
def gen(dec, tok, cache, lg, T, max_new=16):
    """Greedy decode from a cache and the first-token logits, same stopping rules as longctx_eval."""
    first = lg[0, -1].float().clone(); first[LC.bad_first(tok)] = -1e9
    nxt = first.argmax().item(); outs = [nxt]
    for i in range(max_new - 1):
        txt = tok.decode(outs)
        if "\n" in txt or txt.rstrip().endswith("."): break
        pos = torch.tensor([[T + i]], device=DEV)
        o = dec(input_ids=torch.tensor([[nxt]], device=DEV), past_key_values=cache, position_ids=pos, cache_position=pos[0], use_cache=True)
        cache = o.past_key_values; nxt = o.logits[0, -1].argmax().item(); outs.append(nxt)
    return tok.decode(outs).split("\n")[0].strip().rstrip(".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--tag", default="qamix_4b_c28"); ap.add_argument("--block", type=int, default=4)
    a = ap.parse_args()
    m4, tok = load_sdpa(C.PREFILL_MODEL); m17, _ = load_sdpa(C.DECODE_MODEL)
    prev = json.load(open(f"{C.RESULTS}/longctx_squad_ood.json"))[str(a.length)]
    ok17, okst = prev["17b"]["per"], prev[f"reskv:{a.tag}@4b"]["per"]
    idx = [i for i in range(len(ok17)) if ok17[i][0] == 1 and okst[i][0] == 0][:a.n]
    print(f"{len(idx)} examples at {a.length}: re-prefill right, stitched pipeline wrong", flush=True)
    data = qa_data.eval_set(tok, "squad", 500, (a.length,), 0)[a.length]
    cond = Cond(f"reskv:{a.tag}@4b", m4, m17, tok)
    nl = m17.config.num_hidden_layers; blocks = [list(range(s, min(s + a.block, nl))) for s in range(0, nl, a.block)]
    rows = {"stitched": [], "native_kv_4b_first": [], "native": []}
    for b in blocks: rows[f"patch_{b[0]}-{b[-1]}"] = []; rows[f"keep_{b[0]}-{b[-1]}"] = []
    t0 = time.time()
    for n, i in enumerate(idx):
        ids, golds, _ = data[i]; ids = ids.to(DEV); T = ids.shape[1]
        em = lambda pred: max(LC.em_f1(pred, g)[0] for g in golds)
        scache, lg4 = cond.prefill(ids); sk, sv = kv_lists(scache)
        o = m17(input_ids=ids, use_cache=True); nk, nv = kv_lists(o.past_key_values)
        rows["stitched"].append(em(gen(m17, tok, make_cache(sk, sv), lg4, T)))
        rows["native_kv_4b_first"].append(em(gen(m17, tok, make_cache(nk, nv), lg4, T)))
        rows["native"].append(em(gen(m17, tok, make_cache(nk, nv), o.logits, T)))
        for b in blocks:
            ks = [nk[l] if l in b else sk[l] for l in range(nl)]; vs = [nv[l] if l in b else sv[l] for l in range(nl)]
            rows[f"patch_{b[0]}-{b[-1]}"].append(em(gen(m17, tok, make_cache(ks, vs), lg4, T)))
            ks = [sk[l] if l in b else nk[l] for l in range(nl)]; vs = [sv[l] if l in b else nv[l] for l in range(nl)]
            rows[f"keep_{b[0]}-{b[-1]}"].append(em(gen(m17, tok, make_cache(ks, vs), lg4, T)))
        del scache, o, sk, sv, nk, nv; torch.cuda.empty_cache()
        if n % 10 == 9: print(f"  [{n+1}] {time.time()-t0:.0f}s " + " ".join(f"{k}={sum(v)}" for k, v in rows.items() if k.startswith(('stitched','native','patch'))), flush=True)
    N = len(idx)
    print(f"\nEM recovered out of {N} (stitched cache, 4B first token, native K/V swapped in per block):")
    for k, v in rows.items(): print(f"  {k:22s} {sum(v):4.0f} / {N}")
    json.dump({"length": a.length, "tag": a.tag, "idx": idx, "rows": rows}, open(f"{C.RESULTS}/layer_patch_squad4k.json", "w"), indent=1)


if __name__ == "__main__":
    main()
