"""Long-context handoff evaluation: multi-document QA at 2k / 4k / 8k tokens.

The setting: the 4B's reading of the context already exists (a previous turn, a shared prefix,
another agent), and the 1.7B wants to answer without re-reading it.  So the baseline is the
1.7B re-prefilling the whole context, and the pipeline's cost is the bridge alone.

Examples are HotpotQA distractor questions (two gold paragraphs among ten), padded with
paragraphs from other questions until the prompt reaches the target length, gold paragraphs
at random positions.  Short greedy answers, scored by SQuAD-style exact match and token F1.
Per-example scores and per-condition timings (4B prefill, bridge, 1.7B re-prefill) are saved.

  python src/longctx_eval.py --conds 17b,4b,reskv:pw_4b_c28@4b,reskv:pw_self_c28 --lengths 2048,4096,8192 --n 200 --tag handoff
  python src/longctx_eval.py --dataset squad --conds 17b,reskv:qamix_4b_c28@4b --n 500 --tag squad_ood
"""
import sys, json, re, string, time, random, warnings, argparse, collections, torch
warnings.filterwarnings("ignore")
import common as C
from residual_cut import load_sdpa, DEV
from task_eval import Cond

HEADER = ("Answer the question using the documents below. Reply with a short answer only, "
          "as in these examples.\nQuestion: Which city is the capital of France?\nAnswer: Paris\n"
          "Question: In what year did the Second World War end?\nAnswer: 1945\n"
          "Question: Who wrote the novel Pride and Prejudice?\nAnswer: Jane Austen\n\n")


# ------------------------------------------------------------------ data

def build_examples(tok, n, lengths, seed):
    """One prompt per (example, target length).  Returns {L: [(ids, answer, n_docs)]}."""
    from datasets import load_dataset
    ds = list(load_dataset("hotpotqa/hotpot_qa", "distractor")["validation"])
    rng = random.Random(seed); rng.shuffle(ds)
    cache = {}
    def ntok(text):
        if text not in cache: cache[text] = len(tok(text, add_special_tokens=False).input_ids)
        return cache[text]
    def doc(title, sents): return f"Document ({title}): {''.join(sents).strip()}\n\n"
    pool = [doc(t, s) for ex in ds[n:n + 2000] for t, s in zip(ex["context"]["title"], ex["context"]["sentences"])]
    out = {L: [] for L in lengths}
    for ex in ds[:n]:
        gold = set(ex["supporting_facts"]["title"])
        own = [(t in gold, doc(t, s)) for t, s in zip(ex["context"]["title"], ex["context"]["sentences"])]
        q = f"Question: {ex['question']}\nAnswer:"
        base = ntok(HEADER) + ntok(q) + sum(ntok(d) for _, d in own)
        prng = random.Random(seed + hash(ex["id"]) % 10000)
        for L in lengths:
            docs = list(own); budget = L - base - 8
            extra = prng.sample(pool, 400)
            for d in extra:
                if ntok(d) > budget: continue
                docs.append((False, d)); budget -= ntok(d)
                if budget < 40: break
            prng.shuffle(docs)
            text = HEADER + "".join(d for _, d in docs) + q
            ids = tok(text, return_tensors="pt").input_ids
            out[L].append((ids, ex["answer"], len(docs)))
    return out


# ----------------------------------------------------------------- scoring

def normalize(s):
    s = s.lower(); s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())

def em_f1(pred, gold):
    """(exact match, token F1, contains): contains = the normalised gold string occurs in the answer."""
    p, g = normalize(pred), normalize(gold)
    em = float(p == g); has = float(len(g) > 0 and g in p)
    pt, gt = p.split(), g.split()
    common = collections.Counter(pt) & collections.Counter(gt); ns = sum(common.values())
    if ns == 0: return em, 0.0, has
    prec, rec = ns / len(pt), ns / len(gt)
    return em, 2 * prec * rec / (prec + rec), has


# -------------------------------------------------------------- generation

def sync(): torch.cuda.synchronize()

@torch.no_grad()
def prefill_timed(cond, ids):
    """(cache, logits, timings).  For the pipeline the 4B forward and the bridge are timed apart,
    because in the handoff setting only the bridge is on the critical path."""
    if cond.reskv and cond.first4b:
        R, maps, Lc, source = cond.reskv
        box = {}; hk = cond.m4.model.norm.register_forward_hook(lambda m, i, o: box.__setitem__("pre", i[0]))
        sync(); t0 = time.perf_counter()
        try: o4 = cond.m4(input_ids=ids, output_hidden_states=True, use_cache=False)
        finally: hk.remove()
        sync(); t1 = time.perf_counter()
        H = tuple(o4.hidden_states[:-1]) + (box["pre"],)
        cache, _ = R.prefill(cond.m4, cond.m17, maps, ids, Lc, source, H=H)
        sync(); t2 = time.perf_counter()
        return cache, o4.logits, {"prefill_4b": t1 - t0, "bridge": t2 - t1}
    sync(); t0 = time.perf_counter()
    cache, lg = cond.prefill(ids)
    sync(); t1 = time.perf_counter()
    return cache, lg, {"prefill": t1 - t0}

_BAD = {}
def bad_first(tok):
    """Token ids that must not start an answer: newline-only strings and end-of-text.  Qwen3-4B often
    puts a newline first after "Answer:", which makes the answer empty under a newline stop.  A lone
    space is allowed: it is how the 1.7B starts numeric answers ("Answer:", " ", "2002")."""
    if tok not in _BAD:
        ids = [i for i in range(len(tok)) if (lambda d: d.strip() == "" and ("\n" in d or "\r" in d))(tok.decode([i]))] + [tok.eos_token_id]
        _BAD[tok] = torch.tensor(sorted(set(i for i in ids if i is not None)), device=DEV)
    return _BAD[tok]

@torch.no_grad()
def generate(cond, tok, ids, max_new=16):
    cache, lg, tm = prefill_timed(cond, ids)
    T = ids.shape[1]; first = lg[0, -1].float().clone(); first[bad_first(tok)] = -1e9
    nxt = first.argmax().item(); outs = [nxt]
    for i in range(max_new - 1):
        txt = tok.decode(outs)
        if "\n" in txt or txt.rstrip().endswith("."): break
        pos = torch.tensor([[T + i]], device=DEV)
        o = cond.dec(input_ids=torch.tensor([[nxt]], device=DEV), past_key_values=cache, position_ids=pos, cache_position=pos[0], use_cache=True)
        cache = o.past_key_values; nxt = o.logits[0, -1].argmax().item(); outs.append(nxt)
    return tok.decode(outs).split("\n")[0].strip().rstrip("."), tm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default="17b,4b,reskv:pw_4b_c28@4b,reskv:pw_self_c28")
    ap.add_argument("--lengths", default="2048,4096,8192"); ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", required=True)
    ap.add_argument("--dataset", default="hotpot", help="hotpot | squad | duorc (held out, never trained on)")
    a = ap.parse_args()
    m4, tok = load_sdpa(C.PREFILL_MODEL); m17, _ = load_sdpa(C.DECODE_MODEL)
    lengths = [int(x) for x in a.lengths.split(",")]
    if a.dataset == "hotpot": data = build_examples(tok, a.n, lengths, a.seed)
    else: import qa_data; data = qa_data.eval_set(tok, a.dataset, a.n, lengths, a.seed)
    for L in lengths:
        lens = [ids.shape[1] for ids, _, _ in data[L]]
        print(f"[L={L}] {len(data[L])} prompts, tokens {min(lens)}-{max(lens)} (mean {sum(lens)/len(lens):.0f})", flush=True)
    conds = [Cond(c, m4, m17, tok) for c in a.conds.split(",")]
    res = {}
    for L in lengths:
        res[L] = {}
        for c in conds:
            em = f1 = has = 0.0; per = []; preds = []; tms = collections.defaultdict(float); t0 = time.time()
            for ids, ans, nd in data[L]:
                pred, tm = generate(c, tok, ids.to(DEV))
                e, f, h = max((em_f1(pred, g) for g in (ans if isinstance(ans, list) else [ans])), key=lambda x: (x[0], x[1])); em += e; f1 += f; has += h; per.append([e, f, h]); preds.append(pred)
                for k, v in tm.items(): tms[k] += v
            N = len(data[L])
            r = {"em": em / N, "f1": f1 / N, "contains": has / N, "n": N, "per": per, "preds": preds,
                 "ms": {k: 1000 * v / N for k, v in tms.items()}, "sec": round(time.time() - t0)}
            res[L][c.name] = r
            print(f"  L={L:5d} {c.name:22s} EM {r['em']:.3f}  F1 {r['f1']:.3f}  contains {r['contains']:.3f}  " + "  ".join(f"{k} {v:.0f}ms" for k, v in r["ms"].items()) + f"  ({r['sec']}s)", flush=True)
            json.dump(res, open(f"{C.RESULTS}/longctx_{a.tag}.json", "w"), indent=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
