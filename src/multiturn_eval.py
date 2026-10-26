"""Multi-turn handoff on SQuAD: the 4B answers the first question on a context, its answer
becomes part of the conversation, and the 1.7B answers a second question about the same
context without re-reading it.

The cache the 1.7B continues from is the 4B's own: context, question 1, the 4B's generated
answer 1, question 2.  Baseline: the 1.7B re-prefills that whole text.  SQuAD paragraphs
with two or more questions are padded with other SQuAD paragraphs to the target length, gold
paragraph at a random position.  SQuAD is also out of domain for a stitch trained on HotpotQA.
Scored on question 2 by exact match and token F1 (max over the gold answers).

  python src/multiturn_eval.py --conds 17b,4b,reskv:attnqa_4b_c28@4b,reskv:attnqa_4b_c28,reskv:attnqa_self_c28 --n 500 --tag mt_attnqa
"""
import sys, json, time, random, warnings, argparse, collections, torch
warnings.filterwarnings("ignore")
import common as C
from residual_cut import load_sdpa, DEV
from task_eval import Cond
import longctx_eval as LC

HEADER = ("Answer each question using the documents below. Reply with a short answer only, "
          "as in these examples.\nQuestion: Which city is the capital of France?\nAnswer: Paris\n"
          "Question: In what year did the Second World War end?\nAnswer: 1945\n\n")


def build_examples(tok, n, lengths, seed):
    """{L: [(prefix_text, q1, q2, golds2)]}: prefix ends with 'Question: q1\\nAnswer:'."""
    from datasets import load_dataset
    ds = list(load_dataset("rajpurkar/squad")["validation"])
    by_ctx = collections.defaultdict(list)
    for ex in ds: by_ctx[ex["context"]].append(ex)
    ctxs = [c for c, exs in by_ctx.items() if len(exs) >= 2]
    rng = random.Random(seed); rng.shuffle(ctxs)
    cache = {}
    def ntok(t):
        if t not in cache: cache[t] = len(tok(t, add_special_tokens=False).input_ids)
        return cache[t]
    def doc(i, c): return f"Document {i}: {c.strip()}\n\n"
    pool = ctxs[n:n + 1500]
    out = {L: [] for L in lengths}
    for c in ctxs[:n]:
        q1, q2 = rng.sample(by_ctx[c], 2)
        prng = random.Random(seed + hash(c) % 10000)
        for L in lengths:
            docs = [c]; budget = L - ntok(HEADER) - ntok(c) - 220
            for d in prng.sample(pool, 200):
                if ntok(d) > budget: continue
                docs.append(d); budget -= ntok(d)
                if budget < 60: break
            prng.shuffle(docs)
            prefix = HEADER + "".join(doc(i + 1, d) for i, d in enumerate(docs)) + f"Question: {q1['question']}\nAnswer:"
            out[L].append((prefix, q1, q2))
    return out


@torch.no_grad()
def turn1_answers(m4, tok, data):
    """The 4B's own greedy answer to question 1, for every example and length."""
    c4 = Cond("4b", None, m4, tok); c4.dec = m4
    ans = {}
    for L, exs in data.items():
        ans[L] = []
        for prefix, q1, q2 in exs:
            ids = tok(prefix, return_tensors="pt").input_ids.to(DEV)
            pred, _ = LC.generate(c4, tok, ids)
            ans[L].append(pred if pred else q1["answers"]["text"][0])
    return ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default="17b,4b,reskv:attnqa_4b_c28@4b,reskv:attnqa_4b_c28,reskv:attnqa_self_c28")
    ap.add_argument("--lengths", default="2048,4096,8192"); ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    m4, tok = load_sdpa(C.PREFILL_MODEL); m17, _ = load_sdpa(C.DECODE_MODEL)
    lengths = [int(x) for x in a.lengths.split(",")]
    data = build_examples(tok, a.n, lengths, a.seed)
    t0 = time.time(); a1 = turn1_answers(m4, tok, data); print(f"turn-1 answers by the 4B: {time.time()-t0:.0f}s", flush=True)
    full = {L: [] for L in lengths}
    for L in lengths:
        for (prefix, q1, q2), ans1 in zip(data[L], a1[L]):
            text = prefix + " " + ans1 + f"\nQuestion: {q2['question']}\nAnswer:"
            full[L].append((tok(text, return_tensors="pt").input_ids, q2["answers"]["text"]))
        lens = [ids.shape[1] for ids, _ in full[L]]
        print(f"[L={L}] {len(full[L])} prompts, tokens {min(lens)}-{max(lens)} (mean {sum(lens)/len(lens):.0f})", flush=True)
    conds = [Cond(c, m4, m17, tok) for c in a.conds.split(",")]
    res = {}
    for L in lengths:
        res[L] = {}
        for c in conds:
            em = f1 = has = 0.0; per = []; preds = []; tms = collections.defaultdict(float); t0 = time.time()
            for ids, golds in full[L]:
                pred, tm = LC.generate(c, tok, ids.to(DEV))
                e, f, h = max((LC.em_f1(pred, g) for g in golds), key=lambda x: (x[0], x[1]))
                em += e; f1 += f; has += h; per.append([e, f, h]); preds.append(pred)
                for k, v in tm.items(): tms[k] += v
            N = len(full[L])
            r = {"em": em / N, "f1": f1 / N, "contains": has / N, "n": N, "per": per, "preds": preds,
                 "ms": {k: 1000 * v / N for k, v in tms.items()}, "sec": round(time.time() - t0)}
            res[L][c.name] = r
            print(f"  L={L:5d} {c.name:24s} EM {r['em']:.3f}  F1 {r['f1']:.3f}  contains {r['contains']:.3f}  " + "  ".join(f"{k} {v:.0f}ms" for k, v in r["ms"].items()) + f"  ({r['sec']}s)", flush=True)
            json.dump(res, open(f"{C.RESULTS}/longctx_{a.tag}.json", "w"), indent=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
