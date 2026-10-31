"""QA sources in one format, for training the stitch on a mix and for held-out evaluation.

A record is (docs, turns, golds): docs are (title, text) passages that must appear in the prompt,
turns are (question, answer) pairs ending with the scored question (answer None), golds are the
accepted answers for it.  build() lays each record out in the longctx_eval prompt format: header,
the record's documents mixed with distractor passages from the same source up to a target length,
then the question turns.

Training mix: hotpot, drop, musique, coqa.  Held out: squad, duorc (never used for training).
"""
import random
import longctx_eval as LC

SOURCES = {}
def source(fn): SOURCES[fn.__name__] = fn; return fn


@source
def hotpot(split):
    from datasets import load_dataset
    for ex in load_dataset("hotpotqa/hotpot_qa", "distractor")[split]:
        docs = [(t, "".join(s)) for t, s in zip(ex["context"]["title"], ex["context"]["sentences"])]
        yield docs, [(ex["question"], None)], [ex["answer"]]

@source
def drop(split):
    from datasets import load_dataset
    for ex in load_dataset("ucinlp/drop")[split]:
        spans = ex["answers_spans"]["spans"]
        if len(spans) == 1 and len(spans[0]) < 60: yield [(None, ex["passage"])], [(ex["question"], None)], spans

@source
def musique(split):
    from datasets import load_dataset
    for ex in load_dataset("dgslibisey/MuSiQue")[split]:
        if not ex["answerable"]: continue
        docs = [(p["title"], p["paragraph_text"]) for p in ex["paragraphs"]]
        yield docs, [(ex["question"], None)], [ex["answer"]] + list(ex["answer_aliases"])

@source
def coqa(split, max_turns=4):
    """Conversational: one to four earlier question and answer turns precede the scored one."""
    from datasets import load_dataset
    rng = random.Random(1)
    for ex in load_dataset("stanfordnlp/coqa")[split]:
        qs, ans = ex["questions"], ex["answers"]["input_text"]
        k = rng.randint(1, min(max_turns, len(qs)))
        if any(len(a) > 60 or a.lower() == "unknown" for a in ans[:k]): continue
        turns = [(q, a) for q, a in zip(qs[:k - 1], ans[:k - 1])] + [(qs[k - 1], None)]
        yield [(None, ex["story"])], turns, [ans[k - 1]]

@source
def squad(split):
    from datasets import load_dataset
    for ex in load_dataset("rajpurkar/squad")[split]:
        yield [(None, ex["context"])], [(ex["question"], None)], list(ex["answers"]["text"])

@source
def duorc(split):
    from datasets import load_dataset
    for ex in load_dataset("ibm-research/duorc", "SelfRC")[split]:
        if ex["no_answer"] or not ex["answers"]: continue
        yield [(ex["title"], ex["plot"])], [(ex["question"], None)], list(ex["answers"])


@source
def triviaqa(split, window=3000):
    """TriviaQA rc.wikipedia: trivia questions with the Wikipedia pages of the entities involved.  Each
    page is cut to a window of about 3,000 characters around the first answer alias it contains
    (pages run to 30k characters); questions whose pages contain no alias are skipped.  Held out."""
    from datasets import load_dataset
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", split=split)
    for ex in ds:
        aliases = [ex["answer"]["value"]] + list(ex["answer"]["aliases"])
        if any(len(a) > 60 for a in aliases[:1]): continue
        docs = []
        for title, page in zip(ex["entity_pages"]["title"], ex["entity_pages"]["wiki_context"]):
            low = page.lower(); pos = min((low.find(a.lower()) for a in aliases if len(a) > 2 and low.find(a.lower()) >= 0), default=-1)
            if pos < 0: continue
            lo = max(0, pos - window // 2); docs.append((title, page[lo:lo + window]))
        if docs: yield docs[:2], [(ex["question"], None)], aliases


def fmt(title, text):
    text = " ".join(text.split())
    return f"Document ({title}): {text}\n\n" if title else f"Document: {text}\n\n"


def load(name, split, n, seed, pool_n=3000):
    """n shuffled records of a source plus a pool of distractor passages from other records."""
    recs = list(SOURCES[name](split))
    rng = random.Random(seed); rng.shuffle(recs)
    pool = [fmt(t, x) for r in recs[n:n + pool_n] for t, x in r[0]]
    return recs[:n], pool


def build(tok, recs, pool, lengths, seed, every_length=False, header=LC.HEADER):
    """Prompt texts.  every_length: {L: [(text, golds)]} with one prompt per record and length
    (evaluation); otherwise [(text, golds)] with a random length per record (training)."""
    cache = {}
    def ntok(t):
        if t not in cache: cache[t] = len(tok(t, add_special_tokens=False).input_ids)
        return cache[t]
    rng = random.Random(seed)
    out = {L: [] for L in lengths} if every_length else []
    for i, (docs, turns, golds) in enumerate(recs):
        own = [fmt(t, x) for t, x in docs]
        q = "".join(f"Question: {q}\nAnswer: {a}\n" for q, a in turns[:-1]) + f"Question: {turns[-1][0]}\nAnswer:"
        base = ntok(header) + ntok(q) + sum(ntok(d) for d in own)
        prng = random.Random(seed + i)
        for L in (lengths if every_length else [rng.choice(lengths)]):
            ds = list(own); budget = L - base - 8
            for d in prng.sample(pool, min(300, len(pool))):
                if ntok(d) > budget: continue
                ds.append(d); budget -= ntok(d)
                if budget < 40: break
            prng.shuffle(ds)
            text = header + "".join(ds) + q
            (out[L] if every_length else out).append((text, golds))
    return out


def train_mix(tok, n, seed, lengths, names=("hotpot", "drop", "musique", "coqa")):
    """[(ids, P)] over the mix, n records in total, loss on the answer tokens after position P."""
    import torch
    out = []
    for k, name in enumerate(names):
        recs, pool = load(name, "train", n // len(names), seed + k)
        for text, golds in build(tok, recs, pool, lengths, seed + k):
            pid = tok(text, return_tensors="pt").input_ids
            aid = tok(" " + golds[0].strip() + "\n", return_tensors="pt", add_special_tokens=False).input_ids
            if aid.shape[1] < 2: continue
            out.append((torch.cat([pid, aid], 1), pid.shape[1]))
        print(f"  {name}: {len(recs)} records", flush=True)
    random.Random(seed).shuffle(out)
    print(f"  qa mix: {len(out)} examples, prompt tokens {min(p for _, p in out)}-{max(p for _, p in out)}", flush=True)
    return out


def eval_set(tok, name, n, lengths, seed):
    """{L: [(ids, golds, n_docs)]} from the validation split, same layout as longctx_eval."""
    recs, pool = load(name, "validation", n, seed)
    built = build(tok, recs, pool, lengths, seed, every_length=True)
    return {L: [(tok(t, return_tensors="pt").input_ids, g, t.count("Document")) for t, g in built[L]] for L in lengths}
