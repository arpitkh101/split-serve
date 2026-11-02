"""QA sources in one format, for training the stitch on a mix and for held-out evaluation.

A record is (docs, turns, golds): docs are (title, text) passages that must appear in the prompt,
turns are (question, answer) pairs ending with the scored question (answer None), golds are the
accepted answers for it.  build() lays each record out in the longctx_eval prompt format: header,
the record's documents mixed with distractor passages from the same source up to a target length,
then the question turns.

Training mix: hotpot, drop, musique, coqa.  Held out: squad, duorc (never used for training).
copy: a synthetic source whose answers are exact strings from the context (robot-log state
tracking with its own vocabularies, and "which command was at step N" recall over a long list
of formal commands); see copy_mix.  statetrack2 and alfworld stay held out.
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


def train_mix(tok, n, seed, lengths, names=("hotpot", "drop", "musique", "coqa"), copy_n=0):
    """[(ids, P)] over the mix, n records in total, loss on the answer tokens after position P.
    copy_n: that many extra examples from the synthetic copy source (copy_mix), golds 3 to 10 tokens."""
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
    if copy_n:
        shown = set()
        for text, golds, kind in copy_mix(tok, copy_n, seed + 1000, lengths):
            pid = tok(text, return_tensors="pt").input_ids
            aid = tok(" " + golds[0].strip() + "\n", return_tensors="pt", add_special_tokens=False).input_ids
            out.append((torch.cat([pid, aid], 1), pid.shape[1]))
            if kind not in shown:                                    # one decoded example per kind, to eyeball
                shown.add(kind); dec = tok.decode(pid[0, -90:]); gold = tok.decode(aid[0])
                print(f"  copy example ({kind}, {pid.shape[1]} prompt tokens, {aid.shape[1]} gold tokens)\n"
                      f"    prompt tail: ...{dec!r}\n    gold: {gold!r}", flush=True)
        print(f"  copy: {copy_n} examples", flush=True)
    random.Random(seed).shuffle(out)
    print(f"  qa mix: {len(out)} examples, prompt tokens {min(p for _, p in out)}-{max(p for _, p in out)}", flush=True)
    return out


# ---------------------------------------------------------------- copy source: exact-string answers

# Robot-log state tracking in the manner of statetrack.make2, but with its own rooms, surfaces, objects,
# sentence templates, filler and header, so the statetrack2 evaluation stays held out.  Every move
# template contains the literal phrase "on the {s} in the {r2}", which is the gold.
CP_ROOMS = ["atrium", "cellar", "nursery", "library", "porch", "attic", "greenhouse", "laundry room",
            "dining room", "foyer", "mudroom", "sunroom"]
CP_SURFACES = ["ledge", "trolley", "workbench", "windowsill", "dresser", "sideboard", "ottoman", "nightstand",
               "pedestal", "footstool", "mantel", "lid of the chest"]
CP_OBJECTS = ["colander", "whisk", "spatula", "notepad", "harmonica", "lantern", "compass", "trophy", "binder",
              "canteen", "teacup", "saucer", "pitcher", "mallet", "chisel", "pliers", "funnel", "tongs", "grater",
              "crayon box", "jump rope", "sun hat", "hairbrush", "nail file", "shoehorn", "coaster", "napkin ring",
              "corkscrew", "egg timer", "oven mitt", "tea strainer", "pepper mill", "bike pump", "tape measure",
              "glue stick", "paperweight", "hole punch", "desk fan", "puzzle box", "rain boot", "dog leash",
              "bird feeder", "garden trowel", "flower pot", "seed packet", "magnifying glass", "chess set",
              "yo-yo", "kite", "frisbee", "snow globe", "music box", "photo album", "wall clock", "doormat",
              "coat hook", "bath sponge", "soap dish", "lint roller", "night light"]
CP_TEMPLATES = [
    "Next the robot lifts the {o} from the {r1}, carries it over, and leaves it on the {s} in the {r2}.",
    "The {o} goes from the {r1} to a new spot: on the {s} in the {r2}.",
    "Relocating the {o}: out of the {r1}, down on the {s} in the {r2}.",
    "The robot grabs the {o} in the {r1} and parks it on the {s} in the {r2}.",
    "Having fetched the {o} from the {r1}, the robot drops it on the {s} in the {r2}.",
    "The {o} leaves the {r1} and is now on the {s} in the {r2}.",
]
CP_FILLER = [
    "The robot tops up its water tank in the {r}.",
    "Nothing to move in the {r}; the robot logs a scan of the {s}.",
    "The robot waits by the {s} in the {r} for a minute.",
    "A quick dusting of the {s} in the {r}.",
    "The robot re-plans its route from the {r}.",
    "The robot straightens the {s} in the {r} and moves on.",
    "Battery check in the {r}: fine.",
]
CP_HEADER = ("Below is the activity log of a home robot that moves household items between rooms, in order. "
             "Answer with the item's final location, copied exactly from the log, as in these examples.\n"
             "Question: Where does the log leave the kettle?\nAnswer: on the ledge in the cellar\n"
             "Question: Where does the log leave the hairbrush?\nAnswer: on the trolley in the porch\n\n")

# Command recall: a long numbered list of commands from a small formal grammar (verbs, numbered objects
# and receptacles, "in/on" phrasing), and a question whose gold is one exact command line.
CMD_OBJECTS = ["bowl", "mug", "sponge", "ladle", "jar", "towel", "candle", "remote", "notebook", "spoon", "plate",
               "bottle", "soap bar", "pencil", "spray can", "tissue box", "key ring", "lunchbox", "mitten", "trivet",
               "apple", "sock", "comb", "ruler", "marker"]
CMD_RECEPTACLES = ["crate", "locker", "hamper", "bin", "ledge", "trolley", "cupboard", "basin", "rack", "dresser",
                   "sideboard", "drawer", "cart", "shelf", "nightstand", "footstool", "bucket", "tray"]
CMD_TEMPLATES = [
    "walk to {R} {m}", "walk to {R} {m}", "grab {o} {n} from {R} {m}", "grab {o} {n} from {R} {m}",
    "place {o} {n} in/on {R} {m}", "place {o} {n} in/on {R} {m}", "stow {o} {n} in/on {R} {m}",
    "inspect {o} {n}", "inspect {R} {m}", "open {R} {m}", "shut {R} {m}", "rinse {o} {n} with basin {m}",
    "warm {o} {n} with stove {m}", "chill {o} {n} with icebox {m}", "switch on lamp {m}", "switch off lamp {m}",
    "wipe {o} {n} with cloth {m}", "hand over {o} {n}", "look around",
]
CMD_HEADER = ("Below is the numbered list of commands sent to a household robot, in order. Answer with the exact "
              "text of one command, as in these examples.\n"
              "Question: What was the command at step 3?\nAnswer: grab mug 2 from crate 1\n"
              "Question: What was the last command that mentioned towel 1?\nAnswer: place towel 1 in/on rack 2\n\n")


def _ntoks(tok, lines):
    return [len(x) for x in tok(lines, add_special_tokens=False).input_ids]


def copy_state(rng, tok, L, n_obj=20, n_moves=60):
    """(text, gold): robot log, 20 objects and 60 moves, filler lines up to about L tokens, and the final
    location of one object whose deciding move sits in a random band of the log (last tenth, middle, first tenth)."""
    objs = rng.sample(CP_OBJECTS, n_obj); state = {o: (rng.choice(CP_SURFACES), rng.choice(CP_ROOMS)) for o in objs}
    init = "Starting layout: " + "; ".join(f"the {o} is on the {s} in the {r}" for o, (s, r) in state.items()) + ".\n\nActivity log:\n"
    budget = L - len(tok(CP_HEADER + init, add_special_tokens=False).input_ids) - 48
    moves = []
    for _ in range(n_moves):
        o = rng.choice(objs); s1, r1 = state[o]; r2 = rng.choice([r for r in CP_ROOMS if r != r1]); s2 = rng.choice(CP_SURFACES)
        moves.append((o, rng.choice(CP_TEMPLATES).format(o=o, r1=r1, r2=r2, s=s2) + "\n")); state[o] = (s2, r2)
    lines = [m[1] for m in moves]; who = [m[0] for m in moves]
    used = sum(_ntoks(tok, lines))
    fills = [rng.choice(CP_FILLER).format(r=rng.choice(CP_ROOMS), s=rng.choice(CP_SURFACES)) + "\n" for _ in range(max(8, budget // 6))]
    for f, n in zip(fills, _ntoks(tok, fills)):
        if used + n > budget: break
        pos = rng.randrange(len(lines) + 1); lines.insert(pos, f); who.insert(pos, None); used += n
    N = len(lines); last = {}
    for i, o in enumerate(who):
        if o: last[o] = i
    cnt = {o: who.count(o) for o in objs}
    lo, hi = {0: (0.9, 1.0), 1: (0.4, 0.6), 2: (0.0, 0.1)}[rng.randrange(3)]
    cands = [o for o in objs if o in last and lo * N <= last[o] < hi * N and cnt[o] >= 2]
    if not cands: cands = [o for o in objs if o in last and cnt[o] >= 2] or list(last)
    o = rng.choice(cands); s, r = state[o]
    return CP_HEADER + init + "".join(lines) + f"\nQuestion: Where does the log leave the {o}?\nAnswer:", f"on the {s} in the {r}"


def copy_cmds(rng, tok, L):
    """(text, gold): a numbered command list up to about L tokens and one of five recall questions
    (command at step N, right after or before step N, last or first command mentioning an item)."""
    def cmd():
        return rng.choice(CMD_TEMPLATES).format(o=rng.choice(CMD_OBJECTS), n=rng.randint(1, 4), R=rng.choice(CMD_RECEPTACLES), m=rng.randint(1, 3))
    budget = L - len(tok(CMD_HEADER, add_special_tokens=False).input_ids) - 48
    cands = [cmd() for _ in range(max(16, budget // 5))]
    lines = [f"Step {i + 1}: {c}\n" for i, c in enumerate(cands)]
    cmds, used = [], 0
    for c, n in zip(cands, _ntoks(tok, lines)):
        if used + n > budget: break
        cmds.append(c); used += n
    N = len(cmds); text = CMD_HEADER + "".join(f"Step {i + 1}: {c}\n" for i, c in enumerate(cmds))
    kind = rng.randrange(5)
    if kind == 0:
        i = rng.randrange(N); q, gold = f"What was the command at step {i + 1}?", cmds[i]
    elif kind == 1:
        i = rng.randrange(N - 1); q, gold = f"What command came right after step {i + 1}?", cmds[i + 1]
    elif kind == 2:
        i = rng.randrange(1, N); q, gold = f"What command came right before step {i + 1}?", cmds[i - 1]
    else:
        item = f"{rng.choice(CMD_OBJECTS)} {rng.randint(1, 4)}" if rng.random() < 0.6 else f"{rng.choice(CMD_RECEPTACLES)} {rng.randint(1, 3)}"
        hits = [c for c in cmds if (" " + c + " ").find(" " + item + " ") >= 0]
        if not hits: return copy_cmds(rng, tok, L)
        q, gold = (f"What was the last command that mentioned {item}?", hits[-1]) if kind == 3 else (f"What was the first command that mentioned {item}?", hits[0])
    return text + f"\nQuestion: {q}\nAnswer:", gold


def copy_mix(tok, n, seed, lengths, min_tok=3, max_tok=10):
    """[(text, [gold], kind)]: n examples, half state tracking and half command recall, a random target
    length per example, golds of min_tok to max_tok tokens (as " gold\n")."""
    rng = random.Random(seed); out = []
    while len(out) < n:
        kind = "state" if len(out) % 2 == 0 else "cmds"
        text, gold = (copy_state if kind == "state" else copy_cmds)(rng, tok, rng.choice(lengths))
        k = len(tok(" " + gold + "\n", add_special_tokens=False).input_ids)
        if min_tok <= k <= max_tok: out.append((text, [gold], kind))
    return out


def eval_set(tok, name, n, lengths, seed):
    """{L: [(ids, golds, n_docs)]} from the validation split, same layout as longctx_eval."""
    recs, pool = load(name, "validation", n, seed)
    built = build(tok, recs, pool, lengths, seed, every_length=True)
    return {L: [(tok(t, return_tensors="pt").input_ids, g, t.count("Document")) for t, g in built[L]] for L in lengths}
