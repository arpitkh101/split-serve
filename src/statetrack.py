"""Synthetic embodied state tracking, held out from everything the stitch trained on.  A log of a
mobile robot moving objects between rooms, hundreds of moves long, then "Which room is the X in
now?".  The answer is the room of the object's LAST move, so the model has to find the most recent
mention of the object among earlier, stale mentions and later moves of other objects: lookup plus
state update, no world knowledge, one-word gold.  Example i puts the deciding move in band i % 3 of
the log: 0 = last tenth, 1 = middle, 2 = first tenth (every later move is about other objects).

  python src/longctx_eval.py --dataset statetrack --conds 17b,4b,reskv:qamix_4b_c28@4b,reskv:qamix_self_c28 --n 300 --tag statetrack
"""
import random, torch

ROOMS = ["kitchen", "garage", "lab", "office", "bedroom", "hallway", "workshop", "pantry", "lounge", "studio"]
SURFACES = ["table", "shelf", "bench", "counter", "cart", "desk", "rack", "crate", "floor", "tray", "stool", "cabinet"]
OBJECTS = ["mug", "wrench", "notebook", "lamp", "kettle", "hammer", "tablet", "plant", "bottle", "camera", "scarf",
           "stapler", "helmet", "basket", "pillow", "speaker", "toolbox", "teapot", "folder", "cushion", "globe",
           "ladle", "charger", "clock", "vase", "ruler", "candle", "blanket", "trowel", "backpack", "umbrella",
           "pencil case", "spray bottle", "measuring tape", "coffee tin", "picture frame", "water jug",
           "screwdriver", "flashlight", "sketchbook", "keyboard", "watering can", "thermos", "paint roller",
           "ice tray", "fruit bowl", "extension cord", "soldering iron", "cutting board", "paper bag",
           "tennis ball", "doorstop", "bread box", "dustpan", "hand mixer", "yoga mat", "safety goggles",
           "label maker", "spice rack", "wire cutters"]
TEMPLATES = [
    "The robot picks up the {o} in the {r1} and puts it on the {s} in the {r2}.",
    "Moving the {o} from the {r1} to the {r2}, the robot leaves it on the {s}.",
    "The {o} is carried out of the {r1} and set down on the {s} in the {r2}.",
    "After a trip to the {r2}, the {o} that was in the {r1} now sits on the {s} there.",
    "The robot takes the {o} off its spot in the {r1}; it ends up on the {s} in the {r2}.",
]
HEADER = ("Below is the log of a mobile robot moving objects between rooms, in order. Answer with the "
          "name of the room where the object is at the end of the log, as in these examples.\n"
          "Question: Which room is the saucepan in now?\nAnswer: kitchen\n"
          "Question: Which room is the drill in now?\nAnswer: garage\n\n")


def make(rng, tok, L, band, n_obj=40):
    objs = rng.sample(OBJECTS, n_obj); state = {o: rng.choice(ROOMS) for o in objs}
    init = "Initial positions: " + "; ".join(f"the {o} is in the {state[o]}" for o in objs) + ".\n\nLog:\n"
    budget = L - len(tok(HEADER + init, add_special_tokens=False).input_ids) - 40
    events, who = [], []
    while budget > 0:
        o = rng.choice(objs); r1 = state[o]; r2 = rng.choice([r for r in ROOMS if r != r1]); s = rng.choice(SURFACES)
        line = rng.choice(TEMPLATES).format(o=o, r1=r1, r2=r2, s=s) + "\n"
        n = len(tok(line, add_special_tokens=False).input_ids)
        if n > budget: break
        events.append(line); who.append(o); state[o] = r2; budget -= n
    N = len(events); last = {}
    for i, o in enumerate(who): last[o] = i
    moves = {o: who.count(o) for o in objs}
    lo, hi = {0: (0.9, 1.0), 1: (0.4, 0.6), 2: (0.0, 0.1)}[band]
    cands = [o for o in objs if o in last and lo * N <= last[o] < hi * N and moves[o] >= 2]
    if not cands: cands = [o for o in objs if o in last and moves[o] >= 2] or list(last)
    o = rng.choice(cands)
    text = HEADER + init + "".join(events) + f"\nQuestion: Which room is the {o} in now?\nAnswer:"
    return text, [state[o]], N, (N - 1 - last[o])


def eval_set(tok, n, lengths, seed):
    """{L: [(ids, golds, n_events)]}; example i has its deciding move in band i % 3."""
    out = {}
    for L in lengths:
        rng = random.Random(seed * 1000 + L); rows = []
        for i in range(n):
            text, golds, N, back = make(rng, tok, L, i % 3)
            rows.append((tok(text, return_tensors="pt").input_ids, golds, N))
        out[L] = rows
    return out


def by_band(per):
    """EM per band from the harness's per-example hits, in example order."""
    return {b: round(100 * sum(p[0] for p in per[b::3]) / max(1, len(per[b::3])), 1) for b in range(3)}
