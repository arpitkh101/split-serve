"""ALFWorld next-action prediction as a handoff test.  Trajectories from THUDM/AgentInstruct (336
household episodes, observation and action turns).  A prompt is the environment instruction, then
earlier completed episodes as the robot's history until the length budget is full, then the current
episode cut before one of its actions, ending in "ACTION:".  Gold is that action string, scored by
exact match after normalisation.  Held out from everything the stitch trained on.

  python src/longctx_eval.py --dataset alfworld --conds 17b,4b,reskv:qamix_4b_c28@4b,reskv:qamix_self_c28 --n 300 --tag alfworld
"""
import random, re, torch


def episodes():
    from datasets import load_dataset
    ds = load_dataset("THUDM/AgentInstruct", split="alfworld"); out = []
    for ex in ds:
        conv = ex["conversations"]
        instr = conv[0]["value"].strip(); steps = []
        for t in conv[2:]:
            if t["from"] == "human": steps.append([t["value"].strip(), None])
            else:
                m = re.search(r"ACTION:\s*(.+)", t["value"]); act = m.group(1).strip() if m else None
                if steps and steps[-1][1] is None: steps[-1][1] = act
        steps = [s for s in steps if s[1]]
        if len(steps) >= 3: out.append((instr, steps))
    return out


def render(steps, upto=None):
    return "".join(f"{obs}\nACTION: {act}\n\n" for obs, act in steps[:upto])


def eval_set(tok, n, lengths, seed):
    """{L: [(ids, [gold action], n_prior_episodes)]}.  Same episode and cut point at every length."""
    eps = episodes(); rng = random.Random(seed)
    cache = {}
    def ntok(t):
        if t not in cache: cache[t] = len(tok(t, add_special_tokens=False).input_ids)
        return cache[t]
    picks = []
    for i in range(n):
        e = rng.randrange(len(eps)); instr, steps = eps[e]
        k = rng.randrange(2, len(steps))                     # predict the action of step k, after at least two actions
        picks.append((e, k))
    out = {}
    for L in lengths:
        rows = []
        for i, (e, k) in enumerate(picks):
            instr, steps = eps[e]
            cur = render(steps, k) + steps[k][0] + "\nACTION:"
            head = instr + "\n\nEarlier episodes in this household, oldest first, then the current one.\n\n"
            budget = L - ntok(head) - ntok("=== Current episode ===\n" + cur) - 8
            prng = random.Random(seed * 7919 + i); others = [j for j in range(len(eps)) if j != e]; prng.shuffle(others)
            prior = []
            for j in others:
                blk = "=== Earlier episode ===\n" + render(eps[j][1]) + "\n"
                if ntok(blk) > budget: continue
                prior.append(blk); budget -= ntok(blk)
                if budget < 60: break
            prior.reverse()
            text = head + "".join(prior) + "=== Current episode ===\n" + cur
            rows.append((tok(text, return_tensors="pt").input_ids, [steps[k][1]], len(prior)))
        out[L] = rows
    return out
