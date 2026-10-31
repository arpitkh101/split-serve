"""Closed-book check for a QA set: the same questions with no documents, so a score on the open-book
prompts can be read as "memory plus context".  python src/closedbook.py --dataset triviaqa --n 500"""
import sys, json, argparse, warnings, torch
warnings.filterwarnings("ignore")
import common as C, qa_data, longctx_eval as LC
from residual_cut import load_sdpa, DEV
from task_eval import Cond

ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="triviaqa"); ap.add_argument("--n", type=int, default=500); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
m4, tok = load_sdpa(C.PREFILL_MODEL); m17, _ = load_sdpa(C.DECODE_MODEL)
recs, _ = qa_data.load(a.dataset, "validation", a.n, a.seed)
res = {}
for name in ["17b", "4b"]:
    cond = Cond(name, m4, m17, tok); em = 0; per = []
    for docs, turns, golds in recs:
        ids = tok(LC.HEADER + f"Question: {turns[-1][0]}\nAnswer:", return_tensors="pt").input_ids.to(DEV)
        pred, _ = LC.generate(cond, tok, ids)
        e = max(LC.em_f1(pred, g)[0] for g in golds); em += e; per.append(e)
    res[name] = {"em": em / len(recs), "per": per}; print(f"{a.dataset} closed book {name}: EM {100 * em / len(recs):.1f}", flush=True)
json.dump(res, open(f"{C.RESULTS}/closedbook_{a.dataset}.json", "w"))
