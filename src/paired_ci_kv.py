"""Paired bootstrap for a KV stitch pair: 4B-fed stitch vs the same stitch fed the 1.7B's own
cache, on identical chunks, plus the untrained 1.7B and 4B on the same chunks.
"""
import sys, json, warnings, argparse, torch
warnings.filterwarnings("ignore")
import common as C
from probe import continuation_nll
from kv_stitch_mlp import make_cache
from latency_kv import load_bridge as _lb
from kv_stitch import BridgeV2, parse_hidden

DEV, OUT = "cuda:0", C.WORK
CHUNK, PROMPT, N_EVAL = 512, 256, 64


def load_fp32(tag):
    ck = torch.load(f"{OUT}/{tag}.pt")
    Lt, k = len(ck["src_map"]), ck["k"]
    W = torch.zeros(Lt, 8, k * 256, 256); b = torch.zeros(Lt, 8, 256)
    br = BridgeV2(ck["src_map"], W, b, ck["rank"], parse_hidden(ck["hidden"], Lt), ck.get("share_heads", False))
    br.load_state_dict(ck["state"]); return br.to(DEV).float().eval()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); a = ap.parse_args()
    br4, brs = load_fp32(f"v2_{a.tag}"), load_fp32(f"v2_{a.tag}_self")
    m4, tok = C.load_model(C.PREFILL_MODEL, DEV); m17, _ = C.load_model(C.DECODE_MODEL, DEV)
    chunks = C.load_text_chunks(tok, N_EVAL, CHUNK, split="test", seed=0)
    nll = {k: [] for k in ("1.7B", "4B", "bridge", "self")}
    for ch in chunks:
        ids = ch.unsqueeze(0).to(DEV)
        nll["1.7B"].append(continuation_nll(m17, ch, PROMPT)); nll["4B"].append(continuation_nll(m4, ch, PROMPT))
        kv4 = C.capture_kv(m4, ids[:, :PROMPT], strip=True)
        nll["bridge"].append(continuation_nll(m17, ch, PROMPT, cache=make_cache(br4, kv4, m17, PROMPT)))
        kv17 = C.capture_kv(m17, ids[:, :PROMPT], strip=True)
        nll["self"].append(continuation_nll(m17, ch, PROMPT, cache=make_cache(brs, kv17, m17, PROMPT)))
    N = {k: torch.tensor(v, dtype=torch.float64) for k, v in nll.items()}
    print(f"[{a.tag}] reproduced ppl: " + "  ".join(f"{k} {v.mean().exp():.3f}" for k, v in N.items()))
    idx = torch.randint(0, N_EVAL, (10000, N_EVAL), generator=torch.Generator().manual_seed(0))
    res = {}
    for x, y, what in (("self", "bridge", "value of the 4B cache"), ("1.7B", "bridge", "gain over untrained 1.7B"),
                       ("bridge", "4B", "distance to untrained 4B")):
        d = N[x][idx].mean(1).exp() - N[y][idx].mean(1).exp()
        pt = N[x].mean().exp().item() - N[y].mean().exp().item()
        lo, hi = torch.quantile(d, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
        wins = int((N[y] < N[x]).sum())
        res[f"{x}_minus_{y}"] = dict(point=pt, ci95=[lo, hi], y_better_chunks=wins, what=what)
        print(f"  {x:7s} - {y:7s} = {pt:+6.3f}  95% CI [{lo:+6.3f}, {hi:+6.3f}]  {y} better on {wins}/{N_EVAL}  ({what})")
    json.dump(res, open(f"{C.RESULTS}/paired_ci_{a.tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
