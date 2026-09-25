"""Does a KV stitch trained on 256-token prompts hold at 512 / 1,024 / 2,048? Scores the same
continuation under the untrained 1.7B, the 4B, the stitch, and the self-bridge control.
"""
import sys, json, warnings, torch
warnings.filterwarnings("ignore")
import common as C
import probe as S
from kv_stitch_mlp import make_cache
from paired_ci_kv import load_fp32

DEV, OUT, CONT, N_EVAL = "cuda:0", C.WORK, 256, 64


@torch.no_grad()
def main():
    br4, brs = load_fp32("v2_k1_full_h0"), load_fp32("v2_k1_full_h0_self")
    m4, tok = C.load_model(C.PREFILL_MODEL, DEV); m17, _ = C.load_model(C.DECODE_MODEL, DEV)
    rows = []
    for P in (256, 512, 1024, 2048):
        chunks = C.load_text_chunks(tok, N_EVAL, P + CONT, split="test", seed=0)
        r = {"prompt": P}
        acc = {k: [] for k in ("1.7B", "4B", "bridge", "self")}
        for ch in chunks:
            ids = ch.unsqueeze(0).to(DEV)
            acc["1.7B"].append(S.continuation_nll(m17, ch, P)); acc["4B"].append(S.continuation_nll(m4, ch, P))
            kv4 = C.capture_kv(m4, ids[:, :P], strip=True)
            acc["bridge"].append(S.continuation_nll(m17, ch, P, cache=make_cache(br4, kv4, m17, P)))
            kv17 = C.capture_kv(m17, ids[:, :P], strip=True)
            acc["self"].append(S.continuation_nll(m17, ch, P, cache=make_cache(brs, kv17, m17, P)))
            del kv4, kv17
        for k, v in acc.items(): r[k] = torch.tensor(v).mean().exp().item()
        rows.append(r)
        print(f"  prompt {P:5d}:  1.7B {r['1.7B']:7.3f}   4B {r['4B']:7.3f}   "
              f"bridge {r['bridge']:7.3f}   self-bridge {r['self']:7.3f}   "
              f"(bridge − 1.7B {r['bridge']-r['1.7B']:+.2f}, bridge − self {r['bridge']-r['self']:+.2f})", flush=True)
    json.dump(rows, open(f"{C.RESULTS}/length_gen.json", "w"), indent=2)


if __name__ == "__main__":
    main()
