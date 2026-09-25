"""Wikitext cost/benefit of a native suffix: the residual-space pipeline prefills the first T-S
prompt tokens and the full 1.7B recomputes the last S over that cache. Eval only.
"""
import sys, json, warnings, argparse, torch
warnings.filterwarnings("ignore")
import common as C
import residual_stitch as R
from residual_cut import load_sdpa
from residual_cut_train import nll_of, ppl, CONT, N_EVAL

DEV = "cuda:0"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="reskv_4b"); ap.add_argument("--suffixes", default="0,16,32,64,128")
    a = ap.parse_args()
    m4, tok = load_sdpa(C.PREFILL_MODEL); m17, _ = load_sdpa(C.DECODE_MODEL)
    maps, Lc, source = R.load_maps(a.tag); R.DEV = DEV
    out = {}
    for P in (256, 512, 1024, 2048):
        chunks = C.load_text_chunks(tok, N_EVAL, P + CONT, split="test", seed=0); out[P] = {}
        for S in [int(x) for x in a.suffixes.split(",")]:
            nll = []
            for ch in chunks:
                ids = ch.unsqueeze(0).to(DEV)
                cache, _ = R.prefill(m4, m17, maps, ids[:, :P - S], Lc, source)
                pos = torch.arange(P - S, ids.shape[1], device=DEV).unsqueeze(0)
                lg = m17(input_ids=ids[:, P - S:], past_key_values=cache, position_ids=pos, cache_position=pos[0], use_cache=True).logits[0].float()
                nll.append(nll_of(lg[S:-1], ids, P).item())      # score continuation tokens P+1.. as everywhere else
            out[P][S] = {"ppl": ppl(nll), "nll": nll}
        print(f"  P{P}: " + "  ".join(f"S{S} {out[P][S]['ppl']:.3f}" for S in out[P]), flush=True)
    json.dump(out, open(f"{C.RESULTS}/suffix_{a.tag}.json", "w"))


if __name__ == "__main__":
    main()
