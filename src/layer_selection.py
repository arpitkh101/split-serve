"""Three ways to choose source layers, at matched k with everything else identical: even
spacing, KL-matched logit-lens distributions, and R2-matched caches.
"""
import sys, json, warnings, torch
warnings.filterwarnings("ignore")
import common as C
from ridge_sweep import fit_ridge_blocked, run

DEV, OUT = "cuda:0", C.WORK
CHUNK, PROMPT, N_EVAL = 512, 256, 64


def main():
    aff = torch.load(f"{OUT}/affinity.pt"); R2, KL = aff["R2"], aff["KL"]
    Ls, Lt = R2.shape
    Xcpu = torch.load(f"{OUT}/kv_4b.pt")["kv"]; Ycpu = torch.load(f"{OUT}/kv_17b.pt")["kv"]
    m4, tok = C.load_model(C.PREFILL_MODEL, DEV)
    m17, _ = C.load_model(C.DECODE_MODEL, DEV)
    chunks = C.load_text_chunks(tok, N_EVAL, CHUNK, split="test", seed=0)
    base = json.load(open(f"{C.RESULTS}/baselines_eval64.json"))

    print("=" * 78); print("LAYER-SELECTION METRIC COMPARISON"); print("=" * 78)
    print(f"  Qwen3-4B standalone {base['Qwen3-4B']:.3f}   "
          f"Qwen3-1.7B standalone {base['Qwen3-1.7B']:.3f}\n")
    res = {}
    for k in (1, 4, 8):
        print(f"  --- k = {k} source layers per target ---")
        maps = {
            # contiguous window of exactly k layers centred on the linspace
            # position -- clamping with a set gave short lists at the ends
            "uniform": [list(range(max(0, min(Ls - k,
                                              round(j*(Ls-1)/(Lt-1)) - k//2)),
                                   max(0, min(Ls - k,
                                              round(j*(Ls-1)/(Lt-1)) - k//2)) + k))
                        for j in range(Lt)],
            "KL-min":  [KL[:, j].topk(k, largest=False).indices.tolist() for j in range(Lt)],
            "R2-max":  [R2[:, j].topk(k, largest=True).indices.tolist() for j in range(Lt)],
        }
        for nm, sm in maps.items():
            W, b, r2 = fit_ridge_blocked(sm, Xcpu, Ycpu)
            ppl = run(f"{nm:10s} (fit R2 {r2:.4f})", m4, sm, W, b, m17, chunks)
            res[f"k{k}_{nm}"] = {"ppl": ppl, "r2": r2}
            del W, b; torch.cuda.empty_cache()
        print()
    json.dump(res, open(f"{C.RESULTS}/metric_compare.json", "w"), indent=2)

    print("=" * 78); print("SUMMARY  (ppl, lower is better)"); print("=" * 78)
    print(f"  {'k':>3s}  {'uniform':>10s} {'KL-min':>10s} {'R2-max':>10s}")
    for k in (1, 4, 8):
        row = [res[f"k{k}_{n}"]["ppl"] for n in ("uniform", "KL-min", "R2-max")]
        print(f"  {k:3d}  " + " ".join(f"{v:10.2f}" for v in row))
    print(f"\n  reference: 1.7B standalone = {base['Qwen3-1.7B']:.3f}")


if __name__ == "__main__":
    main()
