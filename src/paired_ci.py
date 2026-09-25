"""Paired bootstrap confidence intervals (10,000 resamples) from saved per-chunk NLLs
(out/hv2_*.json) and per-example task hits (out/task_*.json).

  python src/paired_ci.py hv2 TAG_A TAG_B [TAG_C TAG_D ...]   perplexity gaps, pipeline vs control
  python src/paired_ci.py task TAG                             accuracy gaps vs the 17b row and the self control
"""
import sys, json, math, torch
import common as C

R, OUT = 10000, C.RESULTS


def boot_ppl_gap(a, b, g):
    a, b = torch.tensor(a), torch.tensor(b); n = a.numel()
    idx = torch.randint(0, n, (R, n), generator=g)
    d = (a[idx].mean(1).exp() - b[idx].mean(1).exp())
    return (a.mean().exp() - b.mean().exp()).item(), d.quantile(0.025).item(), d.quantile(0.975).item(), (d > 0).float().mean().item()


def boot_acc_gap(a, b, g):
    a, b = torch.tensor(a, dtype=torch.float), torch.tensor(b, dtype=torch.float); n = a.numel()
    idx = torch.randint(0, n, (R, n), generator=g)
    d = a[idx].mean(1) - b[idx].mean(1)
    return (a.mean() - b.mean()).item(), d.quantile(0.025).item(), d.quantile(0.975).item()


def main():
    g = torch.Generator().manual_seed(0); mode = sys.argv[1]; res = {}
    if mode == "hv2":
        pairs = list(zip(sys.argv[2::2], sys.argv[3::2]))
        for ta, tb in pairs:
            A, B = json.load(open(f"{OUT}/hv2_{ta}.json")), json.load(open(f"{OUT}/hv2_{tb}.json"))
            print(f"\n{ta}  vs  {tb}   (gap = pipeline − control, ppl)")
            for P in A["nll_final"]:
                gap, lo, hi, p = boot_ppl_gap(A["nll_final"][P], B["nll_final"][P], g)
                print(f"  P{P:>5}: {A['final'][P]:7.3f} vs {B['final'][P]:7.3f}   gap {gap:+.3f}  95% [{lo:+.3f}, {hi:+.3f}]   P(gap>0) {p:.3f}")
                res[f"{ta}|{tb}|{P}"] = {"a": A["final"][P], "b": B["final"][P], "gap": gap, "lo": lo, "hi": hi, "p_gt0": p}
        json.dump(res, open(f"{OUT}/paired_ci_hv2.json", "w"), indent=2)
    else:
        T = json.load(open(f"{OUT}/task_{sys.argv[2]}.json"))
        for task, conds in T.items():
            names = list(conds); ref = "17b"
            print(f"\n[{task}]  n={conds[ref]['n']}   acc: " + "  ".join(f"{c} {conds[c]['acc']:.3f}" for c in names))
            for c in names:
                if c == ref: continue
                gap, lo, hi = boot_acc_gap(conds[c]["per"], conds[ref]["per"], g)
                print(f"  {c:16s} − 1.7B: {gap*100:+.1f} pts  95% [{lo*100:+.1f}, {hi*100:+.1f}]")
                res[f"{task}|{c}"] = {"gap": gap, "lo": lo, "hi": hi}
            for c in [n for n in names if n.startswith(("pipe:", "hv2:", "reskv:")) and "self" not in n]:
                ctrl = [n for n in names if "self" in n]
                if ctrl:
                    gap, lo, hi = boot_acc_gap(conds[c]["per"], conds[ctrl[0]]["per"], g)
                    print(f"  {c:16s} − {ctrl[0]}: {gap*100:+.1f} pts  95% [{lo*100:+.1f}, {hi*100:+.1f}]")
                    res[f"{task}|{c}|ctrl"] = {"gap": gap, "lo": lo, "hi": hi}
        json.dump(res, open(f"{OUT}/paired_ci_task_{sys.argv[2]}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
