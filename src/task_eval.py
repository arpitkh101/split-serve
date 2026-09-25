"""Task-level retention on ARC-Easy, ARC-Challenge, PIQA (5-shot) and LAMBADA. Log-likelihood
multiple choice, lm-eval style: the few-shot prompt is the prefill (where the pipeline runs),
each answer choice is decoded by the 1.7B off that cache. Per-example hits are saved for
paired bootstrap. Conditions: 17b, 4b, pipe:TAG (residual-cut pipeline), hv2:TAG (with its
tail LoRA), reskv:TAG (residual-space stitch), reskv:TAG@S (last S prompt tokens recomputed
natively), reskv:TAG@4b (first decode token from the 4B's own logits).
"""
import sys, json, math, time, warnings, argparse, random, torch
warnings.filterwarnings("ignore")
import common as C
from residual_cut import load_sdpa, HiddenStitch, prefill_pipeline, DEV, OUT
from kv_stitch import BridgeV2, parse_hidden
from residual_cut_train import TARGETS

FEWSHOT = 5


# ------------------------------------------------------------------ tasks

def load_task(name, n, seed):
    from datasets import load_dataset
    rng = random.Random(seed)
    if name in ("arc_easy", "arc_challenge"):
        cfg = "ARC-Easy" if name == "arc_easy" else "ARC-Challenge"
        ds = load_dataset("allenai/ai2_arc", cfg); train, test = list(ds["train"]), list(ds["test"])
        def fmt(ex): return f"Question: {ex['question']}\nAnswer:"
        def mc(ex):
            gold = ex["choices"]["label"].index(ex["answerKey"])
            return fmt(ex), [" " + t for t in ex["choices"]["text"]], gold
        shots = "".join(fmt(ex) + " " + ex["choices"]["text"][ex["choices"]["label"].index(ex["answerKey"])] + "\n\n"
                        for ex in rng.sample(train, FEWSHOT))
        return "mc", shots, [mc(ex) for ex in rng.sample(test, min(n, len(test)))]
    if name == "piqa":
        ds = load_dataset("baber/piqa"); train, test = list(ds["train"]), list(ds["validation"])
        def fmt(ex): return f"Question: {ex['goal']}\nAnswer:"
        def mc(ex): return fmt(ex), [" " + ex["sol1"], " " + ex["sol2"]], ex["label"]
        shots = "".join(fmt(ex) + " " + ex["sol1" if ex["label"] == 0 else "sol2"] + "\n\n" for ex in rng.sample(train, FEWSHOT))
        return "mc", shots, [mc(ex) for ex in rng.sample(test, min(n, len(test)))]
    if name == "lambada":
        ds = load_dataset("EleutherAI/lambada_openai", "default"); test = list(ds["test"])
        exs = []
        for ex in rng.sample(test, min(n, len(test))):
            ctx, last = ex["text"].rsplit(" ", 1); exs.append((ctx, [" " + last], 0))
        return "lambada", "", exs
    raise ValueError(name)


# ------------------------------------------------------------- conditions

class Cond:
    """prefill(ids) -> (cache, logits over the prompt); the 1.7B (or 4B) then decodes."""
    def __init__(self, name, m4, m17, tok):
        self.name, self.m4, self.m17, self.tok = name, m4, m17, tok
        self.dec = m4 if name == "4b" else m17
        self.pipe = False; self.reskv = None
        if name.startswith("reskv:"):
            import residual_stitch as R
            tag = name.split(":")[1]; self.suffix = 0; self.first4b = False
            if "@" in tag:
                tag, sfx = tag.split("@")
                if sfx == "4b": self.first4b = True      # first decode token from the 4B's own logits (DistServe-style)
                else: self.suffix = int(sfx)            # last S prompt tokens recomputed natively
            maps, Lc, source = R.load_maps(tag)
            self.reskv = (R, maps, Lc, source); return
        if ":" in name:
            kind, tag = name.split(":")
            ci = torch.load(f"{OUT}/{'hidden' if kind == 'pipe' else 'hv2'}_{tag}.pt")
            self.Lc, self.js = ci["cut"], ci["src_layer"]
            self.source = "17b" if "self" in tag else "4b"
            ck = torch.load(f"{OUT}/v2_k1_full_h0{'' if self.source == '4b' else '_self'}.pt"); Lt = len(ck["src_map"])
            self.kvbr = BridgeV2(ck["src_map"], torch.zeros(Lt, 8, 256, 256), torch.zeros(Lt, 8, 256), 0, parse_hidden("0", Lt))
            self.kvbr.load_state_dict(ci["kv"]); self.kvbr = self.kvbr.to(DEV).float()
            self.hbr = HiddenStitch(ci["h"]["W"], ci["h"]["b"]).to(DEV).float()
            self.pipe = True
            if ci.get("lora"):
                from peft import LoraConfig, get_peft_model
                r = next(iter(ci["lora"].values())).shape[0]
                get_peft_model(m17, LoraConfig(r=r, lora_alpha=r, lora_dropout=0.0, target_modules=TARGETS,
                                               layers_to_transform=list(range(self.Lc, 28))))
                sd = dict(m17.named_parameters()); miss = [n for n in ci["lora"] if n not in sd]
                assert not miss, miss[:3]
                for n, p in ci["lora"].items(): sd[n].data.copy_(p.to(sd[n].dtype))
                for p in m17.parameters(): p.requires_grad_(False)
                print(f"  [{name}] tail LoRA r={r} loaded ({len(ci['lora'])} tensors)", flush=True)

    @torch.no_grad()
    def prefill(self, ids):
        if self.reskv:
            R, maps, Lc, source = self.reskv; T = ids.shape[1]; S = min(self.suffix, T - 1)
            if self.first4b:
                # one 4B forward gives both the source residuals and the prefiller's own next-token logits
                box = {}; hk = self.m4.model.norm.register_forward_hook(lambda m, i, o: box.__setitem__("pre", i[0]))
                try: o4 = self.m4(input_ids=ids, output_hidden_states=True, use_cache=False)
                finally: hk.remove()
                H = tuple(o4.hidden_states[:-1]) + (box["pre"],)
                cache, _ = R.prefill(self.m4, self.m17, maps, ids, Lc, source, H=H)
                return cache, o4.logits
            if S <= 0: return R.prefill(self.m4, self.m17, maps, ids, Lc, source)
            cache, _ = R.prefill(self.m4, self.m17, maps, ids[:, :T - S], Lc, source)
            pos = torch.arange(T - S, T, device=DEV).unsqueeze(0)
            o = self.m17(input_ids=ids[:, T - S:], past_key_values=cache, position_ids=pos, cache_position=pos[0], use_cache=True)
            return o.past_key_values, o.logits
        if self.pipe:
            return prefill_pipeline(self.m4, self.m17, self.kvbr, self.hbr, ids, self.Lc, self.js, self.source)
        o = self.dec(input_ids=ids, use_cache=True)
        return o.past_key_values, o.logits

    @torch.no_grad()
    def score(self, ctx_ids, conts):
        """Per continuation: (sum logprob, n tokens, greedy-all-correct)."""
        T = ctx_ids.shape[1]
        cache, lg = self.prefill(ctx_ids); first = torch.log_softmax(lg[0, -1].float(), -1)
        out = []
        for c in conts:
            c = c.to(DEV); lp = first[c[0, 0]].item(); ok = first.argmax().item() == c[0, 0].item()
            if c.shape[1] > 1:
                pos = torch.arange(T, T + c.shape[1], device=DEV).unsqueeze(0)
                l2 = torch.log_softmax(self.dec(input_ids=c, past_key_values=cache, position_ids=pos,
                                                cache_position=pos[0], use_cache=True).logits[0, :-1].float(), -1)
                tgt = c[0, 1:]; lp += l2[torch.arange(tgt.numel()), tgt].sum().item()
                ok = ok and bool((l2.argmax(-1) == tgt).all())
                cache.crop(T)
            out.append((lp, c.shape[1], ok))
        return out


def run_task(cond, tok, kind, shots, exs):
    acc = accn = ok_last = 0; nll = 0.0; ntok = 0; t0 = time.time(); per = []
    for ctx, conts, gold in exs:
        ctx_ids = tok(shots + ctx, return_tensors="pt").input_ids.to(DEV)
        cont_ids = [tok(c, return_tensors="pt", add_special_tokens=False).input_ids for c in conts]
        sc = cond.score(ctx_ids, cont_ids)
        if kind == "mc":
            hit = int(max(range(len(sc)), key=lambda i: sc[i][0]) == gold); acc += hit; per.append(hit)
            accn += int(max(range(len(sc)), key=lambda i: sc[i][0] / sc[i][1]) == gold)
        else:
            lp, n, ok = sc[0]; nll -= lp; ntok += n; ok_last += int(ok); per.append(int(ok))
    N = len(exs)
    r = {"acc": acc / N, "acc_norm": accn / N} if kind == "mc" else {"acc": ok_last / N, "ppl": math.exp(nll / ntok)}
    r.update({"n": N, "sec": round(time.time() - t0), "per": per})
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="arc_easy,arc_challenge,piqa,lambada")
    ap.add_argument("--conds", default="17b,4b,pipe:cut20_4b,pipe:cut20_self")
    ap.add_argument("--n", type=int, default=500); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    m4, tok = load_sdpa(C.PREFILL_MODEL); m17, _ = load_sdpa(C.DECODE_MODEL)
    conds = [Cond(c, m4, m17, tok) for c in a.conds.split(",")]
    res = {}
    for t in a.tasks.split(","):
        kind, shots, exs = load_task(t, a.n, a.seed)
        n_ctx = tok(shots + exs[0][0], return_tensors="pt").input_ids.shape[1]
        print(f"[{t}] {len(exs)} examples, {FEWSHOT if kind == 'mc' else 0}-shot, first prompt {n_ctx} tokens", flush=True)
        res[t] = {}
        for c in conds:
            res[t][c.name] = run_task(c, tok, kind, shots, exs)
            print(f"  {c.name:16s} " + "  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                                  for k, v in res[t][c.name].items() if k != "per"), flush=True)
        json.dump(res, open(f"{C.RESULTS}/task_{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
