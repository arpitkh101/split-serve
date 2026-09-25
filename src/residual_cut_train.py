"""Training levers on the residual-cut pipeline: a LoRA restricted to the 1.7B's layers at and
above the cut (the ones that read the stitched residual), mixed prompt lengths, and a choice of
corpus (wikitext-2, wikitext-103, an instruction set, or a mix). Starts from a saved pipeline
checkpoint. Per-chunk NLLs at 256 / 512 / 1024 / 2048 are saved for paired bootstrap.
"""
import sys, json, math, time, warnings, argparse, torch
warnings.filterwarnings("ignore")
import common as C
from residual_cut import (load_sdpa, collect_hidden, ridge_fit, HiddenStitch, prefill_pipeline,
                           continuation_logits, DEV, OUT)
from kv_stitch import BridgeV2, parse_hidden

CONT, N_EVAL = 256, int(__import__("os").environ.get("HV2_NEVAL", 64))
N_TRAIN = {256: 3000, 512: 1500, 1024: 800, 2048: 400}     # wikitext-2 train is ~2.4M tokens
N_VAL = {256: 32, 512: 8, 1024: 8, 2048: 8}
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def nll_of(logits, ids, P):
    tgt = ids[0, P + 1:]
    return -torch.log_softmax(logits, -1)[torch.arange(tgt.numel()), tgt].mean()


def pipeline_nll(m4, m17, kvbr, hbr, ids, P, Lc, js, source):
    cache, _ = prefill_pipeline(m4, m17, kvbr, hbr, ids[:, :P], Lc, js, source)
    return nll_of(continuation_logits(m17, cache, ids, P), ids, P)


@torch.no_grad()
def evaluate(m4, m17, kvbr, hbr, chunks, P, Lc, js, source):
    return [pipeline_nll(m4, m17, kvbr, hbr, ch.unsqueeze(0).to(DEV), P, Lc, js, source).item() for ch in chunks]


@torch.no_grad()
def baseline(m, chunks, P):
    out = []
    for ch in chunks:
        ids = ch.unsqueeze(0).to(DEV)
        out.append(nll_of(m(input_ids=ids, use_cache=False).logits[0, P:-1].float(), ids, P).item())
    return out


def ppl(nll): return math.exp(sum(nll) / len(nll))


_CORPUS = {}
def corpus_ids(tok, name):
    """Token stream for a training corpus, cached per process.
    wt103  : 8% of wikitext-103 train (~9.8M tokens), articles disjoint from wikitext-2 test
    platy  : open-platypus (25k instruction/answer pairs, ~8M tokens) as "Question: ..\nAnswer: ..\n\n"
    """
    if name not in _CORPUS:
        from datasets import load_dataset
        if name == "wt103":
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:8%]")
            text = "\n\n".join(t for t in ds["text"] if t.strip())
        else:
            ds = load_dataset("garage-bAInd/open-platypus", split="train").shuffle(seed=0)
            text = "".join(f"Question: {e['instruction']}\nAnswer: {e['output']}\n\n" for e in ds)
        _CORPUS[name] = tok(text, return_tensors="pt").input_ids[0]
        print(f"  corpus {name}: {_CORPUS[name].numel()/1e6:.1f}M tokens", flush=True)
    return _CORPUS[name]


def load_corpus_chunks(tok, corpus, n, chunk_len, seed):
    """Random chunks; 'mixqa' = half wt103, half platypus, shuffled."""
    names = ["wt103", "platy"] if corpus == "mixqa" else [corpus]
    out = []
    for i, nm in enumerate(names):
        ids = corpus_ids(tok, nm); g = torch.Generator().manual_seed(seed + i)
        k = n // len(names) + (n % len(names) if i == 0 else 0)
        starts = torch.randperm(ids.numel() - chunk_len, generator=g)[:k]
        out.append(torch.stack([ids[s: s + chunk_len] for s in starts.tolist()]))
    out = torch.cat(out); g = torch.Generator().manual_seed(seed)
    return out[torch.randperm(out.shape[0], generator=g)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", type=int, default=20); ap.add_argument("--source", default="4b", choices=["4b", "17b"])
    ap.add_argument("--init", default=None, help="hidden_{TAG}.pt to start from (else ridge init)")
    ap.add_argument("--lora-rank", type=int, default=0)
    ap.add_argument("--lengths", default="256", help="training prompt lengths, comma separated")
    ap.add_argument("--eval-lengths", default="256,512,1024,2048")
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", required=True)
    ap.add_argument("--wd", type=float, default=0.0, help="weight decay on the LoRA group")
    ap.add_argument("--corpus", default="wt2", choices=["wt2", "wt103", "platy", "mixqa"],
                    help="wt103: train on 8%% of wikitext-103 (~8M tokens) so a 10M LoRA cannot memorise it")
    a = ap.parse_args(); Lc = a.cut
    lens = [int(x) for x in a.lengths.split(",")]; elens = [int(x) for x in a.eval_lengths.split(",")]

    m4, tok = load_sdpa(C.PREFILL_MODEL); m17, _ = load_sdpa(C.DECODE_MODEL)

    # --- stitches: from checkpoint, or ridge init exactly as residual_cut.py does it
    ck = torch.load(f"{OUT}/v2_k1_full_h0{'' if a.source == '4b' else '_self'}.pt")
    Lt = len(ck["src_map"])
    kvbr = BridgeV2(ck["src_map"], torch.zeros(Lt, 8, 256, 256), torch.zeros(Lt, 8, 256), 0, parse_hidden("0", Lt))
    kvbr.load_state_dict(ck["state"]); kvbr = kvbr.to(DEV).float()
    if a.init:
        ci = torch.load(f"{OUT}/hidden_{a.init}.pt"); assert ci["cut"] == Lc, ci["cut"]
        js = ci["src_layer"]; hbr = HiddenStitch(ci["h"]["W"], ci["h"]["b"]).to(DEV).float()
        kvbr.load_state_dict(ci["kv"]); print(f"[{a.tag}] init from hidden_{a.init}.pt (src layer {js})", flush=True)
    else:
        calib = torch.load(f"{OUT}/calib_ids.pt")
        H4, H17 = collect_hidden(m4, m17, calib, Lc, 64)
        if a.source == "4b":
            best = (-1, None, None, -1e9)
            for j in range(H4.shape[0]):
                W, b, r2 = ridge_fit(H4[j], H17)
                if r2 > best[3]: best = (j, W, b, r2)
            js, Wh, bh, _ = best
        else:
            js, Wh, bh = Lc - 1, torch.eye(H17.shape[1], device=DEV), torch.zeros(H17.shape[1], device=DEV)
        del H4, H17; hbr = HiddenStitch(Wh, bh).to(DEV).float()

    # --- LoRA on the reader layers only (injected in place; m17 keeps its structure)
    lora = []
    if a.lora_rank:
        from peft import LoraConfig, get_peft_model
        get_peft_model(m17, LoraConfig(r=a.lora_rank, lora_alpha=a.lora_rank, lora_dropout=0.0,
                                       target_modules=TARGETS, layers_to_transform=list(range(Lc, 28))))
        lora = [p for p in m17.parameters() if p.requires_grad]
        for p in lora: p.data = p.data.float()
    n_st = sum(p.numel() for p in hbr.parameters()) + sum(p.numel() for p in kvbr.layers[:Lc].parameters())
    n_lo = sum(p.numel() for p in lora)
    print(f"[{a.tag}] cut {Lc} source {a.source} stitch {n_st/1e6:.1f}M  lora {n_lo/1e6:.1f}M  "
          f"train lengths {lens}  steps {a.steps}", flush=True)

    # --- data
    ev = {P: C.load_text_chunks(tok, N_EVAL, P + CONT, split="test", seed=0) for P in elens}
    tr, va = {}, {}
    for P in lens:
        if a.corpus == "wt2":
            allc = C.load_text_chunks(tok, N_TRAIN[P] + N_VAL[P], P + CONT, split="train", seed=777)
        else:
            allc = load_corpus_chunks(tok, a.corpus, N_TRAIN[P] + N_VAL[P], P + CONT, seed=777)
        tr[P], va[P] = allc[:N_TRAIN[P]], allc[N_TRAIN[P]:]

    def full_eval(label):
        res = {}
        for P in elens:
            nll = evaluate(m4, m17, kvbr, hbr, ev[P], P, Lc, js, a.source); res[P] = {"ppl": ppl(nll), "nll": nll}
        print(f"  {label}: " + "  ".join(f"P{P} {res[P]['ppl']:.3f}" for P in elens), flush=True)
        return res

    base = {}
    for P in elens:
        b17, b4 = baseline(m17, ev[P], P), baseline(m4, ev[P], P)
        base[P] = {"17b": ppl(b17), "4b": ppl(b4), "nll_17b": b17, "nll_4b": b4}
    print("  baselines: " + "  ".join(f"P{P} 1.7B {base[P]['17b']:.3f} / 4B {base[P]['4b']:.3f}" for P in elens), flush=True)
    r0 = full_eval("step 0")

    # --- train
    params = list(hbr.parameters()) + list(kvbr.layers[:Lc].parameters())
    torch.manual_seed(a.seed)
    groups = [{"params": params, "lr": a.lr, "weight_decay": 0.0}] + \
             ([{"params": lora, "lr": a.lora_lr, "weight_decay": a.wd}] if lora else [])
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, a.steps))
    t0, run = time.time(), None
    for step in range(a.steps):
        P = lens[torch.randint(0, len(lens), (1,)).item()]
        ids = tr[P][torch.randint(0, len(tr[P]), (1,)).item()].unsqueeze(0).to(DEV)
        loss = pipeline_nll(m4, m17, kvbr, hbr, ids, P, Lc, js, a.source)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params + lora, 1.0); opt.step(); sched.step()
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if step % 250 == 0:
            print(f"  [{step:4d}] lm ema={run:.4f} train-ppl~{math.exp(run):6.2f}  {time.time()-t0:.0f}s", flush=True)
        if step > 0 and step % 1000 == 0:
            print("        -> held-out train: " + "  ".join(
                f"P{P} {ppl(evaluate(m4, m17, kvbr, hbr, va[P], P, Lc, js, a.source)):.3f}" for P in lens), flush=True)
    rF = full_eval("final") if a.steps else r0

    print(f"\n[{a.tag}] RESULT cut={Lc} source={a.source} lora={a.lora_rank} lengths={lens}  " +
          "  ".join(f"P{P}: {r0[P]['ppl']:.3f}->{rF[P]['ppl']:.3f} (1.7B {base[P]['17b']:.2f})" for P in elens), flush=True)
    json.dump({"tag": a.tag, "cut": Lc, "source": a.source, "src_layer": js, "init": a.init, "lora_rank": a.lora_rank,
               "train_lengths": lens, "steps": a.steps, "corpus": a.corpus, "wd": a.wd, "lora_lr": a.lora_lr, "stitch_M": n_st / 1e6, "lora_M": n_lo / 1e6,
               "step0": {P: r0[P]["ppl"] for P in elens}, "final": {P: rF[P]["ppl"] for P in elens},
               "nll_final": {P: rF[P]["nll"] for P in elens}, "base": base},
              open(f"{C.RESULTS}/hv2_{a.tag}.json", "w"), indent=2)
    torch.save({"h": hbr.state_dict(), "kv": kvbr.state_dict(), "cut": Lc, "src_layer": js,
                "lora": {n: p.detach().cpu() for n, p in m17.named_parameters() if p.requires_grad}},
               f"{OUT}/hv2_{a.tag}.pt")


if __name__ == "__main__":
    main()
