"""Train the KV stitch to match the 4B's next-token distribution instead of the text:
KL(P_4B || P_1.7B off the stitched cache). --source 4b reads the 4B's cache, --source 17b the
1.7B's own (the control). If the 4B's cache carries knowledge the text signal cannot supply,
the first should match the teacher better than the second.
"""
import sys, json, math, time, warnings, argparse, torch
import torch.nn as nn
warnings.filterwarnings("ignore")
import common as C
from probe import continuation_nll
from ridge_sweep import fit_ridge_blocked
from kv_stitch_mlp import make_cache
from kv_stitch import BridgeV2, parse_hidden

DEV, OUT = "cuda:0", C.WORK
CHUNK, PROMPT, N_EVAL = 512, 256, 64


@torch.no_grad()
def teacher_logp(m4, ids):
    """4B next-token log-probs for tokens PROMPT+1..end, [255, V] fp32."""
    return torch.log_softmax(m4(input_ids=ids, use_cache=False).logits[0, PROMPT:-1].float(), -1)


def student_logits(bridge, m17, kv_src, ids, grad):
    cache = make_cache(bridge, kv_src, m17, PROMPT, grad=grad)
    cont = ids[:, PROMPT:]
    pos = torch.arange(PROMPT, ids.shape[1], device=DEV).unsqueeze(0)
    return m17(input_ids=cont, past_key_values=cache, position_ids=pos,
               cache_position=pos[0], use_cache=True).logits[0, :-1].float()


def kl_loss(t_logp, s_logits):
    s_logp = torch.log_softmax(s_logits, -1)
    return (t_logp.exp() * (t_logp - s_logp)).sum(-1).mean()


@torch.no_grad()
def evaluate(bridge, src_model, m4, m17, chunks):
    bridge.eval(); nll, kl = [], []
    for ch in chunks:
        ids = ch.unsqueeze(0).to(DEV)
        kv = C.capture_kv(src_model, ids[:, :PROMPT], strip=True)
        s = student_logits(bridge, m17, kv, ids, grad=False)
        tgt = ids[0, PROMPT + 1:]
        nll.append(-torch.log_softmax(s, -1)[torch.arange(tgt.numel()), tgt].mean().item())
        kl.append(kl_loss(teacher_logp(m4, ids), s).item())
    bridge.train()
    return torch.tensor(nll).mean().exp().item(), sum(kl) / len(kl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="4b", choices=["4b", "17b"])
    ap.add_argument("--loss", default="kl4b", choices=["kl4b", "lm"])
    ap.add_argument("--k", type=int, default=1); ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-only", default="", help="evaluate an existing checkpoint tag")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    R2 = torch.load(f"{OUT}/affinity.pt")["R2"]; Ls, Lt = R2.shape
    Ycpu = torch.load(f"{OUT}/kv_17b.pt")["kv"]
    if a.source == "4b":
        Xcpu = torch.load(f"{OUT}/kv_4b.pt")["kv"]
        src_map = [R2[:, j].topk(a.k).indices.tolist() for j in range(Lt)]
    else:
        Xcpu = Ycpu
        st = lambda j: max(0, min(Lt - a.k, j - a.k // 2))
        src_map = [list(range(st(j), st(j) + a.k)) for j in range(Lt)]
    W, b, r2 = fit_ridge_blocked(src_map, Xcpu, Ycpu); del Xcpu, Ycpu
    bridge = BridgeV2(src_map, W.float(), b.float(), 0, parse_hidden("0", Lt)).to(DEV).float()
    if a.eval_only:
        bridge.load_state_dict(torch.load(f"{OUT}/{a.eval_only}.pt")["state"])

    m4, tok = C.load_model(C.PREFILL_MODEL, DEV); m17, _ = C.load_model(C.DECODE_MODEL, DEV)
    for p in m17.parameters(): p.requires_grad_(False)
    src_model = m4 if a.source == "4b" else m17
    eval_chunks = C.load_text_chunks(tok, N_EVAL, CHUNK, split="test", seed=0)
    all_tr = C.load_text_chunks(tok, 3032, CHUNK, split="train", seed=777)
    train_chunks, val_chunks = all_tr[:-32], all_tr[-32:]

    n_par = sum(p.numel() for p in bridge.parameters())
    cfg = f"source={a.source} loss={a.loss} k={a.k} seed={a.seed}"
    print(f"[{a.tag}] {cfg}  params={n_par/1e6:.1f}M", flush=True)
    p0, kl0 = evaluate(bridge, src_model, m4, m17, eval_chunks)
    print(f"  step 0     test ppl {p0:8.3f}   KL-to-4B {kl0:.4f}", flush=True)
    if a.eval_only:
        json.dump({"tag": a.tag, "eval_of": a.eval_only, "source": a.source, "ppl": p0, "kl_to_4b": kl0},
                  open(f"{C.RESULTS}/distill_{a.tag}.json", "w"), indent=2)
        return

    torch.manual_seed(a.seed)
    opt = torch.optim.AdamW(bridge.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    t0, run = time.time(), None
    for step in range(a.steps):
        ids = train_chunks[torch.randint(0, len(train_chunks), (1,)).item()].unsqueeze(0).to(DEV)
        with torch.no_grad():
            kv = C.capture_kv(src_model, ids[:, :PROMPT], strip=True)
            t_logp = teacher_logp(m4, ids) if a.loss == "kl4b" else None
        s = student_logits(bridge, m17, kv, ids, grad=True)
        loss = kl_loss(t_logp, s) if a.loss == "kl4b" else \
               nn.functional.cross_entropy(s, ids[0, PROMPT + 1:])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0); opt.step(); sched.step()
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if step % 250 == 0:
            print(f"  [{step:4d}] {a.loss} ema={run:.4f}  {time.time()-t0:.0f}s", flush=True)
        if step > 0 and step % 1000 == 0:
            pv, kv_ = evaluate(bridge, src_model, m4, m17, val_chunks)
            print(f"        -> held-out train ppl {pv:.3f}  KL {kv_:.4f}", flush=True)
        del kv, t_logp
    pB, klB = evaluate(bridge, src_model, m4, m17, eval_chunks)
    print(f"\n[{a.tag}] RESULT {cfg}  test ppl {pB:.3f}   KL-to-4B {klB:.4f}   (step0 {p0:.3f} / {kl0:.4f})")
    json.dump({"tag": a.tag, "source": a.source, "loss": a.loss, "k": a.k, "seed": a.seed,
               "params_M": n_par / 1e6, "step0_ppl": p0, "step0_kl": kl0, "ppl": pB, "kl_to_4b": klB},
              open(f"{C.RESULTS}/distill_{a.tag}.json", "w"), indent=2)
    torch.save({"state": bridge.state_dict(), "src_map": src_map, "rank": 0, "hidden": "0", "k": a.k},
               f"{OUT}/distill_{a.tag}.pt")


if __name__ == "__main__":
    main()
