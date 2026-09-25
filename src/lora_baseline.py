"""Fair adaptation baselines: a LoRA on each model, trained on the same chunks for the same
steps with the same loss, doing its own prefill. The adapted 1.7B is the bar a bridge has to
beat; the adapted 4B is the ceiling.
"""
import sys, json, math, time, warnings, argparse, torch
import torch.nn as nn
warnings.filterwarnings("ignore")
import common as C
from probe import continuation_nll
from peft import LoraConfig, get_peft_model

DEV = "cuda:0"
CHUNK, PROMPT, N_EVAL = 512, 256, 64
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def ppl_on(model, chunks):
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        nll = [continuation_nll(model, c, PROMPT) for c in chunks]
    model.train()
    return torch.tensor(nll).mean().exp().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="17b", choices=["17b", "4b"])
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--train-chunks", type=int, default=3000)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()

    repo = C.DECODE_MODEL if a.model == "17b" else C.PREFILL_MODEL
    model, tok = C.load_model(repo, DEV)
    eval_chunks = C.load_text_chunks(tok, N_EVAL, CHUNK, split="test", seed=0)
    all_tr = C.load_text_chunks(tok, a.train_chunks + 32, CHUNK, split="train", seed=777)
    train_chunks, val_chunks = all_tr[:-32], all_tr[-32:]      # identical to the stitch runs

    p_before = ppl_on(model, eval_chunks)
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.rank,
                                             lora_dropout=0.0, target_modules=TARGETS))
    trainable = [p for p in model.parameters() if p.requires_grad]
    for p in trainable:
        p.data = p.data.float()                   # fp32 master weights, bf16 compute
    n_tr = sum(p.numel() for p in trainable)
    print(f"LoRA r={a.rank} on {repo}: {n_tr/1e6:.1f}M trainable params")
    print(f"  before adaptation   test ppl = {p_before:8.3f}", flush=True)

    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    model.train(); t0 = time.time(); run = None
    for step in range(a.steps):
        ids = train_chunks[torch.randint(0, len(train_chunks), (1,)).item()] \
                  .unsqueeze(0).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, use_cache=False).logits[0, PROMPT:-1]
        loss = nn.functional.cross_entropy(logits.float(), ids[0, PROMPT + 1:])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); sched.step()
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if step % 100 == 0:
            print(f"  [{step:4d}] lm={loss.item():.4f} ema={run:.4f} "
                  f"train-ppl~{math.exp(run):6.2f}  {time.time()-t0:.0f}s", flush=True)
        if a.val_every and step > 0 and step % a.val_every == 0:
            print(f"        -> held-out train ppl={ppl_on(model, val_chunks):8.3f}"
                  f"   test ppl={ppl_on(model, eval_chunks[:16]):8.3f}", flush=True)

    p_after = ppl_on(model, eval_chunks)
    print("\n" + "=" * 70)
    print(f"  {repo}  before {p_before:8.3f}   after LoRA {p_after:8.3f}")
    print("=" * 70)
    json.dump({"model": repo, "rank": a.rank, "trainable_M": n_tr / 1e6,
               "before": p_before, "after": p_after},
              open(f"{C.RESULTS}/lora_{a.model}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
