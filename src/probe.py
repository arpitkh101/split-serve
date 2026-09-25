"""Sanity checks before any experiment: both models' dims and cache shapes, the RoPE
strip/apply round trip, cache injection (a model fed its own captured cache must reproduce an
uninterrupted forward), and standalone continuation perplexity for both models.
"""
import sys, json, torch
import common as C

DEV = "cuda:0"
CHUNK, PROMPT = 512, 256


def build_cache(kv_stripped, model, dtype):
    """[(K_noRoPE, V)] -> a DynamicCache with RoPE re-applied at positions 0..T-1."""
    from transformers import DynamicCache
    T = kv_stripped[0][0].shape[2]
    cos, sin = C.rope_cos_sin(model, T, DEV)
    cache = DynamicCache()
    for i, (K, V) in enumerate(kv_stripped):
        Kr = C.apply_rope(K.to(DEV).float(), cos, sin).to(dtype)
        cache.update(Kr, V.to(DEV).to(dtype), i, {})
    return cache


@torch.no_grad()
def continuation_nll(model, chunk_ids, prompt_len, cache=None):
    """Mean NLL over the tokens after `prompt_len`.

    cache=None  -> the model prefills the prompt itself (standalone baseline).
    cache given -> the model decodes on an imported/mapped KV cache.

    Both paths score EXACTLY the same targets: tokens prompt_len+1 .. end.
    The token at prompt_len is excluded because on the cache path it is
    predicted by the cache's owner, not by the model under test.
    """
    ids = chunk_ids.unsqueeze(0).to(DEV)
    if cache is None:
        out = model(input_ids=ids, use_cache=False)
        logits = out.logits[0, prompt_len:-1]
    else:
        cont = ids[:, prompt_len:]
        pos = torch.arange(prompt_len, ids.shape[1], device=DEV).unsqueeze(0)
        out = model(input_ids=cont, past_key_values=cache,
                    position_ids=pos, cache_position=pos[0], use_cache=True)
        logits = out.logits[0, :-1]
    tgt = ids[0, prompt_len + 1:]
    assert logits.shape[0] == tgt.numel(), (logits.shape, tgt.shape)
    lp = torch.log_softmax(logits.float(), -1)
    return -lp[torch.arange(tgt.numel()), tgt].mean().item()


def main():
    torch.manual_seed(0)
    print("=" * 72); print("A. ARCHITECTURE"); print("=" * 72)
    m4, tok4 = C.load_model(C.PREFILL_MODEL, DEV)
    m17, tok17 = C.load_model(C.DECODE_MODEL, DEV)
    d4, d17 = C.model_dims(m4), C.model_dims(m17)
    for k in d4:
        flag = "  <-- SAME" if d4[k] == d17[k] else ""
        print(f"  {k:12s}  4B={str(d4[k]):>8s}   1.7B={str(d17[k]):>8s}{flag}")
    kvw4 = d4["n_kv_heads"] * d4["head_dim"]
    kvw17 = d17["n_kv_heads"] * d17["head_dim"]
    print(f"\n  per-layer KV width:  4B={kvw4}   1.7B={kvw17}   "
          f"{'MATCHED' if kvw4 == kvw17 else 'MISMATCHED'}")
    print(f"  same tokenizer vocab: {tok4.vocab_size == tok17.vocab_size} "
          f"({tok4.vocab_size})")

    chunks = C.load_text_chunks(tok17, 128, CHUNK)
    ids = chunks[0].unsqueeze(0).to(DEV)

    print("\n" + "=" * 72); print("B. RoPE ROUND-TRIP"); print("=" * 72)
    for name, m in (("4B", m4), ("1.7B", m17)):
        raw = C.capture_kv(m, ids, strip=False)
        strp = C.capture_kv(m, ids, strip=True)
        cos, sin = C.rope_cos_sin(m, ids.shape[1], DEV)
        back = C.apply_rope(strp[0][0], cos, sin)
        err = (back - raw[0][0]).abs().max().item()
        scale = raw[0][0].abs().max().item()
        print(f"  {name:5s} K shape {tuple(raw[0][0].shape)}  "
              f"max|reroped-orig|={err:.3e}  (K scale {scale:.2f})  "
              f"rel={err/scale:.2e}")

    print("\n" + "=" * 72); print("C. CACHE INJECTION FIDELITY"); print("=" * 72)
    print("  feeding each model its own captured KV must reproduce its logits")
    for name, m in (("4B", m4), ("1.7B", m17)):
        full = continuation_nll(m, chunks[0], PROMPT, cache=None)
        kv = C.capture_kv(m, ids[:, :PROMPT], strip=True)
        inj = continuation_nll(m, chunks[0], PROMPT,
                               cache=build_cache(kv, m, m.dtype))
        print(f"  {name:5s} standalone NLL={full:.4f}   via-own-cache NLL={inj:.4f}"
              f"   delta={abs(full-inj):.5f}")

    print("\n" + "=" * 72); print("D. BASELINE CONTINUATION PERPLEXITY"); print("=" * 72)
    print(f"  wikitext-2, {len(chunks)} chunks of {CHUNK} tok, "
          f"scored on the last {CHUNK-PROMPT}")
    res = {}
    for name, m in (("Qwen3-4B", m4), ("Qwen3-1.7B", m17)):
        nlls = [continuation_nll(m, c, PROMPT) for c in chunks]
        ppl = torch.tensor(nlls).mean().exp().item()
        res[name] = ppl
        print(f"  {name:12s} ppl = {ppl:.3f}")
    gap = res["Qwen3-1.7B"] - res["Qwen3-4B"]
    print(f"\n  GAP TO CLOSE: {gap:.3f} ppl "
          f"({100*gap/res['Qwen3-1.7B']:.1f}% of the 1.7B's perplexity)")
    json.dump(res, open(f"{C.RESULTS}/baselines.json", "w"), indent=2)


if __name__ == "__main__":
    main()
