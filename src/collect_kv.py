"""Collect paired, RoPE-stripped KV tensors from both models on a calibration set.
Both models see the same token ids at the same positions, so row t of each dump describes the
same token; that pairing is what makes a regression between them meaningful. Calibration uses
the wikitext-2 train split, evaluation always uses test. Output per model:
fp16 [n_layers, n_tokens, n_kv_heads, 2*head_dim], keys (RoPE removed) and values concatenated.
"""
import sys, time, torch
import common as C

DEV = "cuda:0"
N_CHUNKS, CHUNK = 64, 512
OUT = C.WORK


@torch.no_grad()
def collect(repo, chunks, tag):
    model, _ = C.load_model(repo, DEV)
    d = C.model_dims(model)
    L, H, D = d["n_layers"], d["n_kv_heads"], d["head_dim"]
    N = chunks.shape[0] * chunks.shape[1]
    buf = torch.empty(L, N, H, 2 * D, dtype=torch.float16)
    t0, off = time.time(), 0
    for ci, ch in enumerate(chunks):
        ids = ch.unsqueeze(0).to(DEV)
        kv = C.capture_kv(model, ids, strip=True)
        T = ids.shape[1]
        for li, (K, V) in enumerate(kv):
            # K,V: [1,H,T,D] -> [T,H,2D]
            buf[li, off:off + T] = torch.cat([K[0], V[0]], dim=-1) \
                                        .permute(1, 0, 2).half().cpu()
        off += T
        if ci % 16 == 0:
            print(f"    {tag} chunk {ci}/{len(chunks)}  {time.time()-t0:.0f}s", flush=True)
    path = f"{OUT}/kv_{tag}.pt"
    torch.save({"kv": buf, "dims": d}, path)
    print(f"  {tag}: {tuple(buf.shape)} -> {path} "
          f"({buf.numel()*2/1e9:.2f} GB, {time.time()-t0:.0f}s)", flush=True)
    del model, buf
    torch.cuda.empty_cache()


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.DECODE_MODEL)
    chunks = C.load_text_chunks(tok, N_CHUNKS, CHUNK, split="train", seed=1234)
    torch.save(chunks, f"{OUT}/calib_ids.pt")
    print(f"calibration: {N_CHUNKS} x {CHUNK} = {N_CHUNKS*CHUNK} tokens "
          f"(wikitext-2 TRAIN)", flush=True)
    collect(C.PREFILL_MODEL, chunks, "4b")
    collect(C.DECODE_MODEL, chunks, "17b")


if __name__ == "__main__":
    main()
