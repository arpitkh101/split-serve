"""Shared utilities: model loading, wikitext chunks, KV capture with RoPE stripped from the
keys (so a fitted map is position-free), RoPE re-application, cache injection, scoring.
Paths: RESULTS holds small json results (committed), WORK holds large intermediates and
checkpoints (set SPLITSERVE_RESULTS / SPLITSERVE_WORK to override).
"""
import os, json, math, time
import torch
import common as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.environ.get("SPLITSERVE_RESULTS", os.path.join(ROOT, "out"))   # small json results, committed
WORK = os.environ.get("SPLITSERVE_WORK", "/dev/shm/splitserve")             # large intermediates and checkpoints
PREFILL_MODEL = "Qwen/Qwen3-4B"
DECODE_MODEL  = "Qwen/Qwen3-1.7B"


def load_model(name, device="cuda:0", dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, attn_implementation="eager",
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def model_dims(model):
    c = model.config
    return dict(
        n_layers=c.num_hidden_layers,
        hidden=c.hidden_size,
        n_q_heads=c.num_attention_heads,
        n_kv_heads=c.num_key_value_heads,
        head_dim=getattr(c, "head_dim", c.hidden_size // c.num_attention_heads),
        rope_theta=getattr(c, "rope_theta", None),
        vocab=c.vocab_size,
    )


# ---------------------------------------------------------------- RoPE utils

def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rope_cos_sin(model, seq_len, device, dtype=torch.float32):
    """(cos, sin) of shape [1, T, head_dim] for positions 0..T-1."""
    pos = torch.arange(seq_len, device=device).unsqueeze(0)
    dummy = torch.zeros(1, seq_len, 1, device=device, dtype=model.dtype)
    cos, sin = model.model.rotary_emb(dummy, pos)
    return cos.to(dtype), sin.to(dtype)


def apply_rope(k, cos, sin):
    """k: [B, H, T, D];  cos/sin: [B, T, D] -> rotate by +theta."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return (k * cos) + (_rotate_half(k) * sin)


def strip_rope(k, cos, sin):
    """Inverse of apply_rope: rotate by -theta, recovering the pre-RoPE key."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return (k * cos) - (_rotate_half(k) * sin)


# ---------------------------------------------------------------- KV capture

def _cache_layers(past):
    """Return [(K, V), ...] per layer, across transformers Cache API versions."""
    if hasattr(past, "key_cache") and past.key_cache:          # <= 4.53
        return list(zip(past.key_cache, past.value_cache))
    if hasattr(past, "layers"):                                 # >= 4.54
        return [(l.keys, l.values) for l in past.layers]
    return [(k, v) for k, v in past]                            # legacy tuple


@torch.no_grad()
def capture_kv(model, input_ids, strip=True, want_logits=False):
    """Prefill `input_ids` and return per-layer (K, V) in float32.

    With strip=True the keys are RoPE-inverted, so K[t] no longer depends on
    the absolute position t -- the space a position-free map should be fit in.
    """
    out = model(input_ids=input_ids, use_cache=True)
    layers = _cache_layers(out.past_key_values)
    T = input_ids.shape[1]
    if strip:
        cos, sin = rope_cos_sin(model, T, input_ids.device)
    kv = []
    for K, V in layers:
        K = K.float()
        if strip:
            K = strip_rope(K, cos, sin)
        kv.append((K, V.float()))
    return (kv, out.logits) if want_logits else kv


# ------------------------------------------------------- hidden-state probe

@torch.no_grad()
def layer_logit_lens(model, input_ids):
    """Next-token distribution read off every layer via the model's own head.

    Used only by the KL layer-selection metric.  Returns log-probs of shape
    [n_layers, T, vocab] on CPU (float16 to keep it affordable).
    """
    out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    norm, head = model.model.norm, model.lm_head
    res = []
    for h in out.hidden_states[1:]:                 # skip the embedding output
        logits = head(norm(h))
        res.append(torch.log_softmax(logits.float(), dim=-1)[0].half().cpu())
    return torch.stack(res)


# ------------------------------------------------------------------ dataset

def load_text_chunks(tokenizer, n_chunks, chunk_len, split="test", seed=0):
    """Tokenize wikitext-2 and cut it into `n_chunks` blocks of `chunk_len`."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    need = n_chunks * chunk_len
    assert ids.numel() >= need, f"corpus has {ids.numel()} tokens, need {need}"
    g = torch.Generator().manual_seed(seed)
    starts = torch.randperm(ids.numel() - chunk_len, generator=g)[:n_chunks]
    return torch.stack([ids[s : s + chunk_len] for s in starts.tolist()])
