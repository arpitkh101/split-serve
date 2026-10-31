# SplitServe

Disaggregated LLM inference across two model sizes. Qwen3-4B runs the prefill, Qwen3-1.7B
runs the decode, and a learned stitch carries the prefiller's activations into the decoder
so the decoder never touches the prompt. Every number in this repo was measured; the
write-up of how the design got here, including what failed, is in [learning.md](learning.md).

## The idea

Prefill (reading the prompt) is compute-bound; decode (producing tokens) is
memory-bandwidth-bound. Serving them on one GPU means a long prompt arriving mid-stream
stalls every running request. Splitting them across GPUs removes that tail. This project
asks a harder version of the question: can the prefill be done by a *different, larger*
model, so the decoder inherits a cache it never computed?

The two models have the same KV cache shape (8 KV heads x 128), but what their layers
represent is different, so the 4B's cache is meaningless to the 1.7B as-is: passing it
straight through gives perplexity 34,000. Something has to translate.

## The design

```
prompt ──> Qwen3-4B prefill (36 layers, residual stream exposed)
               │
               │  28 linear maps, one per 1.7B layer: 4B residual at its
               │  best-matching layer (2,560) ──> 1.7B residual (2,048)
               ▼
           Qwen3-1.7B's own input_layernorm, k_proj, v_proj, RoPE
               │  build the whole KV cache from the mapped residuals
               ▼
           4B's own logits emit token 1 ──> 1.7B decodes from token 2 off that cache
```

The stitch is 28 linear maps (152M parameters), ridge-initialised and trained end to end
on the decoder's next-token loss through the frozen decoder. It runs as one CUDA graph in
3 / 11 / 43 ms at 256 / 1k / 4k prompt tokens, 5 to 7 percent of the 4B prefill. The
decoder does zero prompt compute.

## Headline results

1,000 examples per task, paired bootstrap confidence intervals (10,000 resamples) against
the control: the same maps and training fed the 1.7B's own activations.

| task | 1.7B alone | pipeline | control | pipeline minus control [95% CI] |
|---|---|---|---|---|
| ARC-Easy (5-shot) | 79.9 | 79.4 | 79.1 | +0.3 [-1.8, +2.4] |
| ARC-Challenge (5-shot) | 49.4 | 48.7 | 48.3 | +0.4 [-2.4, +3.1] |
| PIQA (5-shot) | 71.6 | 72.2 | 73.6 | -1.4 [-3.4, +0.6] |
| LAMBADA (last word) | 51.2 | 59.4 | 53.3 | measures the 4B, which emits that token |
| wikitext-2 ppl, 256-token prompt | 19.13 | 11.32 | 11.23 | within noise at every length |

Interference, fixed arrival rate of 2 to 4 prefills/s, decoder at batch 8: a colocated
prefill costs the decoder 8 to 33 percent of throughput and a 40 to 55 percent p99 tail;
with prefill on the other GPU there is no tail and under 4 percent cost.

**Context handoff at long context.** The setting where the bridge saves work: the 4B's
reading of a long context already exists (a previous turn, a shared prefix, another agent)
and the 1.7B continues from it instead of re-reading. Multi-document QA (HotpotQA distractor
padded to 2k to 32k tokens), short greedy answers, paired bootstrap against the 1.7B
re-prefilling the same prompt. Stitch trained with attention matching on a mix of four QA
train splits at 2k to 8k. Timing is batch 1, bf16, both models on flash attention, the
re-prefill keeping only the last position's logits as a server does; the bridge is one
CUDA graph.

| context | n | 1.7B re-prefill EM / F1 | pipeline EM / F1 | EM gap [95% CI] | bridge vs re-prefill |
|---|---|---|---|---|---|
| 2k | 500 | 38.8 / 52.1 | 44.2 / 58.2 | +5.4 [+1.4, +9.6] | 29 ms vs 119 ms, 4.2x |
| 4k | 500 | 36.2 / 50.5 | 37.6 / 51.9 | +1.4 [-2.6, +5.4] | 56 ms vs 246 ms, 4.4x |
| 8k | 500 | 35.6 / 48.5 | 33.0 / 45.0 | -2.6 [-6.6, +1.6] | 109 ms vs 535 ms, 4.9x |
| 16k | 200 | 27.0 / 39.9 | 29.0 / 41.4 | +2.0 [-4.0, +8.0] | 219 ms vs 1,310 ms, 6.0x |
| 32k | 200 | 27.5 / 37.5 | 24.5 / 37.0 | -3.0 [-8.5, +2.0] | 436 ms vs 3,622 ms, 8.3x |

Parity with re-prefill from 2k to 32k, four times the training length, with the control
(same maps and objective on the 1.7B's own activations) drifting by the same amount at
8k and 32k. On held-out sets the picture splits by question style. TriviaQA with its
Wikipedia pages, never trained on, is at parity: +1.2 / -2.4 / +0.8 EM at 2k / 4k / 8k
against re-prefill, every interval spanning zero, with the pages worth 36 EM over closed
book. Span questions over one passage are not: 4 to 9 EM under re-prefill on DuoRC, 6 to
14 under on multi-turn SQuAD and 16 to 23 under on SQuAD single-turn. Giving the pipeline the native 1.7B's first token (an
oracle) halves those gaps; swapping native keys and values into the stitched cache block
by block puts the other half in layers 12 to 23. A native question suffix, ridge-only
maps, attention matching at prompt positions, two source layers in that band and a mixed
first token were each tried and none moved it (learning.md section 18).

## How it got here

Each step was decided by the result of the one before. Details and tables in learning.md.

1. **Harness first.** A model fed its own captured cache reproduces an uninterrupted
   forward to 0.001 nats. Every later failure is the map, not the plumbing.
2. **Closed-form KV bridge.** Per-head ridge regression, 4B cache to 1.7B cache, RoPE
   stripped and re-applied. Best case 25 percent worse than the 1.7B alone. Choosing
   source layers by R2 beats KL matching and even spacing.
3. **Trained KV stitch, and the control that decides everything.** Trained on next-token
   loss the stitch scores 12.4 against 19.1 untrained, better than the 4B itself. The same
   stitch fed the 1.7B's *own* cache scores 11.3. Four seeds under LM loss, two under
   distillation: the large model's cache never improves the small decoder over its own
   prefill. Most of a trained bridge's gain is adaptation.
4. **Interference and throughput.** Disaggregation removes the decode latency tail at
   every arrival rate measured. At one prefill GPU per decode GPU it delivers about half
   the tokens per second of two independent replicas.
5. **Residual stitch with a native tail.** Map the 4B's residual stream into the 1.7B's at
   a cut layer and let the 1.7B run its last layers itself. Gap to control 1.0 to 0.4 ppl.
6. **Tasks, not perplexity.** That 0.4 ppl gap was hiding a 12-point loss on
   ARC-Challenge. Eight training recipes moved perplexity and nothing else. A component
   swap localised the loss: ARC blamed the early-layer KV maps, LAMBADA the hand-over
   residual at the final positions.
7. **Residual-space stitch.** Remove KV-to-KV maps entirely; map residuals at every early
   layer and let the 1.7B's own projections build the cache. ARC-Challenge 34.6 to 42.4.
8. **Patches, then their removal.** A 64-token native suffix reached parity on three
   tasks but cost half a native prefill on short prompts. Dropping the tail and suffix
   exposed a hole in the training objective (the prompt's last-position logit was never
   scored, and the native layers had been repairing it silently). With the objective
   fixed, everything the decoder reads off the cache transfers; only the last position's
   output state does not, by any map tried, including a causal attention branch over the
   source positions. The prefiller emits that token anyway. Parity, one mechanism.
9. **The handoff test, and retrieval.** On long multi-document QA the wikitext-trained
   stitch lost 14 EM points: the 1.7B could complete an answer the 4B's first token had
   started, but could not find one itself (EM 11 on its own). Three objectives on wikitext
   gave the same result. Training the same maps with an attention-matching loss on QA data
   restored retrieval (EM 35 on its own, parity or better with the 4B's first token at 2k
   and 4k), with the control still at native. Training on prompts up to 8k extended parity
   to 8k; a second seed reproduced it; out of domain the stitch keeps half the gap.
10. **More QA sources.** Training on four QA sources instead of one gained 1 to 4 EM on
   every held-out set and 4 in domain, but left 8 to 17 points under re-prefill at 8k out
   of domain. More data of the same kind is not the lever; the map itself is.

## Findings that transfer beyond this pair

- A bridge's gain has to be measured against a same-budget adaptation of the receiver
  alone. Here adaptation was five times the transfer effect.
- Perplexity on the training domain measures fluency, not whether the context survived.
  A 0.4 ppl gap hid a 12-point task loss.
- Cache-to-cache maps have an intrinsic mismatch: decode queries come from the receiver's
  own residual, keys from a learned map. Mapping the residual stream and letting the
  receiver's own projections build its cache removes it.
- Score every prompt position when training a stitch. A loss that only scores the
  continuation leaves the final position untrained, and any native layers in the pipeline
  will hide that.
- The one representation that does not transfer between the two models is the last
  position's output state (LAMBADA 23 mapped vs 53 native). Route around it: the prefill
  instance emits the first token, as disaggregated serving systems already do.
- Retrieval is a property of keys, and next-token loss on prose never asks for it. A stitch
  that is perfect on perplexity can be useless for finding an answer 2k tokens back. Train on
  data that needs the lookup, with a loss on the attention outputs, and it becomes learnable.

## Repository layout

```
src/
  common.py             model loading, chunks, KV capture with RoPE stripped, cache injection
  probe.py              sanity checks: shapes, RoPE round trip, injection fidelity, baselines
  collect_kv.py         paired calibration caches for both models
  select_layers.py      R2 and KL layer affinity matrices
  ridge_fit.py          closed-form per-head KV bridge and its evaluation
  ridge_sweep.py        source-layer-count sweep with self-map and oracle controls
  layer_damage.py       bridge a prefix or suffix of layers: where the loss lives
  layer_selection.py    even vs KL vs R2 selection at matched k
  kv_stitch_mlp.py      first trained stitch (ridge init + MLP), MSE vs LM loss
  kv_stitch.py          compact KV stitch (rank sweep, one source layer), self-bridge control
  kv_stitch_distill.py  stitch trained to match the 4B's distribution
  lora_baseline.py      LoRA on each model alone, same data and steps
  length_gen.py         stitch trained at 256 tokens evaluated to 2,048
  latency*.py           single-request timing; CUDA-graph bridge timing
  interference.py       two-process prefill/decode interference at fixed arrival rates
  throughput.py         two-GPU throughput: replicas vs same-model split vs 4B split
  residual_cut.py       residual stitch with a native tail (cut layer)
  residual_cut_train.py training levers on it (tail LoRA, mixed lengths, corpora)
  task_eval.py          ARC-E / ARC-C / PIQA / LAMBADA with per-example hits
  diag_task.py          component swap: early KV vs hand-over residual
  residual_stitch.py    residual-space stitch at every layer (the final design)
  residual_suffix.py    wikitext cost of a native suffix
  longctx_eval.py       long-context multi-document QA in the handoff setting, with timings
  longctx_long.py       the same at 16k and 32k: last-logit forwards, flash attention, source layers on the CPU
  latency_fair.py       bridge vs flash-attention last-logit re-prefill, 2k to 32k
  layer_patch.py        native keys and values swapped into the stitched cache block by block
  qa_data.py            QA sources in one format: the training mix and the held-out sets (SQuAD, DuoRC, TriviaQA)
  statetrack.py         synthetic robot state-tracking log, a held-out non-QA test
  closedbook.py         the same questions with no documents, to see what the context is worth
  multiturn_eval.py     multi-turn handoff on SQuAD: the 4B answers turn one, the 1.7B turn two
  paired_ci*.py         paired bootstrap CIs from saved per-chunk NLLs and task hits
out/                    every result as json (perplexities, per-chunk NLLs, task hits, timings)
learning.md             the full account, step by step, failures included
```

Result-file prefixes in `out/`: `v2_*` KV stitch runs, `hidden_*` residual-cut runs,
`hv2_*` trained pipelines with per-chunk NLLs (residual-cut levers and residual-space
stitches share this format), `task_*` task evals, `paired_ci_*` intervals.

## Reproduce

Tested on RTX A5000 (24 GB) with torch 2.8, transformers 4.53, peft 0.18. Models are
resolved through `HF_HOME`. Small results go to `out/`; large intermediates and checkpoints
go to `SPLITSERVE_WORK` (default `/dev/shm/splitserve`).

```bash
export HF_HOME=/path/to/hf_cache SPLITSERVE_WORK=/path/to/scratch
PY=python

# harness, calibration, layer selection, linear bridge
$PY src/probe.py
$PY src/collect_kv.py ; $PY src/select_layers.py
$PY src/ridge_sweep.py ; $PY src/layer_damage.py ; $PY src/layer_selection.py

# trained KV stitch and its controls
$PY src/kv_stitch.py --k 1 --tag k1_full_h0
$PY src/kv_stitch.py --k 1 --tag k1_full_h0_self --source 17b
$PY src/paired_ci_kv.py --tag k1_full_h0
$PY src/lora_baseline.py --model 17b ; $PY src/lora_baseline.py --model 4b
$PY src/kv_stitch_distill.py --source 4b ; $PY src/kv_stitch_distill.py --source 17b
$PY src/length_gen.py ; $PY src/latency_kv_graph.py --tag v2_k1_full_h0

# interference and throughput (two GPUs)
$PY src/interference.py --rate 2 ; $PY src/interference.py --rate 4
$PY src/throughput.py --batch 8 --prompt 1024 --gen 64

# residual stitch with a native tail, tasks, diagnostic
$PY src/residual_cut.py --cut 20 --source 4b --tag cut20_4b
$PY src/residual_cut.py --cut 20 --source 17b --tag cut20_self
$PY src/task_eval.py --tag phase4 --n 1000
$PY src/diag_task.py --n 1000

# residual-space stitch, final design: cut 28, prompt-position loss, 4B emits token 1
$PY src/residual_stitch.py --fit-only --layers 0-28
$PY src/residual_stitch.py --cut 28 --prompt-w 1 --source 4b  --tag pw_4b_c28
$PY src/residual_stitch.py --cut 28 --prompt-w 1 --source 17b --tag pw_self_c28
$PY src/task_eval.py --tag pw_c28 --conds 17b,reskv:pw_4b_c28,reskv:pw_self_c28 --n 1000
$PY src/task_eval.py --tag pw4bfirst_c28 --conds 17b,4b,reskv:pw_4b_c28@4b --n 1000
$PY src/paired_ci.py task pw4bfirst_c28
$PY src/latency_residual.py --tag pw_4b_c28

# handoff setting: attention-matching objective on QA data, long-context evaluation
$PY src/residual_stitch.py --cut 28 --teacher attn --corpus hotpot --qa-lengths 2048,4096,8192 --source 4b  --tag attnqa8k_4b_c28
$PY src/residual_stitch.py --cut 28 --teacher attn --corpus hotpot --qa-lengths 2048,4096,8192 --source 17b --tag attnqa8k_self_c28
$PY src/longctx_eval.py --conds 17b,4b --n 500 --tag handoff_base
$PY src/longctx_eval.py --conds reskv:attnqa8k_4b_c28@4b,reskv:attnqa8k_4b_c28,reskv:attnqa8k_self_c28 --n 500 --tag handoff_attnqa8k
$PY src/paired_ci.py longctx handoff_base handoff_attnqa8k
$PY src/multiturn_eval.py --conds 17b,4b --n 500 --tag mt_base                      # out-of-domain multi-turn handoff on SQuAD
$PY src/multiturn_eval.py --conds reskv:attnqa8k_4b_c28@4b,reskv:attnqa8k_self_c28 --n 500 --tag mt_attnqa8k
$PY src/residual_stitch.py --cut 28 --teacher attn --corpus qamix --qa-lengths 2048,4096,8192 --source 4b --tag qamix_4b_c28   # four QA sources
$PY src/longctx_eval.py --dataset squad --conds 17b,4b,reskv:qamix_4b_c28@4b --n 500 --tag squad_ood   # held out; also --dataset duorc, triviaqa, statetrack
$PY src/closedbook.py --dataset triviaqa --n 500
$PY src/latency_residual.py --tag attnqa_4b_c28 --lengths 2048,4096,8192
$PY src/latency_fair.py --tag qamix_4b_c28                                    # headline timing column
$PY src/longctx_long.py --conds 17b,reskv:qamix_4b_c28@4b,reskv:qamix_self_c28 --lengths 32768 --n 200 --tag handoff_qamix_long32
$PY src/longctx_eval.py --dataset squad --conds reskv:qamix_4b_c28@17b --n 500 --tag squad_ood_oracle   # oracle first token
$PY src/layer_patch.py --n 100 --length 4096
```

## Caveats

- One model pair. The final design is one seed; the residual-space stitch before it
  reproduced on a second seed.
- The long-context parity result is in domain: the stitch was trained on QA train splits
  and evaluated on their validation questions. On held-out QA sets it is 4 to 23 EM under
  re-prefill, and on SQuAD the 4B itself is 8 to 15 under the 1.7B. Two seeds, one model pair.
- The 16k and 32k rows are 200 questions, so their intervals are wide, and at 32k the
  choice of bf16 attention kernel alone moves the re-prefill baseline by 2 to 3 EM;
  every row compares conditions run in the same process with the same kernel.
- With the 4B emitting the first token, LAMBADA measures the 4B, and the ARC and PIQA
  log-likelihood sums mix one 4B log-probability with the 1.7B's.
- Per request the split costs more: about 2.75x the prefill FLOPs of the 1.7B alone and
  3.4x the weights. The benefit is latency isolation, and a prefill service already
  running the larger model is the setting where the heterogeneous version beats a
  same-model split.
- Timing is Hugging Face at batch 1 or 8; a serving engine would change the absolute
  numbers. A torch.compile'd re-prefill is 10 to 15 percent faster than the flash row
  above, which leaves the bridge 3.5x to 7.9x ahead.
