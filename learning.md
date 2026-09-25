# SplitServe: how the design got here

Working notes, kept as the project went. Each section is one step, with what was tried,
what came out, and what it changed. Numbers are wikitext-2 test perplexity on 64 chunks
(256-token prompt, 256-token continuation) unless a task is named; tasks are 1,000
examples each with paired bootstrap confidence intervals. All runs on RTX A5000 24 GB.

## The idea

Serve one model's prefill and another model's decode. Qwen3-4B reads the prompt, Qwen3-1.7B
produces the answer. Two reasons it might be worth doing. Prefill on a separate GPU removes
the latency tail that colocated prefill puts on every running decode. And the larger model
may read the prompt better than the small one, so a decoder that inherits its cache might
answer better than it would from its own prefill.

Both models use 8 KV heads of width 128, so a cache entry is 1,024 numbers per layer in
both. The mismatch is not shape. The two models learned different features, the 4B has 36
layers to the 1.7B's 28, and Qwen3 stores keys after rotary position embedding, so a key
at position 3 and at position 3,000 are rotated differently. Everything below strips RoPE
before mapping and re-applies the receiver's RoPE after; the round trip is exact to bf16.

## 1. Harness first

Before any bridge, the harness had to prove it could do nothing. Feed a model its own
captured cache and it must score exactly as an uninterrupted forward: 0.00000 nats in
fp32, about 0.001 in bf16. The same ridge machinery fitting the 1.7B onto itself recovers
R2 0.9999 and perplexity 19.140 against an oracle of 19.132. This caught a real bug on
day one (the standalone path scored 256 tokens, the cache path 255) and it meant that every
later failure could be blamed on the map, not the plumbing.

Baselines on the 64 test chunks: 1.7B 19.14, 4B 15.98. Anything below 15.98 through a
context transfer is doing something other than transferring context.

## 2. A linear cache bridge

Per-head ridge regression from the 4B's cache to the 1.7B's, per layer, in RoPE-free
space, fitted in closed form on 32,768 calibration tokens from the train split. More
source layers per target help, then stall.

| source layers per target | R2 | ppl |
|---|---|---|
| none (pass the 4B cache through) | | 34,155 |
| 1 | 0.45 | 588 |
| 4 | 0.61 | 67 |
| 8 | 0.68 | 36.1 |
| 24 | 0.78 | 23.9 |
| 1.7B reading its own prompt | 1.00 | 19.1 |

At 24 layers the map explains 78 percent of the target's variance and is still 25 percent
worse than the small model alone. The published version of this recipe keeps most of the
receiver's quality on 3B to 70B pairs; a 1.7B receiver is a harsher regime.

**Choosing source layers.** Three rules, same maps, same data: even spacing, the 4B layer
whose logit-lens distribution is nearest in KL, and the layer with the highest R2. With one
source layer: even 872, KL 1,247, R2 588. By eight layers all three converge (36.2 / 36.9
/ 36.1), because pooling washes out the choice. R2 it is.

**Where the damage lives.** Bridge only the first n or the last n of the 1.7B's layers and
keep the rest as its own cache. Replacing the first 20 costs 2.2 ppl; replacing only the
last 4 costs about the same. Yet the last layers fit better in R2 (0.61 to 0.68) than the
middle (0.55). Reconstruction error is the wrong target. The stitch has to be trained on
the model's actual objective, next-token loss.

## 3. A trained stitch, and the control that decides everything

Initialise at the ridge solution, add a zero-initialised MLP correction, train on the
1.7B's next-token loss with the 1.7B frozen. Perplexity goes to 12.8, better than the 4B
itself at 15.98. A bridge cannot pass on more than its source holds, so this was a red
flag, not a result. With 184M parameters writing the entire cache the decoder attends to,
the stitch can steer the 1.7B toward wikitext the way prefix-tuning does, whatever the 4B
contributed.

The controls that make the number mean something, same data, steps, optimiser and loss:

| condition | trainable | ppl |
|---|---|---|
| stitch, 4B cache to 1.7B | 184M | 12.81 |
| same stitch on the 1.7B's own cache | 184M | 13.88 |
| LoRA on the 1.7B alone | 140M | 12.16 |
| LoRA on the 4B alone | 264M | 10.13 |

Adaptation is worth about 5 points; the 4B's cache about 1. A plain LoRA on the small model
beats the bridge with fewer parameters and no second model. Adding a LoRA to the decoder on
top of a frozen stitch made it worse (15.07); training both jointly reached 13.68, still
behind the LoRA alone.

**Redoing the control properly.** The self-bridge above took eight neighbouring 1.7B layers
as input, which buries the identity in a wide vector and gives the control a harder
optimisation problem than the bridge it controls for. With one source layer the control
starts exactly at identity. Four seeds:

| seed | 4B cache through stitch | own cache through stitch | difference, 95% CI |
|---|---|---|---|
| 0 | 12.37 | 11.34 | -1.03 [-1.27, -0.83] |
| 1 | 12.39 | 11.38 | -1.01 [-1.23, -0.82] |
| 2 | 12.36 | 11.41 | -0.95 [-1.15, -0.77] |
| 3 | 12.35 | 11.41 | -0.95 [-1.17, -0.75] |

Seed spread 0.03, gap thirty times that. At equal stitch and equal budget, the 1.7B decodes
better from its own cache than from the 4B's. The earlier "one point of transfer" was the
control's handicap. This is the cleanest finding in the project and the one prior work on
cross-model cache transfer had not controlled for.

## 4. Making the stitch small and fast

Compressing the closed-form map by rank does not work: the per-head ridge solution has a
flat spectrum, and keeping its top 64 of 256 directions takes perplexity from 36 to 6,900
before training. What works is the opposite cut: keep full rank, feed each 1.7B layer one 4B
layer instead of eight, drop the MLP.

| stitch | params | after training |
|---|---|---|
| 8 sources, full rank + MLP | 184M | 12.81 |
| 8 sources, full rank, no MLP | 118M | 12.66 |
| 1 source, full rank, no MLP | 14.7M | 12.37 |
| 8 sources, rank 32 | 16.6M | 14.00 |
| 1 source, even spacing instead of R2 | 14.7M | 12.82 |
| 1 source, random init instead of ridge | 14.7M | 837 |

Three things each earned a number: R2 selection is worth 0.45 over even spacing, ridge
initialisation is what lets a full-rank map train at all, and more source layers hurt once
you train.

Below a few thousand tokens the bridge is bound by kernel launches (five small kernels
times 28 layers). Captured as one CUDA graph per prompt length it costs 2.4 / 5.2 / 21 ms
at 256 / 1k / 4k tokens, about 3.5 percent of the 4B prefill it follows.

## 5. Interference and throughput

Two processes, no shared state. A decoder holds eight in-flight requests with 1,024-token
contexts and times every step; a prefiller runs on the same GPU or the other one. The first
version let the prefiller loop as fast as it could; a later version fixed the arrival
rate, which turned out to matter.

| prefiller | decoder p50 | p99 | decode tok/s |
|---|---|---|---|
| none | 39.2 ms | 41.0 ms | 204 |
| 1.7B, same GPU, saturated | 56.6 | 61.8 | 142 |
| 4B, same GPU, saturated | 57.2 | 62.2 | 140 |
| 4B, other GPU | 38.5 | 42.0 | 206 |
| 4B + stitch, other GPU (median of 6) | 39.5 | 44.0 | 203 |

Saturated, a colocated prefill costs the decoder 30 percent of throughput and 45 percent on
every token. At 2 to 4 prefills per second, the realistic case, a colocated 1.7B prefill
costs 8 to 15 percent and a 40 to 55 percent p99 tail; a colocated 4B 19 to 33 percent.
Disaggregated: no tail, under 4 percent, with or without the stitch, across four repeats.

Throughput on the same two GPUs, batches of 8 requests with 1,024-token prompts:

| generated tokens | two 1.7B replicas | 1.7B prefill to 1.7B decode | 4B + stitch to 1.7B decode |
|---|---|---|---|
| 64 | 351 tok/s | 158 | 158 |
| 128 | 380 tok/s | 184 | 179 |

A decode step costs about 40 ms at batch 8 whichever model runs it, so system throughput
is the number of decode GPUs times about 200 tok/s. Disaggregation halves it here because
it halves the decoders, and the prefill GPU idles most of the time. This is the textbook
trade-off: latency stability at the price of throughput per GPU, paying off only with
asymmetric pools (one prefill GPU feeding several decoders). Cross-process CUDA IPC was
blocked on this machine, so the cache moved through host memory; a prefill-heavy run
(4k prompts) was transfer-bound and says nothing about the pipeline.

## 6. Two more ways the cache could have helped

**Distillation.** Next-token loss is a signal the self-bridge gets equally well from the
1.7B's own cache, so maybe it cannot show what the 4B's adds. Train the stitch to match the
4B's output distribution instead. Same answer: fed its own cache, the 1.7B matches the 4B
better (KL 0.451) than fed the 4B's through the same stitch (0.533), two seeds. Under both
objectives the large model's cache is a worse context for the small decoder than its own.

**Longer prompts.** The stitch trained on 256-token prompts holds to 2,048 (never below
the untrained 1.7B, equal to the 4B at 1,024), so the position-free map does its job. But
the gap to the self-bridge grows with context, from 1.0 to 2.4 ppl. Long prompts make cache
transfer a worse deal, not a better one.

## 7. Stitch the residual stream instead

Section 2 said the last layers are where transfer hurts, so do not transfer them. Early
layers take KV from the 4B through the stitch; at a cut layer the 4B's residual stream,
mapped by one 2,560 to 2,048 linear, becomes the 1.7B's residual; the 1.7B runs its own
last layers over the prompt and writes native KV for them. Residual streams are far more
linearly alike across model sizes than caches are: R2 0.98 at layer 14, 0.88 at 20,
against 0.45 for the best cache map.

| design | 1.7B prefill layers it runs | ppl | control | gap, 95% CI |
|---|---|---|---|---|
| KV stitch only | 0 / 28 | 12.37 | 11.34 | -1.03 [-1.27, -0.83] |
| residual, cut 24 | 4 / 28 | 12.14 | 11.73 | -0.41 [-0.62, -0.22] |
| residual, cut 20 | 8 / 28 | 11.86 | 11.42 | -0.44 [-0.60, -0.31] |
| residual, cut 14 | 14 / 28 | 11.75 | 11.42 | -0.33 [-0.47, -0.19] |

Returning eight layers to the 1.7B recovers more than half the translation loss. The
control still wins. The 1.7B's tail on the decode GPU costs 2 to 5 percent decode
throughput at 2 to 4 prefills per second, proportional to the work.

## 8. Tasks, not perplexity

Everything so far was wikitext perplexity. Scoring ARC-Easy, ARC-Challenge, PIQA (5-shot,
log-likelihood multiple choice with the few-shot prompt as the prefill) and LAMBADA (last
word) changed the picture.

| task | 1.7B | 4B | pipeline (cut 20) | control | gap, 95% CI |
|---|---|---|---|---|---|
| ARC-Easy | 79.9 | 84.1 | 70.4 | 78.1 | -7.7 [-10.2, -5.2] |
| ARC-Challenge | 49.4 | 56.7 | 34.6 | 46.5 | -11.9 [-14.8, -9.0] |
| PIQA | 71.6 | 75.0 | 68.3 | 71.7 | -3.4 [-5.6, -1.2] |
| LAMBADA | 51.2 | 59.8 | 40.3 | 50.5 | -10.2 [-13.2, -7.1] |

Two conditions 0.4 ppl apart on wikitext are 12 points apart on ARC-Challenge. Perplexity
on the training domain measures fluency; it does not measure whether the context survived.

**Training levers do not move it.** Mixed-length training flattens the wikitext gap across
lengths. A LoRA on the 1.7B's tail layers helps the pipeline about twice as much as the
control, once trained on wikitext-103 rather than wikitext-2, which a 10M-parameter adapter
memorises in one pass. A question-answer corpus changes nothing. Eight recipes land at 11.4
to 11.9 ppl and every one scores 69 to 71 on ARC-Easy and 35 to 37 on ARC-Challenge. The
task floor is structural.

**Which component loses it.** Swap one 4B-sourced piece at a time for the 1.7B's own:

| early KV (layers < 20) | residual at layer 20 | ARC-Challenge | LAMBADA |
|---|---|---|---|
| 4B, through the stitch | 4B, through the map | 34.6 | 40.3 |
| 4B, through the stitch | 1.7B's own | 40.5 | 50.3 |
| 1.7B's own | 4B, through the map | 44.3 | 40.4 |
| 1.7B's own | 1.7B's own | 46.5 | 50.5 |

ARC blames the early-layer KV stitch (6 points alone, 12 with the map), because scoring an
answer attends from the choice tokens back into the early cache of the question. LAMBADA
blames the single residual map at the hand-over layer entirely, because the last word is a
function of the late-layer state at the final positions. Both looked nearly free on
perplexity.

## 9. The stitch that stays in residual space

Remove the KV-to-KV map altogether. For each early 1.7B layer, one linear map carries the
4B's residual at its best-matching layer into the 1.7B's residual; the 1.7B's own input
norm, key and value projections and RoPE produce that layer's cache. 21 maps, 110M
parameters, ridge-initialised, trained on the LM loss. Control: identity-initialised maps
on the 1.7B's own residuals.

| metric | pipeline before | residual-space | control | vs control, 95% CI |
|---|---|---|---|---|
| wikitext, 256 | 11.86 | 11.46 | 11.27 | +0.19 [+0.09, +0.28] |
| wikitext, 2,048 | 11.38 | 9.97 | 10.07 | -0.09 [-0.23, +0.03] |
| ARC-Easy | 70.4 | 74.0 | 78.9 | -4.9 [-7.3, -2.6] |
| ARC-Challenge | 34.6 | 42.4 | 47.2 | -4.8 [-7.5, -2.1] |
| PIQA | 68.3 | 70.7 | 73.7 | -3.0 [-5.1, -1.0] |
| LAMBADA | 40.3 | 42.5 | 54.5 | -12.0 [-15.1, -8.9] |

Eight training recipes had moved ARC-Challenge by at most three points; changing the
mechanism moved it by eight. LAMBADA barely moved, as the diagnostic predicted, because the
map at the hand-over layer was still there. Two seeds agree within 0.1 ppl and two task
points. Training the control twice as long makes it worse (11.27 to 11.42): 88M linear maps
memorise a 10M-token corpus on a second pass, so 3,000 steps is the budget. The bridge, 21
maps plus 20 layers of the 1.7B's own projections, costs 3 / 11 / 43 ms at 256 / 1k / 4k
tokens as a CUDA graph.

Why residual rather than cache, in one line: at decode time the queries come from the
1.7B's own residual, and a learned cache map can produce keys outside the manifold those
queries know how to address. A residual map cannot, because the 1.7B's own projections
build every key.

## 10. The cut, richer maps, and a native suffix

**The cut is not a dial.** Cut 24 matches its control on wikitext with only four native
layers, yet loses more on ARC-Easy; cut 14, with half the model native, is worse on
LAMBADA. Perplexity and tasks disagree about the cut, and neither direction moves LAMBADA.

**A richer hand-over map does not exist.** Three concatenated 4B layers into the cut map
(R2 0.92 against 0.88) and an MLP branch both land inside the single map's confidence
interval on every task.

**Recompute the last tokens natively.** The diagnostic said the LAMBADA loss sits in the
final positions. So give those positions the 1.7B's own: prefill the first T minus S prompt
tokens through the pipeline, run the full 1.7B over the last S. No training.

| native suffix S | ARC-Easy | ARC-Challenge | PIQA | LAMBADA |
|---|---|---|---|---|
| 0 | 74.0 | 42.4 | 70.7 | 42.5 |
| 32 | 73.3 | 41.8 | 72.9 | 49.1 |
| 64 | 76.0 | 44.9 | 73.2 | 53.9 |
| 128 | 78.5 | 48.1 | 73.4 | 55.1 |
| untrained 1.7B | 79.9 | 49.4 | 71.6 | 51.2 |

At S = 64 the pipeline is within noise of the control on ARC-Challenge, PIQA and LAMBADA
and above the untrained 1.7B on two tasks, reproduced on a second seed. But the task
prompts are 77 to 220 tokens, so a 64-token suffix is 30 to 80 percent of them and the
1.7B does half a native prefill. The design had become three stacked mechanisms: residual
maps, a native tail that the cut sweep had not justified, and a suffix that patched a
symptom. Time to pick one.

## 11. One mechanism

Residual maps at all 28 layers, cut 28, no native layers, no suffix. The 1.7B does no prompt
work. Cache-to-cache maps were not revisited, for the reason in section 9.

**The clean design exposed a bug.** LAMBADA collapsed to 8 percent, perplexity 134. The
training loss had only ever scored continuation tokens, decoded natively off the cache. The
prompt's last-position logit, which predicts the first decode token and is LAMBADA's entire
answer, was never in the loss. Below cut 28 the 1.7B's native tail layers had repaired that
position on the way through, which is exactly why the tail looked necessary and why the
diagnostic blamed the hand-over residual. At cut 28 the final map feeds nothing but that
logit, so it got no gradient at all and sat at its ridge initialisation, R2 0.67, the worst
of the 29 layers. ARC's wider gap at cut 28 was the same thing: the first token of every
answer choice is scored from that logit.

Scoring token P from the final logit gives the map a gradient, but one token in 257 is 0.4
percent of the loss, and that noisy term pushed the *control's* final map off the identity
it started at (LAMBADA 51.2 to 44.1). Scoring every prompt position from the prefill
logits, which at cut 28 come straight from the final map, costs nothing extra and gives it
P targets per example.

| objective | ARC-Easy | ARC-Challenge | PIQA | LAMBADA | control LAMBADA |
|---|---|---|---|---|---|
| continuation only | 71.7 | 43.7 | 70.1 | 8.0 | 51.2 |
| + token P from the final logit | 73.7 | 42.7 | 70.1 | 16.5 | 44.1 |
| + LM loss over all prompt positions | 75.5 | 44.0 | 70.8 | 22.8 | 53.3 |

The last column is the proof the objective is right: with the dense prompt loss, the same
maps on the 1.7B's own residuals land above the native model on LAMBADA (53.3, ppl 6.16
against 6.80). Wikitext continuation perplexity does not see any of this; it was within
noise of the control from the first cut-28 run while LAMBADA sat at 8 percent.

**Attention across the prompt.** Every map so far was per position. Since the two models
mix context differently, the information the 1.7B needs at position t might sit at other
positions in the 4B. Each map got a zero-initialised causal self-attention branch over the
4B's positions (LayerNorm on the source, 8 heads of 64, RoPE on q and k, output projection
at zero so step 0 equals the linear map). It matched the linear map within a point on every
cache-dominated row across six paired comparisons, was harmless on the self source, and
gained 2.5 points on LAMBADA (25.3 vs 22.8, control 53). Cross-position mixing is not the
missing ingredient. What this settles is where the deficit lives: everything the decoder
reads off the cache transfers (wikitext within noise, ARC and PIQA within 3 to 4 points),
and the last position's output representation does not, by any map tried.

**The prefiller emits the first token.** In disaggregated serving the prefill instance
produces token one and hands over the cache. Here the 4B's own logits give token one and
the 1.7B decodes from token two off the stitched cache. The final map is then not needed
at all, since the cache holds layers 0 to 27. No training; the same maps, a different first
token.

| task | 1.7B | 4B | pipeline | control | pipeline minus control, 95% CI |
|---|---|---|---|---|---|
| ARC-Easy | 79.9 | 84.1 | 79.4 | 79.1 | +0.3 [-1.8, +2.4] |
| ARC-Challenge | 49.4 | 56.7 | 48.7 | 48.3 | +0.4 [-2.4, +3.1] |
| PIQA | 71.6 | 75.0 | 72.2 | 73.6 | -1.4 [-3.4, +0.6] |
| LAMBADA | 51.2 | 59.8 | 59.4 | 53.3 | measures the 4B |

Within noise of the control on ARC-Easy, ARC-Challenge and PIQA, and within noise of the
untrained 1.7B on all three. LAMBADA is the 4B's number because the last word is the first
token. Compared with the suffix design (76.0 / 44.9 / 73.2 / 53.9) this is the same or
better on every task, and the decoder's share of prompt compute went from half a native
prefill to zero. The bridge is 28 maps plus the 1.7B's projections, 3 to 43 ms.

## 12. Where it stands

The design: one linear map per layer from the prefiller's residual stream into the
decoder's, the decoder's own projections building its cache, the prefiller emitting the
first token, the decoder doing no prompt work. Parity with the decoder's own prefill on
ARC-Easy, ARC-Challenge and PIQA, within 0.1 ppl on wikitext at every length.

What it does not do. It never beats the small model's own prefill on anything. Per request
it costs about 2.75 times the prefill FLOPs of the 1.7B alone and holds 3.4 times the
weights. At one prefill GPU per decode GPU it delivers half the throughput of two replicas.
The benefit is latency isolation for the decoder, and the heterogeneous version only beats
a same-model split when the prefill model is fixed by something else, such as a shared
prefill service already running the larger model.

Open threads, cheapest first: a second seed of the final design; goodput on the
interference harness with the decoder doing no prompt work; a long-context task with 1 to
2k-token prompts, where prefill offload pays and nothing has been measured; a second model
pair.

## Practical lessons

- Build the harness so it can prove it does nothing, then trust it. The injection test
  caught an off-by-one on day one and made every later failure attributable.
- A trained bridge's gain has to be measured against a same-budget adaptation of the
  receiver alone. Run the control with the same optimisation problem, not a harder one.
- Perplexity on the training domain will hide task loss by an order of magnitude. Get a
  task number before the second week.
- When a stacked design works, ask which parts are doing the work. The native tail and
  suffix here were compensating for an objective bug, and the cut sweep had already hinted
  the tail was not earning its place.
- Score every position the pipeline will be asked to produce. A loss that only covers the
  continuation left the final map untrained for six phases.
- Save every artifact you might later compare. LoRA adapters were not saved in the first
  phase, so those comparisons are unpaired; later, checkpoints kept in a RAM-backed scratch
  directory were lost to a cleanup twice. Keep them on disk.
- Guard scripts with a main block from the first file, not the fifth. Module-level calls
  re-ran whole experiments on import.
- Batched linear solves peak at eight times the memory of a per-head loop. Loop.
