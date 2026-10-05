# Flow-only d8 capacity test, T=L=2048

**Completed:** all 4,114 updates, likelihood evaluation and both generation
benchmarks. See `sap_flow_text_d8_results.md` and `sap_flow_text_d8_compiled.log`.
K=32 BPB-bound estimate 2.31868 versus dense 0.95179; batch-16 graph speedup 89.25x.
Samples remain incoherent. Original registration below is retained for provenance.

Registered 2026-10-04 before training, explicitly requested by the user.
This supersedes the d4 follow-up's old restriction against d8. One seed, one
full-budget run on Modal profile `nanochat2`; do not retrain dense.

Preflight: 30 focused text/flow tests pass, including both d4/d8 generation depth,
inverse/Jacobian, fixed noise replay without target access, finite gradients,
counted recognition cost and checkpoint/budget validation. Dense step 1680 and
training tokens verified against metadata. Checkpoint SHA256:
`3288b1328b14ee3ea106732f44b9bc92c1f0f4ed64fb58969880a13988586c38`.
Derived dense budget: 155,164,430,303,232,000 reported FLOPs.

Launched `flow_d8_tl_s1_20261004`, app `ap-b9wNkZXJodxJF7Y6Z48nvX`,
function call `fc-01M43368V65YZ2Y7BMS8CMS79D`. This is a single durable remote
training/evaluation call; no training is duplicated if log collection is retried.
On-device preflight derives 4,114 updates, 269,615,104 target tokens and
155,151,422,223,024,128 counted matrix FLOPs (0.0084% below the recorded baseline
budget). Model: 76,163,104 parameters. GPU: H100 80GB HBM3, torch 2.9.1+cu128.
Held-out scope: 128 rows, 1,239,960 text bytes; initial K=1 bound 3.226414 BPB.
Downloaded both d4/d8 held-out tensors and verified exact equality of every prompt
and target ID (`[128,128]` prompts, `[128,2048]` targets), not just matching byte
counts. The local live stream stalled, but independent read-only volume collection
confirmed continued training through the 3,085-update checkpoint; no job restarted.

## Mechanism and capacity

Preserve the d4 flow-only mechanism and objective. Width 512, 16 attention heads,
two affine coupling layers with two Transformer conditioner blocks each, followed
by four parallel lexical-decoder blocks: **eight generative blocks total**.
Four extra recognition blocks are training-only and fully charged. Keep the
full-length latent at eight channels, observed prompt 128, V=32768, T=L=2048
from the first update. All random variables are drawn up front; no teacher,
distillation, verifier, target-token input at generation, or token-conditioned
resampling loop. Depth is fixed neural computation, not eight generation rounds.

This scales depth, width and training budget together. It is not a pure depth
ablation, a new mechanism, or an assertion that capacity will cure collapse.
FlowSeq remains the nearest prior work; the existing downloaded/parsed reference
is `Literature Review/D19-1437_FlowSeq_Ma_2019.pdf`.

## Budget and implementation

Reuse `S08_dense_L_s1`, step 1680, read-only from `jimpearse01`, volume `nanochat`,
`out/s03_sap/d8/S08_dense_L_s1`. Recorded successful baseline in
`out/s08_lanes/summary_lanes_d8.json`: 440,401,920 tokens at 352,324,600 FLOPs/token.
Derive total budget from that product. Verify checkpoint metadata, tokenizer hash,
depth 8 / width 512 / full attention, and checkpoint step against the saved budget.
Re-evaluate dense on the flow run's exact held-out continuations, not the historical
0.958955 conventional BPB with a different evaluation scope.

Derive flow update count by flooring baseline budget divided by on-device counted
forward/backward matrix FLOPs, including recognition and vocabulary checkpoint
recomputation. This matches a matrix-FLOP proxy to the dense recorded estimator,
not wall-clock cost or identical optimizer recipes. Same d4 optimizer: AdamW
lr 3e-4, betas (0.9,0.95), weight decay 0.01, clip 1; 65,536 target tokens/update,
microbatch 4, 2% LR warmup, cosine decay to 0.1x; KL beta reaches 1 at 10% budget.
No extra objective/capacity sweep is authorized by this test.

One H100, at most two hours. Tiny local CPU correctness tests only. Retain source
snapshot, exact config, checkpoint/tokenizer hashes, data manifest, checkpoint,
optimizer/RNG state, learning curves and full logs. Use existing remote shards.
Projection chunk 128 bounds temporary logits to O(B*128*V); attention work remains
O(B*(T+128)^2*width*depth), hidden activations O(B*(T+128)*width*depth). Vocabulary
projection and attention are expected bottlenecks; measure rather than assume
throughput. Microbatching/gradient accumulation bounds training memory and loader
demand. No queued sweep or unbounded producer is introduced.

## Registered decision and evaluation

Complete the budget despite poor intermediate quality; abort only on engineering
faults, numerical nonfinites, provenance mismatch, or resource limit. This is the
user's requested scaling test, not another T=4 validity gate.

Keep the d4 protocol: 128 fixed held-out rows, K=1 ELBO curves at zero, quarter,
half, three-quarter and full budget; final K=1/8/32 importance BPB-bound estimates,
ESS, uncertainty, reconstruction and posterior/prior latent KL. Report bounds as
bounds in expectation, never exact inference BPB. Both dense/flow joint numerators
include special-token losses; bytes count text. Save 16 full prior samples and
descriptive dense-reference scores/diversity (not a verifier or training target).

Benchmark actual 2048 outputs, batches 1 and 16, same H100, prompt processing,
noise/flow, vocabulary projection and sampling included. Report eager and CUDA
graphs separately, three timed trials after warmup, both-model allocator peak
explicitly scoped. Exclude graph capture/warmup for both.

Research success still requires <=1.01x same-row dense joint BPB and >=2x graph
throughput with plausible samples/diversity. A loose bound cannot prove the exact
quality gap. Compare to d4 bound 2.34450 and final latent KL ~0.00086 nats/token to
test whether scaling changes the collapse diagnosis. Completing the registered
budget is not evidence of convergence or seed robustness. Do not scale further
automatically on a failed result. Download `sap_flow_text_d8_compiled.log`.
