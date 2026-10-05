# S01 — Sampled-in-the-middle refinement for `T=L` SAP

## Claim

The failed `T=L=2048` SAP instantiation samples only after every head layer, so all token
distributions are deterministic functions of the same context and generation draws from a
product of marginals. S01 moves a stochastic cut *inside* the head: a cheap first stage samples
a token field, and later parallel layers condition on the realised field before final sampling.
The target is the original SAP result—one trunk pass emits the full continuation—without the
severe coherence collapse.

The paper bar is either better quality, or quality-neutral generation with at least `2x` batch-16
decode throughput versus autoregressive decoding. Parameter or FLOP reductions without measured
wall-clock speed do not count.

## Sampling-aware training objective

The first implementation exposed a necessary constraint: drawing a prior draft `d ~ p(d|h)` and
minimising `E_d[-log p(y|h,d)]` still has an optimum that ignores `d`. The observed continuation
and draft are independent conditional on `h`, so ordinary CE cannot teach the reference layer
which sampled branch should explain that continuation.

S01 therefore estimates `-log (1/K sum_k p(y|h,d_k))` from `K=4` realised drafts. Detached
posterior reference weights `softmax_k(log p(y|h,d_k))` supply a score-function credit signal to
the discrete draft policy; `sir_soft` additionally supplies the sparse straight-through pathwise
gradient. A fixed token-space posterior bridge replaces each training draft position with its
observed token with probability `0.5`. Token-aligned reference arms mask the diagonal, so output
`i` cannot read observed token `y_i`; sparse hierarchical arms instead treat sampled anchors as
committed decisions that fill positions may read. Draft CE tries to learn the aggregate posterior
that must be sampled without ground-truth anchors at inference. This is a latent-variable denoising
objective, not a teacher-forcing curriculum: the mixture is fixed from the first step.

## Controlled arms

| Mode | Isolated mechanism | Added asymptotic work | Kill condition |
|---|---|---|---|
| `sir` | hard on-policy draft; output `i` can read every sampled draft except `i` | draft readout + one reference layer | synthetic gate fails |
| `sir_soft` | sparse top-k straight-through expectation, hard tokens forward | `O(BTk d)` bridge | no gain over `sir` |
| `sir_compat` | dyadic multiscale low-rank draft messages | `O(BT d r log T)` | no gain over `sir` at equal FLOPs |
| `sir_context` | contextual soft-error distribution over draft top-k | `O(BTkr)` during training | no gain over `sir_compat` |
| `sir_conf` | second draft only replaces the fixed lowest-confidence fraction | second draft readout | no quality gain sufficient to retain `2x` speed |
| `sir_anchor` | sample every `a`th token, broadcast skeleton, parallel fill | `O(B(T/a)Vd)` draft readout | coherence loss at long range |
| `sir_pyramid` | coarse then fine sampled skeletons | two sparse draft readouts | no gain over anchors per FLOP |
| `sir_lattice` | one sum-product message on a learned top-k candidate chain | `O(BTk^2r)` | candidate restriction misses required branches |
| `sir_energy` | rank true blocks above cross-context Frankenstein blocks | one structured-negative score | no gain over `sir` |
| `sir_full` | soft + compatibility + context + confidence + energy | union of the above | misses either paper gate |

`k=16`, `r=32`, `K=4`, posterior mixture `0.5`, and confidence refinement fraction `0.25`
are fixed before the sweep. Draft logits are vocabulary-chunked: a full `B*T*V` tensor is never
retained. Leave-one-out masking prevents the trivial identity solution for token-aligned drafts;
the anchor arms expose only sparse, already-committed sampled decisions.

Anchor spacing is scale-relative: the exact T=4 gate uses coarse/fine strides `2/1`; the
T=L=2048 run uses `16/4`. Using `16/4` at T=4 would collapse both skeletons to position zero and
would not test the pyramid mechanism.

## Funnel and pre-registered gates

1. Wiring smoke: every arm must produce finite loss/gradients, sample valid token IDs, evaluate a
   finite Monte-Carlo block likelihood, respect the leave-one-out gradient mask, and match the KV
   decode loop.
2. Exact phrase-HMM at `T=4`: advance only with block KL `<=0.10` nats/block and invalid phrase
   rate `<=3%`. This tests the dependence mechanism before a costly language run.
3. FineWeb-Edu, depth 8, `T=L=2048`, one block start per sequence, equal dense training FLOPs:
   trunk validation BPB `<=1.01x` dense and reference-model block PPL `<=1.10x` autoregressive.
4. Systems gate: batch-16 end-to-end decode throughput `>=2x` autoregressive on the same H100.

All training arms use `--target-flops` derived from the dense checkpoint. The SAP head's decoder,
draft projections, vocabulary readouts, low-rank messages, lattice, and energy module are included
in the cost model, including all `K` realised training drafts. Report absolute throughput alongside
FLOPs.

## Recorded funnel

- S01-v1 (one on-policy draft, ordinary expected CE): all ten arms failed exactly as predicted by
  the independence argument. Block KL was `4.315–4.348` nats/block against true total correlation
  `4.270`; invalid phrase rate was `61.6–64.1%`.
- S01-v2 (`K=4` marginal likelihood + reference-weight policy credit, no token posterior): all ten
  arms moved away from the product-of-marginals solution, proving that post-sampling credit reaches
  the draft, but remained far outside the gate. Best was `sir_anchor`, block KL `3.443`, invalid
  `54.0%`.
- S01-v3 (v2 plus the fixed token posterior): all ten arms failed. Block KL ranged
  `4.207–6.647`; invalid rate ranged `30.7–53.4%`. Low posterior-conditioned training loss did not
  transfer to prior-only sampling, diagnosing aggregate-posterior mismatch rather than insufficient
  refiner capacity.
- Re-running the hierarchical arms with the correct T=4 strides (`2/1`) improved invalidity to
  `26.3–26.4%`, but `sir_anchor` still had KL `4.792` and `sir_pyramid` KL `8.649`. No arm passed
  the `0.10` / `3%` gate, so the pre-registered depth-8 H100 sweep was not launched.

## Reproduction

```bash
bash scripts/s01_sap.sh --smoke
bash scripts/s01_sap.sh --synthetic
MODAL_PROFILE=blessingjim31-workspace modal run modal_sap.py::s01 --depth 8
modal volume get nanochat out/s01_sap/s01_sap_d8_compiled.log .
python scripts/s01_compile_results.py
```

The Modal entry point starts the dense reference and all requested mechanisms concurrently, then
starts post-evaluations concurrently after checkpoints exist. Its compiled log contains the JSON
summary followed by every training, speed, and generation-evaluation log.
