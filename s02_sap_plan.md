# S02 — convergence audit and correlated tree anchors

## Questions

S02 separates two claims that S01 did not establish:

1. Was the best sampled-anchor arm merely undertrained or undersized?
2. Does changing the *prior factorisation* from independent anchors to correlated ancestral
   anchors remove the invalid-block failure?

The phrase-HMM remains the first gate because its joint distribution and autoregressive reference
are exact. No language-scale run is authorised by improvement alone; an arm must still reach block
KL `<= 0.10` nats and invalid rate `<= 3%` at T=4.

## Registered arms

| Tag | Mechanism | Training |
|---|---|---|
| `s02_audit_w128` | S01-v2 `sir_anchor`, width 128 | joint through step 2k; freeze trunk; head-only through 32k; evaluate 2k/8k/32k |
| `s02_audit_w256` | same, width 256 | identical schedule |
| `s02_tree_l2` | root anchor, one parallel child-anchor round, remaining positions filled in parallel | joint through 2k; frozen-trunk head training through 8k |
| `s02_tree_full` | full balanced tree at T=4; exact factorisation ceiling | same |
| `s02_oracle` | `sir_anchor` refiner trained and evaluated with valid observed anchors | 2k joint + head-only through 8k; prior-only generation remains a control |
| `s02_tree_l2_32k` | clean long shallow-tree convergence curve | 2k joint + head-only through 32k |
| `s02_tree_full_32k` | clean long full-tree convergence curve | identical schedule |

The anchor audit uses the v2 objective exactly: K=4 marginal likelihood, reference-weight policy
credit 1.0, posterior mixture 0. The oracle diagnostic trains a separate refiner with observed
valid anchors (`posterior_mix=1`) and evaluates both observed-anchor and prior-anchor generation.
This avoids treating valid anchors as out-of-distribution for a prior-only-trained refiner.

## Correlated tree factorisation

`sir_tree` commits position zero first. Each following level bisects every uncommitted interval and
samples all positions in that level concurrently, conditioned on every previously committed anchor.
A positive `sap_sir_tree_levels` stops the tree early and fills all remaining positions in one
parallel round. Zero expands the full tree.

Training uses the observed earlier anchors, so its likelihood is exact:

`log p(y|h) = sum_level sum_j log p(y_j | h, earlier anchors) + sum_fill log p(y_i | h, anchors)`.

This is not a posterior latent: the anchors are output variables and the same ancestral
factorisation is used at generation. At T=2048 a full binary tree uses 12 sampling rounds including
the root instead of 2048 left-to-right rounds; a shallow tree uses still fewer rounds plus one fill.

## Interpretation rules

- Oracle anchors `<=3%` invalid while prior anchors remain poor: the refiner is sufficient and the
  independent anchor prior is the bottleneck.
- Large continued improvement from 8k to 32k: S01 was undertrained; run a cost projection before
  scaling to language.
- A flat 8k-to-32k curve: additional training is not the repair.
- `sir_tree_l2` passing is the preferred result because it preserves a genuine parallel fill.
- Only `sir_tree_full` passing shows that tree factorisation works but does not yet establish the
  best speed/quality operating point.

## Results and decision (2026-10-03)

All long arms used a clean 32k trajectory, froze the solved trunk at step 2k, and verified that
no frozen tensor changed. The independent-anchor family is converged far from the gate:

| Arm | 2k KL / invalid | 8k KL / invalid | 32k KL / invalid |
|---|---:|---:|---:|
| anchor, width 128 | 3.4146 / 54.25% | 3.4263 / 52.88% | 3.3365 / 54.00% |
| anchor, width 256 | 3.3928 / 54.35% | 3.3813 / 52.54% | 3.3214 / 52.29% |
| shallow tree (`levels=2`) | 0.1608 / 3.76% | 0.1426 / 3.66% | 0.1223 / 3.17% |
| full tree | 0.1573 / 4.79% | 0.1458 / 3.42% | 0.1118 / 3.91% |

The separate oracle-trained anchor refiner reaches **2.25% invalidity** when evaluated with
consistent observed anchors, while its independently sampled prior reaches **26.22%**. This is a
constrained architecture diagnostic, not a free fit: the same learned refiner is used in both
cases and only the source of the anchors changes. It establishes sufficient conditional-refiner
capacity on this testbed and localises the failure to the independent prior.

**Decision.** Width 2x (head FLOPs 3.67x) and training 16x do not repair independent anchors.
Correlated ancestral factorisation removes roughly 96% of their KL gap and most impossible blocks,
so the mechanism is correct, but neither tree arm clears the registered conjunction of KL `<=0.10`
and invalid `<=3%`. No depth-8 language sweep is authorised. The next experiment must change how
the shallow tree represents unresolved within-subtree dependence; it must not be another width,
step-count, or independently sampled-anchor sweep.

## User-authorised depth-8 exception (2026-10-03)

The user explicitly authorised one full depth-8 `T=L=2048` run for each correlated-tree arm before
choosing the next mechanism. This is an exception to the synthetic gate, not a retroactive gate
change. Both arms receive the dense depth-8 training-FLOP budget and use one block start per sequence
(`sap_block_frac=1/2048`). The existing dense checkpoint is the reference.

Pre-registered success requires all of:

- ordinary validation BPB within 1% of the dense reference at matched training FLOPs;
- exact tree block BPB within 5% of next-token BPB on the same evaluation tokens;
- block-generation reference perplexity within 10% of the arm's AR generation, without a distinct
  3-gram collapse;
- at least 2x batch-16 decode speedup over AR. Eager and CUDA-graph timings are both recorded.

Failure does not revive independent anchors. It means the next mechanism must target unresolved
subtree dependence before another language-scale run.
