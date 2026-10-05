"""Splice codes (nanochat/splice.py): codes are drawn in round 0 from the prompt alone, a lane's
first input carries its junction's code, the junction token is restricted to that class, and the
whole thing is an exact, normalised factorisation."""
import itertools

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.splice import splice_layout


def _model(V=6, K=3, seq=32):
    cfg = GPTConfig(n_layer=2, n_head=2, n_kv_head=2, n_embd=64, vocab_size=V, sequence_len=seq,
                    window_pattern="L", splice_k=K)
    torch.manual_seed(0)
    model = GPT(cfg)
    model.init_weights(verify=True)
    with torch.no_grad():                    # move every weight off its init so the test has power
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.2)
        model.class_of_token.copy_(torch.arange(V) % K)
    return model.eval()


def test_layout_steps_and_visibility():
    """N=13, P=1, L=3 (S=4): prefix, lane 0, then a code slot and four token slots per later lane."""
    lay = splice_layout(13, 1, 3)
    assert lay["orig"].tolist() == [0, 1, 2, 3, 4, -1, 5, 6, 7, 8, -1, 9, 10, 11, 12]
    assert lay["rank"].tolist() == [0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    assert lay["pos"].tolist() == [0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 9, 10, 11, 12]
    assert lay["mask_orig"].tolist() == [4, 8] and lay["junc_orig"].tolist() == [5, 9]
    m = lay["mask"][0, 0]
    code1, first1, last0, code2 = 5, 6, 4, 10
    # A code sees the prompt, the other codes and lane 0's first input (drawn by the prompt), nothing else.
    assert m[code1].nonzero().squeeze(1).tolist() == [0, 1, 5, 10]
    assert m[last0, code1] and m[last0, first1]          # lane 0 writes the junction token knowing its code
    assert not m[first1, last0] and m[first1, code1]     # lane 1's start sees its code, not lane 0's end
    assert m[code2].nonzero().squeeze(1).tolist() == [0, 1, 5, 10]


def _splice_total(model, V=6, P=1, L=2, N=5):
    """log sum over every block of exp(-(token nats + code nats)) after a fixed one-token prompt."""
    seqs = torch.tensor(list(itertools.product(range(V), repeat=N - P + 1)))
    full = torch.cat([torch.full((seqs.size(0), 1), 1), seqs], dim=1)
    x, y = full[:, :N], full[:, 1:].clone()
    lls = []
    with torch.no_grad():
        for xs, ys in zip(x.split(4096), y.split(4096)):
            nll = model(xs, ys, splice=(P, L, V - 1), loss_reduction="none").view(ys.shape)
            lls.append(-nll.sum(-1))
    return torch.logsumexp(torch.cat(lls).double(), 0).item()


def test_splice_codes_are_a_normalised_distribution():
    """Summing p over all V^5 blocks gives 1: codes, masked junction tokens and lanes are exact."""
    assert abs(_splice_total(_model())) < 1e-3
