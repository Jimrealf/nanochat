"""
Engine for efficient inference of our models.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes:
- The engine knows nothing about tokenization, it's purely token id sequences.

The whole thing is made as efficient as possible.
"""

import torch
import torch.nn.functional as F
import signal
import warnings
from contextlib import contextmanager
from collections import deque
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model

# -----------------------------------------------------------------------------
# Calculator tool helpers
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception as e:
        signal.alarm(0)
        # print(f"Warning: Failed to eval {formula}, exception: {e}") # it's ok ignore wrong calculator usage
        return None

def use_calculator(expr):
    """
    Evaluate a Python expression safely.
    Supports both math expressions and string operations like .count()
    """
    # Remove commas from numbers
    expr = expr.replace(",", "")

    # Check if it's a pure math expression (old behavior)
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:  # disallow power operator
            return None
        return eval_with_timeout(expr)

    # Check if it's a string operation we support
    # Allow: strings (single/double quotes), .count(), letters, numbers, spaces, parens
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]):
        return None

    # Disallow dangerous patterns
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    # Only allow .count() method for now (can expand later)
    if '.count(' not in expr:
        return None

    # Evaluate with timeout
    return eval_with_timeout(expr)

# -----------------------------------------------------------------------------
class KVCache:
    """
    KV Cache designed for Flash Attention 3's flash_attn_with_kvcache API.

    Key differences from FA2-style cache:
    - Tensors are (B, T, H, D) not (B, H, T, D)
    - FA3 updates the cache in-place during flash_attn_with_kvcache
    - Position tracked per batch element via cache_seqlens tensor
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype, v_head_dim=None):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim if v_head_dim is not None else head_dim
        # Pre-allocate cache tensors: (n_layers, B, T, H, D)
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, self.head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, self.v_head_dim, device=device, dtype=dtype)
        # Current sequence length per batch element (FA3 needs int32)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        # True when a decode step must be CUDA-graph capturable: the model then reads
        # positions from cache_seqlens on the device instead of via get_pos() on the host.
        self.graph_safe = False

    def reset(self):
        """Reset cache to empty state."""
        self.cache_seqlens.zero_()

    def get_pos(self):
        """Get current position (assumes all batch elements at same position)."""
        return self.cache_seqlens[0].item()

    def get_layer_cache(self, layer_idx):
        """Return (k_cache, v_cache) views for a specific layer."""
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        """Advance the cache position by num_tokens."""
        self.cache_seqlens += num_tokens

    def prefill(self, other):
        """
        Copy cached KV from another cache into this one.
        Used when we do batch=1 prefill and then want to generate multiple samples in parallel.
        """
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        assert self.n_layers == other.n_layers and self.n_heads == other.n_heads and self.head_dim == other.head_dim
        assert getattr(self, 'v_head_dim', self.head_dim) == getattr(other, 'v_head_dim', other.head_dim)
        assert self.max_seq_len >= other.max_seq_len
        other_pos = other.get_pos()
        self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
        self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
        self.cache_seqlens.fill_(other_pos)

# -----------------------------------------------------------------------------
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None, top_p=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1).
    top_p < 1 samples from the smallest set of tokens whose probability reaches top_p (nucleus)."""
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_p is not None and top_p < 1.0:
        probs = F.softmax(logits.float() / temperature, dim=-1)
        sp, si = torch.sort(probs, dim=-1, descending=True)
        keep = sp.cumsum(-1) - sp < top_p                          # tokens before the cut, the first always kept
        sp = torch.where(keep, sp, torch.zeros_like(sp))
        choice = torch.multinomial(sp / sp.sum(-1, keepdim=True), num_samples=1, generator=rng)
        return si.gather(1, choice)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)

# -----------------------------------------------------------------------------

class RowState:
    # Per-row state tracking during generation
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or [] # Current token sequence for this row
        self.forced_tokens = deque() # Queue of tokens to force inject
        self.in_python_block = False # Whether we are inside a python block
        self.python_expr_tokens = [] # Tokens of the current python expression
        self.completed = False # Whether this row has completed generation

class Engine:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer # needed for tool use

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """Same as generate, but does single prefill and then clones the KV cache."""
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        device = self.model.get_device()
        # NOTE: setting the dtype here and in this way is an ugly hack.
        # Currently the repo assumes that cuda -> bfloat16 and everything else -> float32.
        # We need to know the dtype here to call __init__ on KVCache and pre-allocate its tensors.
        # As a quick hack, we're making generate() function inherit and know about this repo-wise assumption.
        # I think there has to be a bigger refactor to deal with device/dtype tracking across the codebase.
        # In particular, the KVCache should allocate its tensors lazily
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # Get the special tokens we need to coordinate the tool use state machine
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>") # if sampled, ends row
        bos = self.tokenizer.get_bos_token_id() # if sampled, ends row

        # 1) Run a batch 1 prefill of the prompt tokens
        m = self.model.config
        # Allow models (e.g. MST) to override KVCache sizing via kv_cache_config property.
        # Otherwise derive from config as usual (GPT path).
        if hasattr(self.model, 'kv_cache_config'):
            kv_model_kwargs = self.model.kv_cache_config
        else:
            head_dim = m.n_embd // m.n_head
            v_head_dim = head_dim
            if hasattr(self.model, 'transformer') and len(self.model.transformer.h) > 0 and hasattr(self.model.transformer.h[0], 'attn') and hasattr(self.model.transformer.h[0].attn, 'v_head_dim'):
                v_head_dim = self.model.transformer.h[0].attn.v_head_dim
            kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": head_dim, "v_head_dim": v_head_dim, "num_layers": m.n_layer}
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :].expand(num_samples, -1)  # (num_samples, vocab_size)

        # 2) Replicate the KV cache for each sample/row
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill # no need to keep this memory around

        # 3) Initialize states for each sample
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) Main generation loop
        num_generated = 0
        while True:
            # Stop condition: we've reached max tokens
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # Stop condition: all rows are completed
            if all(state.completed for state in row_states):
                break

            # Sample the next token for each row
            next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
            sampled_tokens = next_ids[:, 0].tolist()

            # Process each row: choose the next token, update state, optional tool use
            token_column = [] # contains the next token id along each row
            token_masks = [] # contains the mask (was it sampled (1) or forced (0)?) along each row
            for i, state in enumerate(row_states):
                # Select the next token in this row
                is_forced = len(state.forced_tokens) > 0 # are there tokens waiting to be forced in deque?
                token_masks.append(0 if is_forced else 1) # mask is 0 if forced, 1 if sampled
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                # Update the state of this row to include the next token
                state.current_tokens.append(next_token)
                # On <|assistant_end|> or <|bos|>, mark the row as completed
                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                # Handle tool logic
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            # Yield the token column
            yield token_column, token_masks
            num_generated += 1

            # Prepare logits for next iteration
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            logits = self.model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]  # (B, vocab_size)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that just returns the final token sequences.
        Returns a list of token sequences (list of lists of ints).
        Terminal tokens (assistant_end, bos) are not included in the results.
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # Stop if all rows are completed
            if all(completed):
                break
        return results, masks


# -----------------------------------------------------------------------------
# SAP: block decoding against a plain autoregressive loop, both with a KV cache.
#
# These two loops are deliberately symmetric (same prefill, same cache, no tool-use state
# machine) so that tokens/sec measures one thing: one trunk pass per token versus one trunk
# pass per T tokens. Both read out through the same lm_head, T rows per block versus one row
# per token, so the unembedding work per generated token is the same.

def _prompt_ids(tokens, num_samples, device):
    """A list of ids is one prompt repeated num_samples times; a (B, L) tensor is B prompts."""
    if torch.is_tensor(tokens):
        return tokens.to(device=device, dtype=torch.long)
    return torch.tensor([tokens] * num_samples, dtype=torch.long, device=device)


def _kv_cache_for(model, batch_size, seq_len, device, dtype):
    m = model.config
    head_dim = m.n_embd // m.n_head
    return KVCache(batch_size=batch_size, seq_len=seq_len, num_heads=m.n_kv_head,
                   head_dim=head_dim, num_layers=m.n_layer, device=device, dtype=dtype)


@torch.inference_mode()
def generate_ar_kv(model, tokens, max_tokens, num_samples=1, temperature=1.0, seed=42, top_p=None,
                   rep_penalty=None):
    """Next-token decoding with a KV cache: one trunk pass per token. Returns (B, max_tokens).

    tokens: one prompt (list of ids, repeated num_samples times) or a (B, L) tensor of
    equal-length prompts."""
    device = model.get_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    ids = _prompt_ids(tokens, num_samples, device)
    kv = _kv_cache_for(model, ids.size(0), ids.size(1) + max_tokens + 1, device, dtype)
    logits = model.forward(ids, kv_cache=kv)[:, -1, :]
    seen = None
    if rep_penalty is not None and rep_penalty != 1.0:          # CTRL repetition penalty over prompt + output
        seen = torch.zeros(ids.size(0), logits.size(-1), dtype=torch.bool, device=device)
        seen.scatter_(1, ids, True)
    out = []
    for _ in range(max_tokens):
        if seen is not None:
            lf = logits.float()
            logits = torch.where(seen, torch.where(lf > 0, lf / rep_penalty, lf * rep_penalty), lf)
        nxt = sample_next_token(logits, rng, temperature, top_p=top_p)
        out.append(nxt)
        if seen is not None:
            seen.scatter_(1, nxt, True)
        logits = model.forward(nxt, kv_cache=kv)[:, -1, :]
    return torch.cat(out, dim=1)


@torch.inference_mode()
def generate_block_kv(model, tokens, max_tokens, num_samples=1, temperature=1.0, seed=42):
    """SAP block decoding with a KV cache: one trunk pass per T tokens. Returns (B, max_tokens).

    Each step feeds the T tokens of the previous block through the trunk (extending the
    cache), and the block head turns the last hidden state, plus a rolling window of recent
    hidden states, into the next T tokens.
    """
    head = model.sap_head
    assert head is not None, "generate_block_kv needs a model built with sap_block_T > 0"
    device = model.get_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    ids = _prompt_ids(tokens, num_samples, device)
    kv = _kv_cache_for(model, ids.size(0), ids.size(1) + max_tokens + head.T + 1, device, dtype)
    from nanochat.block_head import DEPTH_MODES
    if head.mode in DEPTH_MODES:
        # The block's tokens come from passes through the top layers (one per token for
        # depth_local, one per bisection round for depth_tree), reading the
        # trunk's KV cache (or the trained copies' own cache); the next trunk pass then appends
        # the whole block to the cache(s).
        first = head.depth_layer_ids[0]
        ckv = model.sap_depth_copy_cache(ids.size(0), kv.k_cache.size(2), device, kv.k_cache.dtype)
        out, n = [], 0
        x, st = model.forward(ids, kv_cache=kv, skip_logits=True, sap_capture=True)
        model.sap_depth_extend_copy_cache(ckv, ids, st, kv)
        while n < max_tokens:
            blk = model._sap_depth_sample(kv, model.sap_depth_state(st), model._sap_readout(x[:, -1]),
                                          temperature=temperature, generator=gen, copy_kv=ckv,
                                          top_first=x[:, -1])
            out.append(blk)
            n += blk.size(1)
            if n >= max_tokens:
                break
            x, st = model.forward(blk, kv_cache=kv, skip_logits=True, sap_capture=True)
            model.sap_depth_extend_copy_cache(ckv, blk, st, kv)
        return torch.cat(out, dim=1)[:, :max_tokens]
    x = model.forward(ids, kv_cache=kv, skip_logits=True)
    hist = x[:, -head.W:] if head.W > 0 else x[:, -1:]
    out, n = [], 0
    while n < max_tokens:
        h, ctx, valid = model._sap_last(hist, x[:, -1])
        blk = head.sample(h, ctx, valid, model._sap_readout, embed=model.transformer.wte,
                          embed_table=model.transformer.wte.weight,
                          temperature=temperature, generator=gen)
        out.append(blk)
        n += blk.size(1)
        if n >= max_tokens:
            break
        x = model.forward(blk, kv_cache=kv, skip_logits=True)
        hist = torch.cat([hist, x], dim=1)[:, -max(head.W, 1):]
    return torch.cat(out, dim=1)[:, :max_tokens]


if __name__ == "__main__":
    """
    Quick inline test to make sure that the naive/slow model.generate function
    is equivalent to the faster Engine.generate function here.
    """
    import time
    # init compute
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    # load the model and tokenizer
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    # common hyperparameters
    kwargs = dict(max_tokens=64, temperature=0.0)
    # set the starting prompt
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)
    # generate the reference sequence using the model.generate() function
    generated_tokens = []
    torch.cuda.synchronize()
    t0 = time.time()
    stream = model.generate(prompt_tokens, **kwargs)
    for token in stream:
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Reference time: {t1 - t0:.2f}s")
    reference_ids = generated_tokens
    # generate tokens with Engine
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs) # note: runs in fp32
    torch.cuda.synchronize()
    t0 = time.time()
    for token_column, token_masks in stream:
        token = token_column[0] # only print out the first row
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Engine time: {t1 - t0:.2f}s")
    # compare the two sequences
    for i in range(len(reference_ids)):
        if reference_ids[i] != generated_tokens[i]:
            print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"Match: {reference_ids == generated_tokens}")
