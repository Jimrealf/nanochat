"""Bounded H100 training benchmark; execution changes only, no research training."""
import gc
import json
from pathlib import Path
import statistics
import time

import torch

from nanochat.flow_text import FlowText, FlowTextConfig
from scripts.sap_flow_text import amp, sync


def make_train_callable(model, compile_training=False):
    if compile_training:
        torch._inductor.config.compile_threads = 4
    return torch.compile(model, fullgraph=True, dynamic=False) if compile_training else model


def run(out, commit=None):
    torch.set_num_threads(8)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_float32_matmul_precision("high")
    torch._inductor.config.compile_threads = 4
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    config = FlowTextConfig(width=512, heads=16, conditioner_depth=2, decoder_depth=4, recognition_depth=4)
    cases = [(4, False, False), (16, False, True), (32, False, True), (16, True, True), (32, True, True)]
    for batch, compiled, fused in cases:
        row = {"microbatch": batch, "compiled": compiled, "fused_adamw": fused,
               "effective_batch_tokens": 262144, "accumulation": 128 // batch}
        print("PERF_START " + json.dumps(row), flush=True)
        model = fn = optimizer = None
        try:
            torch.manual_seed(71)
            model = FlowText(config).cuda().train()
            # Exercise nonidentity flow and a nontrivial posterior for gradient comparison.
            with torch.no_grad():
                for layer in model.flow:
                    layer.output.weight.normal_(0, .002)
                model.posterior_head.weight.normal_(0, .002)
            p = torch.randint(config.vocab, (batch, config.prompt), device="cuda")
            y = torch.randint(config.vocab, (batch, config.T), device="cuda")
            eps = torch.randn(batch, config.T, config.latent, device="cuda")
            beta = torch.tensor(.37, device="cuda")
            # Same weights, inputs and stochastic tape for a full-model reference.
            with amp("cuda"):
                reference_loss = model(p, y, beta, eps)
            reference_loss.backward()
            reference = {n: parameter.grad.detach().cpu() for n, parameter in model.named_parameters()}
            reference_value = reference_loss.item()
            model.zero_grad(set_to_none=True)
            del reference_loss
            fn = make_train_callable(model, compiled)
            sync("cuda"); compile_start = time.perf_counter()
            with amp("cuda"):
                actual_loss = fn(p, y, beta, eps)
            actual_loss.backward(); sync("cuda")
            row["first_forward_backward_seconds"] = time.perf_counter() - compile_start
            row["loss_reference"] = reference_value
            row["loss_actual"] = actual_loss.item()
            diff2 = norm2 = 0.
            max_abs = 0.
            for name, parameter in model.named_parameters():
                actual, expected = parameter.grad.detach().cpu(), reference[name]
                if not torch.isfinite(actual).all():
                    raise AssertionError(f"nonfinite gradient in {name}")
                diff2 += (actual.double() - expected.double()).square().sum().item()
                norm2 += expected.double().square().sum().item()
                max_abs = max(max_abs, (actual - expected).abs().max().item())
            row["gradient_relative_l2"] = (diff2 / max(norm2, 1e-30)) ** .5
            row["gradient_max_abs"] = max_abs
            # BF16 fusion/reduction is not bitwise equal; reject material changes.
            if abs(actual_loss.item() - reference_value) > .005 or row["gradient_relative_l2"] > .05:
                raise AssertionError("eager/optimized loss or gradient equivalence failed")
            del reference, actual_loss
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(.9, .95),
                                           weight_decay=.01, fused=fused)
            def update():
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for _ in range(128 // batch):
                    epsilon = torch.randn(batch, config.T, config.latent, device="cuda")
                    with amp("cuda"):
                        loss = fn(p, y, beta, epsilon)
                    (loss / (128 // batch)).backward()
                    losses.append(loss.detach())
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise RuntimeError("nonfinite benchmark gradients")
                optimizer.step()
                return torch.stack(losses).mean()
            for _ in range(2):
                update()
            sync("cuda")
            torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(5):
                sync("cuda"); tick = time.perf_counter()
                loss = update()
                sync("cuda"); times.append(time.perf_counter() - tick)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite benchmark loss")
            row.update(status="ok", trials_seconds=times, seconds_per_update=statistics.median(times),
                       tokens_per_second=262144 / statistics.median(times),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       torch=torch.__version__, gpu=torch.cuda.get_device_name())
        except Exception as exc:
            import traceback
            traceback.print_exc()
            row.update(status="error", error=repr(exc))
        finally:
            rows.append(row)
            (root / "results.json").write_text(json.dumps(rows, indent=2))
            print("PERF_RESULT " + json.dumps(row), flush=True)
            if commit:
                commit()
            del optimizer, fn, model
            torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()
    return rows
