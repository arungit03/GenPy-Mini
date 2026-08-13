"""Small training metrics accumulator."""

import math
import time


class Metrics:
    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.loss_total = 0.0
        self.loss_count = 0

    def update(self, loss: float, tokens: int = 0) -> None:
        self.loss_total += float(loss)
        self.loss_count += 1
        self.tokens = getattr(self, "tokens", 0) + int(tokens)

    def summary(self, *, global_step: int, micro_step: int, learning_rate: float, gradient_norm: float | None = None, tokens_seen: int | None = None) -> dict:
        loss = self.loss_total / max(1, self.loss_count)
        elapsed = max(1e-9, time.perf_counter() - self.started)
        logged_tokens = getattr(self, "tokens", 0) if tokens_seen is None else int(tokens_seen)
        result = {"training_loss": loss, "learning_rate": learning_rate, "global_step": global_step, "micro_step": micro_step, "tokens_seen": logged_tokens, "tokens_per_second": logged_tokens / elapsed, "elapsed_seconds": elapsed}
        if gradient_norm is not None:
            result["gradient_norm"] = float(gradient_norm)
        if loss < 80:
            result["perplexity"] = math.exp(loss)
        return result

    def reset(self) -> None:
        self.loss_total = 0.0
        self.loss_count = 0
