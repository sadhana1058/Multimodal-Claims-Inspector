"""Operational metrics tracker: API calls, tokens, images, cost, runtime."""
import time
from dataclasses import dataclass, field

# Per-million-token pricing (approximate, stated assumptions in report)
MODEL_PRICING = {
    "claude-opus-4-8":   {"input": 5.0,  "output": 25.0},
    "claude-sonnet-4-6": {"input": 3.0,  "output": 15.0},
}


@dataclass
class OpsTracker:
    model: str
    api_calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    images_processed: int = 0
    start_time: float = field(default_factory=time.time)

    def record_call(self, input_tokens: int, output_tokens: int, image_count: int) -> None:
        self.api_calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.images_processed += image_count

    def record_cache_hit(self, image_count: int) -> None:
        self.cache_hits += 1
        self.images_processed += image_count

    @property
    def wall_time(self) -> float:
        return time.time() - self.start_time

    @property
    def est_cost_usd(self) -> float:
        p = MODEL_PRICING.get(self.model, {"input": 3.0, "output": 15.0})
        return (
            self.input_tokens / 1_000_000 * p["input"]
            + self.output_tokens / 1_000_000 * p["output"]
        )

    def summary(self) -> str:
        total = self.api_calls + self.cache_hits
        return (
            f"Model            : {self.model}\n"
            f"API calls        : {self.api_calls}  (cache hits: {self.cache_hits})\n"
            f"Input tokens     : {self.input_tokens:,}\n"
            f"Output tokens    : {self.output_tokens:,}\n"
            f"Images processed : {self.images_processed}\n"
            f"Est. cost (USD)  : ${self.est_cost_usd:.4f}\n"
            f"Wall time        : {self.wall_time:.1f}s\n"
        )
