"""Inspect SFT metrics and report train/validation divergence warnings."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="runs/genpy200m_sft_v1/logs/training_metrics.jsonl")
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.metrics).read_text(encoding="utf-8").splitlines() if line.strip()]
    train = [item for item in records if "train_loss" in item]
    validation = [item for item in records if "validation_loss" in item]
    if not train:
        raise ValueError("no SFT training records found")
    best_validation = min(validation, key=lambda item: item["validation_loss"]) if validation else None
    recent = train[-20:]
    latest = train[-1]
    result = {"current_step": latest.get("global_step", latest.get("step")), "tokens_seen": latest.get("tokens_seen"), "initial_train_loss": train[0]["train_loss"], "latest_train_loss": latest["train_loss"], "best_train_loss": min(item["train_loss"] for item in train), "latest_validation_loss": validation[-1]["validation_loss"] if validation else None, "best_validation_loss": best_validation["validation_loss"] if best_validation else None, "step_of_best_validation": best_validation.get("step") if best_validation else None, "current_learning_rate": latest.get("learning_rate"), "average_recent_gradient_norm": statistics.fmean(item["gradient_norm"] for item in recent), "average_recent_tokens_per_second": statistics.fmean(item["tokens_per_second"] for item in recent), "train_loss_decreasing": latest["train_loss"] < train[0]["train_loss"], "validation_loss_improving": bool(best_validation and validation[-1]["validation_loss"] <= validation[0]["validation_loss"]), "train_validation_divergence_warning": bool(validation and validation[-1]["validation_loss"] > latest["train_loss"] * 1.25), "automatic_early_stopping": False}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
