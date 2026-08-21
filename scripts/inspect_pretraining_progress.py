"""Summarize append-only Checkpoint 7 training metrics without pandas."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="runs/genpy200m_pretrain_v1/logs/training_metrics.jsonl")
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.metrics).read_text(encoding="utf-8").splitlines() if line.strip()]
    train = [record for record in records if "train_loss" in record]
    validation = [record for record in records if "validation_loss" in record]
    if not train:
        raise ValueError("metrics contain no training records")
    latest = train[-1]
    best_train = min(train, key=lambda record: record["train_loss"])
    best_val = min(validation, key=lambda record: record["validation_loss"]) if validation else None
    recent = train[-20:]
    initial_loss = train[0]["train_loss"]
    latest_val = validation[-1]["validation_loss"] if validation else None
    summary = {
        "current_step": latest.get("global_step", latest.get("step")),
        "tokens_seen": latest.get("tokens_seen"),
        "initial_train_loss": initial_loss,
        "latest_train_loss": latest["train_loss"],
        "best_train_loss": best_train["train_loss"],
        "latest_validation_loss": latest_val,
        "best_validation_loss": best_val["validation_loss"] if best_val else None,
        "step_of_best_validation": best_val.get("step") if best_val else None,
        "current_learning_rate": latest.get("learning_rate"),
        "average_recent_gradient_norm": statistics.fmean(record["gradient_norm"] for record in recent if record.get("gradient_norm") is not None),
        "average_recent_tokens_per_second": statistics.fmean(record["tokens_per_second"] for record in recent if record.get("tokens_per_second") is not None),
        "train_loss_decreasing": latest["train_loss"] < initial_loss,
        "validation_loss_improving": bool(best_val and latest_val <= validation[0]["validation_loss"]),
        "train_validation_divergence_warning": bool(latest_val is not None and latest_val > latest["train_loss"] * 1.25),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
