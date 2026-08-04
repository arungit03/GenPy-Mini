"""Bounded-memory sequence statistics."""

from __future__ import annotations

import math
from collections import Counter


def histogram_percentile(histogram: Counter[int], percentile: float) -> int:
    """Return a nearest-rank percentile from a sequence-length histogram."""
    count = sum(histogram.values())
    if count == 0:
        return 0
    rank = max(1, math.ceil(percentile * count))
    cumulative = 0
    for value, frequency in sorted(histogram.items()):
        cumulative += frequency
        if cumulative >= rank:
            return value
    return max(histogram)


def summarize_lengths(histogram: Counter[int], context_length: int) -> dict[str, float | int]:
    """Summarize counts and context fit without retaining every sequence length."""
    records = sum(histogram.values())
    total_tokens = sum(length * count for length, count in histogram.items())
    over_context = sum(count for length, count in histogram.items() if length > context_length)
    return {
        "record_count": records,
        "average_tokens_per_record": total_tokens / records if records else 0.0,
        "median_sequence_length": histogram_percentile(histogram, 0.50),
        "p90_sequence_length": histogram_percentile(histogram, 0.90),
        "p95_sequence_length": histogram_percentile(histogram, 0.95),
        "p99_sequence_length": histogram_percentile(histogram, 0.99),
        "records_longer_than_context": over_context,
        "percentage_fitting_context": 100.0 * (records - over_context) / records
        if records
        else 0.0,
    }
