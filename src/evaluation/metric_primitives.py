"""Small deterministic numeric and text primitives used by evaluation metrics."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Sequence


def ranking_scores(
    gold: set[str], ranked: Sequence[str], cutoff: int
) -> tuple[float, float, float, float]:
    """Return hit, recall, reciprocal rank and binary nDCG for one ranking."""

    hits = [rank for rank, source in enumerate(ranked, 1) if source in gold]
    distinct_hits = len(set(ranked) & gold)
    gains: list[float] = []
    credited: set[str] = set()
    for rank, source in enumerate(ranked, 1):
        gain = 1.0 if source in gold and source not in credited else 0.0
        gains.append(gain / math.log2(rank + 1))
        credited.add(source)
    ideal = sum(
        1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), cutoff) + 1)
    )
    return (
        float(bool(hits)),
        distinct_hits / len(gold),
        1.0 / hits[0] if hits else 0.0,
        sum(gains) / ideal if ideal else 0.0,
    )


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
