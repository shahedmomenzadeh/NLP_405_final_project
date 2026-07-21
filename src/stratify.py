"""Stratified sampling of 20 representative test samples.

Strategy:
  1. Bin samples into 3 length buckets: short (1-7), medium (8-14), long (15+).
  2. Allocate a target count to each bin: short=6, medium=10, long=4.
  3. Within each bin, greedily pick samples that maximise the *set* of
     BIO entity types covered across the whole subset, until the bucket
     is full. Falls back to random fill if a bucket is exhausted.
  4. Returns a list of indices into the original sample list.

No training data is used here — this only inspects the test set to
ensure the 20 selected samples span the full distribution.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import List, Sequence, Tuple


def length_bin(n_tokens: int) -> str:
    if n_tokens <= 7:
        return "short"
    if n_tokens <= 14:
        return "medium"
    return "long"


def stratified_sample(
    samples: Sequence[dict],
    n_total: int = 20,
    bucket_targets: dict = None,
    seed: int = 42,
) -> List[int]:
    """Return indices into `samples` for the stratified subset.

    Args:
        samples: list of {"tokens": [...], "tags": [...], "intent": ...}
        n_total: total number of samples to pick
        bucket_targets: dict of {bin_name: count}; default short=6, medium=10, long=4.
            If provided targets don't sum to n_total, they are rescaled proportionally.
        seed: RNG seed
    """
    if bucket_targets is None:
        bucket_targets = {"short": 6, "medium": 10, "long": 4}
    s = sum(bucket_targets.values())
    if s != n_total:
        bucket_targets = {k: max(0, round(v * n_total / s)) for k, v in bucket_targets.items()}
        diff = n_total - sum(bucket_targets.values())
        if diff:
            largest = max(bucket_targets, key=bucket_targets.get)
            bucket_targets[largest] += diff

    rng = random.Random(seed)
    bins: dict = {b: [] for b in bucket_targets}
    for i, s_ in enumerate(samples):
        bins[length_bin(len(s_["tokens"]))].append(i)

    covered: set = set()
    chosen: List[int] = []

    for bin_name, target in bucket_targets.items():
        if target == 0:
            continue
        pool = bins[bin_name]
        rng.shuffle(pool)
        picked: List[int] = []

        for idx in pool:
            if len(picked) >= target:
                break
            tags = samples[idx]["tags"]
            new_types = {t for t in tags if t not in ("O", "PAD", "X")} - covered
            if new_types or len(picked) < target // 2:
                picked.append(idx)
                covered.update(t for t in tags if t not in ("O", "PAD", "X"))

        if len(picked) < target:
            remaining = [i for i in pool if i not in picked]
            rng.shuffle(remaining)
            picked.extend(remaining[: target - len(picked)])

        chosen.extend(picked)

    return chosen


def summarise_subset(samples: Sequence[dict], indices: Sequence[int]) -> dict:
    """Return a summary of the stratified subset for inspection."""
    sub = [samples[i] for i in indices]
    bin_counter = Counter(length_bin(len(s["tokens"])) for s in sub)
    intent_counter = Counter(s["intent"] for s in sub)
    tag_counter = Counter()
    for s in sub:
        tag_counter.update(t for t in s["tags"] if t not in ("O", "PAD", "X"))
    return {
        "n_samples": len(sub),
        "length_bins": dict(bin_counter),
        "n_intent_classes": len(intent_counter),
        "intents": dict(intent_counter),
        "n_distinct_entity_types": len(tag_counter),
        "entity_type_counts": dict(tag_counter.most_common()),
    }
