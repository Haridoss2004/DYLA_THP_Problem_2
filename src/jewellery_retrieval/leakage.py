"""Leakage-aware query selection for image-level retrieval evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .catalogue import CatalogueRecord, perceptual_hash, sha256_image


def hamming_distance(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right))


@dataclass(frozen=True)
class LeakageDecision:
    status: str
    reason: str
    nearest_distance: int | None


class LeakageChecker:
    """Check query images against catalogue hashes without changing source data."""

    def __init__(self, catalogue: Iterable[CatalogueRecord], near_threshold: int = 4):
        self.catalogue = list(catalogue)
        self.near_threshold = near_threshold
        self._sha256 = {record.sha256 for record in self.catalogue}
        self._phash = {record.phash for record in self.catalogue}
        self._buckets: dict[str, list[str]] = {}
        for value in self._phash:
            for band in range(4):
                key = f"{band}:{value[band * 16:(band + 1) * 16]}"
                self._buckets.setdefault(key, []).append(value)

    def check_hashes(self, sha256: str, phash: str) -> LeakageDecision:
        if sha256 in self._sha256:
            return LeakageDecision("contaminated", "exact_sha256_overlap", 0)
        if phash in self._phash:
            return LeakageDecision("contaminated", "same_perceptual_hash", 0)
        candidates: set[str] = set()
        for band in range(4):
            key = f"{band}:{phash[band * 16:(band + 1) * 16]}"
            candidates.update(self._buckets.get(key, []))
        nearest = min((hamming_distance(phash, value) for value in candidates), default=None)
        if nearest is not None and nearest <= self.near_threshold:
            return LeakageDecision(
                "contaminated",
                f"perceptual_hamming_distance<={self.near_threshold}",
                nearest,
            )
        return LeakageDecision("clean", "no_configured_overlap", nearest)

    def check_image(self, image: object) -> LeakageDecision:
        return self.check_hashes(sha256_image(image), perceptual_hash(image))