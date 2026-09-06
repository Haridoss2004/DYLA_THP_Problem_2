"""Top-k image retrieval over a local FAISS index."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np

from .catalogue import CatalogueRecord
from .embeddings import ImageEmbedder
from .index import load_faiss_index


class JewelleryRetriever:
    def __init__(self, index_path: str | Path, manifest_path: str | Path, embedder: ImageEmbedder | None = None):
        self.index = load_faiss_index(index_path)
        with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
            self.records = [CatalogueRecord(**row) for row in csv.DictReader(handle)]
        if self.index.ntotal != len(self.records):
            raise ValueError("FAISS index size does not match catalogue manifest size.")
        self.embedder = embedder or ImageEmbedder()

    def search(self, image: Any, top_k: int = 5) -> dict[str, Any]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        start = time.perf_counter()
        query = self.embedder.embed(image).reshape(1, -1)
        similarities, indices = self.index.search(query.astype(np.float32), min(top_k, self.index.ntotal))
        results = []
        for rank, (similarity, index) in enumerate(zip(similarities[0], indices[0]), start=1):
            if index < 0:
                continue
            record = self.records[int(index)]
            results.append({
                "rank": rank,
                "catalogue_id": record.catalogue_id,
                "label": record.label,
                "similarity": float(similarity),
                "image_path": record.image_path,
            })
        return {
            "query_id": None,
            "latency_ms": (time.perf_counter() - start) * 1000.0,
            "results": results,
        }