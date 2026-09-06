"""Evaluate clean image-level retrieval queries against the train catalogue."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jewellery_retrieval.embeddings import ImageEmbedder
from jewellery_retrieval.retrieval import JewelleryRetriever


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", default="data/evaluation/clean_queries.csv")
    parser.add_argument("--index", default="data/index/catalogue.faiss")
    parser.add_argument("--manifest", default="data/index/catalogue_manifest.csv")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    embedder = ImageEmbedder()
    retriever = JewelleryRetriever(args.index, args.manifest, embedder)
    with Path(args.queries).open(newline="", encoding="utf-8") as handle:
        queries = [row for row in csv.DictReader(handle) if row["leakage_status"] == "clean"]
    results: list[dict[str, object]] = []
    latencies: list[float] = []
    category_stats: dict[str, Counter[str]] = defaultdict(Counter)
    for query in queries:
        started = time.perf_counter()
        retrieval = retriever.search(query["image_path"], args.top_k)
        latency = (time.perf_counter() - started) * 1000.0
        latencies.append(latency)
        retrieved_ids = [item["catalogue_id"] for item in retrieval["results"]]
        image_id = f"JEW_{int(query['dataset_index']) + 1:06d}" if query["dataset_split"] == "train" else None
        # Validation/test rows are distinct image identities from the train catalogue;
        # category agreement is therefore the only dataset-grounded auxiliary signal.
        top1_category = retrieval["results"][0]["label"] if retrieval["results"] else None
        category_stats[query["label"]]["queries"] += 1
        category_stats[query["label"]]["category_top1"] += int(top1_category == query["label"])
        results.append({
            "query_id": query["query_id"],
            "dataset_split": query["dataset_split"],
            "dataset_index": int(query["dataset_index"]),
            "label": query["label"],
            "top1_catalogue_id": retrieved_ids[0] if retrieved_ids else None,
            "top5_catalogue_ids": retrieved_ids,
            "top1_similarity": retrieval["results"][0]["similarity"] if retrieval["results"] else None,
            "latency_ms": latency,
            "image_identity_match": False,
            "note": "No verified product identity; validation/test image is not present in train catalogue.",
        })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "evaluation_type": "image-level retrieval with category diagnostics",
        "product_identity_supported": False,
        "catalogue_size": len(retriever.records),
        "embedding_dimension": embedder.dimension,
        "query_count_clean": len(queries),
        "top1_accuracy": None,
        "top5_accuracy": None,
        "mrr": None,
        "category_top1_accuracy": {
            category: stats["category_top1"] / stats["queries"]
            for category, stats in sorted(category_stats.items())
        },
        "latency_ms": {
            "median": statistics.median(latencies) if latencies else 0.0,
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
        },
        "limitation": "The source dataset has no product IDs. Clean validation/test queries cannot be scored for product or image identity against a train-only catalogue; category agreement is reported only as a diagnostic.",
    }
    (output_dir / "baseline_results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# Baseline Results", "", "## Scope", "", "This is an image-level retrieval baseline. Product-level accuracy is not reported because the dataset provides no verified product identities.", "", f"- Catalogue size: **{summary['catalogue_size']}**", f"- Embedding dimension: **{summary['embedding_dimension']}**", f"- Clean query count: **{summary['query_count_clean']}**", "", "## Metrics", "", "- Top-1 image/product accuracy: **Not computable** for train-only catalogue versus validation/test image identities.", "- Top-5 image/product accuracy: **Not computable** for the same reason.", "- MRR: **Not computable** without a verified relevance identity.", "", "Category top-1 agreement is a diagnostic, not retrieval identity accuracy:", ""]
    lines.extend(f"- {category}: {value:.3f}" for category, value in summary["category_top1_accuracy"].items())
    lines += ["", "## Latency", "", f"- Median: **{summary['latency_ms']['median']:.2f} ms**", f"- p95: **{summary['latency_ms']['p95']:.2f} ms**", f"- p99: **{summary['latency_ms']['p99']:.2f} ms**", "", "## Limitation", "", summary["limitation"]]
    (output_dir / "baseline_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_dir / 'baseline_results.json'}")
    print(f"Wrote {output_dir / 'baseline_results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())