"""Build an explicit leakage-aware validation/test query manifest."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import load_dataset

from jewellery_retrieval.catalogue import DATASET_NAME, load_catalogue, perceptual_hash, sha256_image
from jewellery_retrieval.leakage import LeakageChecker


FIELDS = [
    "query_id",
    "dataset_split",
    "dataset_index",
    "image_path",
    "label",
    "leakage_status",
    "catalogue_exclusion_reason",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--catalogue", default="data/catalogue/catalogue.csv")
    parser.add_argument("--output", default="data/evaluation/clean_queries.csv")
    parser.add_argument("--near-threshold", type=int, default=4)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    catalogue = load_catalogue(args.catalogue)
    checker = LeakageChecker(catalogue, args.near_threshold)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    rows: list[dict[str, object]] = []
    query_number = 1
    for split_name in ("validation", "test"):
        split = dataset[split_name]
        names = list(getattr(split.features.get("label"), "names", []) or [])
        for index, row in enumerate(split):
            image = row["image"]
            label = names[row["label"]] if names and isinstance(row["label"], int) else str(row["label"])
            decision = checker.check_hashes(sha256_image(image), perceptual_hash(image))
            counts[decision.status] += 1
            rows.append({
                "query_id": f"Q_{query_number:06d}",
                "dataset_split": split_name,
                "dataset_index": index,
                "image_path": str(getattr(image, "filename", "") or ""),
                "label": label,
                "leakage_status": decision.status,
                "catalogue_exclusion_reason": decision.reason,
            })
            query_number += 1
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    report = Path("reports/evaluation_protocol.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    clean = counts["clean"]
    contaminated = counts["contaminated"]
    report.write_text(
        "\n".join([
            "# Evaluation Protocol", "",
            "This protocol evaluates **image-level retrieval**, not verified product retrieval. "
            "The source dataset has no product IDs.", "",
            "## Catalogue", "",
            f"The catalogue is the train split with {len(catalogue)} records. Catalogue IDs are image-level IDs.", "",
            "## Candidate Queries", "",
            f"Candidate queries are validation and test rows: **{len(rows)}**.",
            f"Contaminated queries: **{contaminated}**.",
            f"Clean queries: **{clean}**.",
            f"Excluded from baseline metrics: **{contaminated}**.", "",
            "## Contamination Rule", "",
            f"A query is marked contaminated, and excluded from clean metrics, when it has an exact SHA-256 overlap, the same 64-bit average perceptual hash, or an average-hash Hamming distance <= {args.near_threshold} from any catalogue image.",
            "No source image is deleted or modified. Every candidate remains in `clean_queries.csv` with its status and reason.", "",
            "## Interpretation", "",
            "The clean set is conservative and measures retrieval against rows without the configured image-level overlap. It still does not establish physical-product identity or generalization to a new photograph of a known product.",
        ]) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output}: {len(rows)} candidates, {clean} clean, {contaminated} contaminated.")
    print(f"Wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())