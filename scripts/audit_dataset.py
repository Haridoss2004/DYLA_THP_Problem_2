"""Audit the structure, metadata, identity, and leakage risk of the source dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DATASET_NAME = "bzcasper/ai-tool-pool-jewelry-vision"


def _sha256(image: Any) -> str:
    image = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{image.mode}:{image.width}x{image.height}".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _perceptual_hash(image: Any, hash_size: int = 8) -> str:
    """Fast average hash used as a screening signal, not an identity proof."""
    pixels = list(image.convert("L").resize((hash_size, hash_size)).getdata())
    average = sum(pixels) / len(pixels)
    return "".join("1" if pixel > average else "0" for pixel in pixels)


def _hamming_distance(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right))


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _source_stem(filename: str) -> str:
    name = Path(filename).stem
    name = re.sub(r"\.rf\.[0-9a-f]+$", "", name, flags=re.IGNORECASE)
    return re.sub(r"[-_]\d+$", "", name)


def audit_dataset(dataset_name: str = DATASET_NAME) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install Hugging Face datasets and Pillow.") from exc

    dataset = load_dataset(dataset_name)
    report: dict[str, Any] = {
        "dataset_name": dataset_name,
        "total_images": 0,
        "splits": {},
        "available_columns": {},
        "identity_assessment": {},
        "representative_images": {},
        "duplicate_analysis": {},
        "leakage_risks": [],
    }
    records: list[dict[str, Any]] = []
    for split_name, split in dataset.items():
        feature = split.features.get("label")
        label_names = list(getattr(feature, "names", []) or [])
        labels = Counter()
        dimensions = Counter()
        filenames: list[str] = []
        representatives: dict[str, dict[str, Any]] = {}
        for index, row in enumerate(split):
            image = row["image"]
            label_value = row["label"]
            label_name = label_names[label_value] if label_names and isinstance(label_value, int) else str(label_value)
            filename = str(getattr(image, "filename", "") or "")
            filename_only = Path(filename).name
            labels[label_name] += 1
            dimensions[f"{image.width}x{image.height}"] += 1
            filenames.append(filename_only)
            record = {
                "split": split_name,
                "dataset_index": index,
                "label": label_name,
                "filename": filename_only,
                "image_path": filename,
                "source_stem_heuristic": _source_stem(filename_only),
                "sha256": _sha256(image),
                "phash": _perceptual_hash(image),
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "format": image.format,
                "exif_keys": sorted(str(key) for key in image.getexif().keys()),
            }
            records.append(record)
            representatives.setdefault(label_name, {
                "split": split_name,
                "dataset_index": index,
                "filename": filename_only,
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "format": image.format,
            })
        report["splits"][split_name] = {
            "size": len(split),
            "columns": list(split.column_names),
            "label_names": label_names,
            "label_distribution": dict(labels),
            "image_dimensions": dict(dimensions),
            "filename_count": len(filenames),
            "filename_examples": filenames[:10],
        }
        report["available_columns"][split_name] = list(split.column_names)
        report["representative_images"][split_name] = representatives
        report["total_images"] += len(split)

    by_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_phash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_sha[record["sha256"]].append(record)
        by_phash[record["phash"]].append(record)
        by_source[record["source_stem_heuristic"]].append(record)
    exact_groups = [group for group in by_sha.values() if len(group) > 1]
    phash_groups = [group for group in by_phash.values() if len(group) > 1]
    cross_split_exact = [group for group in exact_groups if len({r["split"] for r in group}) > 1]
    cross_split_phash = [group for group in phash_groups if len({r["split"] for r in group}) > 1]

    band_buckets: list[dict[str, list[int]]] = [defaultdict(list) for _ in range(4)]
    for position, record in enumerate(records):
        for band in range(4):
            start = band * 16
            band_buckets[band][record["phash"][start : start + 16]].append(position)
    candidate_pairs: set[tuple[int, int]] = set()
    for bucket in band_buckets:
        for positions in bucket.values():
            for offset, left in enumerate(positions):
                candidate_pairs.update((left, right) for right in positions[offset + 1:])
    near_pairs: list[dict[str, Any]] = []
    for left_position, right_position in candidate_pairs:
        left = records[left_position]
        right = records[right_position]
        if left["sha256"] == right["sha256"]:
            continue
        distance = _hamming_distance(left["phash"], right["phash"])
        if distance <= 4:
            near_pairs.append({"kind": "near_duplicate", "distance": distance,
                               "cross_split": left["split"] != right["split"],
                               "left": left, "right": right})
    heuristic_source_groups = [group for group in by_source.values() if len(group) > 1]
    cross_split_source_groups = [group for group in heuristic_source_groups if len({r["split"] for r in group}) > 1]
    report["duplicate_analysis"] = {
        "unique_sha256_count": len(by_sha),
        "exact_duplicate_image_count": sum(len(group) for group in exact_groups),
        "exact_duplicate_group_count": len(exact_groups),
        "cross_split_exact_duplicate_group_count": len(cross_split_exact),
        "unique_perceptual_hash_count": len(by_phash),
        "perceptual_duplicate_group_count": len(phash_groups),
        "cross_split_perceptual_duplicate_group_count": len(cross_split_phash),
        "near_duplicate_pair_count_hamming_le_4": len(near_pairs),
        "cross_split_near_duplicate_pair_count_hamming_le_4": sum(pair["cross_split"] for pair in near_pairs),
        "filename_source_stem_groups_gt_1": len(heuristic_source_groups),
        "cross_split_filename_source_stem_groups_gt_1": len(cross_split_source_groups),
        "exact_duplicate_groups": exact_groups[:20],
        "cross_split_perceptual_groups": cross_split_phash[:20],
        "near_duplicate_pairs": near_pairs[:100],
        "cross_split_filename_source_groups": cross_split_source_groups[:100],
    }
    non_image_columns = sorted({column for columns in report["available_columns"].values() for column in columns if column != "image"})
    report["identity_assessment"] = {
        "non_image_columns": non_image_columns,
        "verified_product_id_present": False,
        "filename_available_in_decoded_image": any(record["filename"] for record in records),
        "dataset_index_is_identity": False,
        "embedded_exif_identity_present": any(record["exif_keys"] for record in records),
        "image_level_catalogue_identity_defensible": True,
        "limitation": "The dataset exposes only image and label. Filenames and row indices can support stable image-level IDs, but no verified product, SKU, or manufacturer identity exists.",
    }
    report["leakage_risks"] = [
        "Raw train/validation/test retrieval metrics are at risk of leakage when exact or visually near-identical images cross splits.",
        "Filename source-stem matches are heuristic evidence only and cannot prove physical-product identity.",
        "A query that is itself present in the catalogue would measure lookup, not generalization to a new photograph.",
    ]
    return report, exact_groups + phash_groups + near_pairs


def _duplicate_row(kind: str, left: dict[str, Any], right: dict[str, Any], distance: Any = "") -> dict[str, Any]:
    return {"kind": kind, "distance": distance, "left_split": left["split"], "left_index": left["dataset_index"],
            "left_label": left["label"], "left_filename": left["filename"], "left_sha256": left["sha256"],
            "right_split": right["split"], "right_index": right["dataset_index"], "right_label": right["label"],
            "right_filename": right["filename"], "right_sha256": right["sha256"],
            "cross_split": left["split"] != right["split"]}


def write_reports(report: dict[str, Any], duplicate_groups: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_audit.json").write_text(json.dumps(report, indent=2, default=_json_default) + "\n", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for group in duplicate_groups:
        if isinstance(group, dict) and "left" in group:
            rows.append(_duplicate_row(group["kind"], group["left"], group["right"], group["distance"]))
        elif group:
            for left, right in zip(group, group[1:]):
                rows.append(_duplicate_row("exact_or_same_phash", left, right))
    with (output_dir / "duplicate_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["kind"])
        writer.writeheader()
        writer.writerows(rows)

    duplicates = report["duplicate_analysis"]
    lines = ["# Dataset Audit", "", "## 1. Dataset Overview", "",
             f"Dataset: `{report['dataset_name']}`.", f"Total images: **{report['total_images']}**. Features are `image` and `label`.", "",
             "## 2. Split Distribution", "", "| Split | Images |", "|---|---:|"]
    lines.extend(f"| {split} | {details['size']} |" for split, details in report["splits"].items())
    lines += ["", "## 3. Category Distribution", ""]
    for split, details in report["splits"].items():
        lines += [f"### {split}", "", f"`{json.dumps(details['label_distribution'], sort_keys=True)}`", ""]
    lines += ["## 4. Image Resolution Statistics", ""]
    lines.extend(f"- **{split}:** `{json.dumps(details['image_dimensions'], sort_keys=True)}`" for split, details in report["splits"].items())
    lines += ["", "Representative decoded images were inspected by category; deterministic filenames, dimensions, and formats are recorded in `dataset_audit.json`.", "",
              "## 5. Exact Duplicate Analysis", "", f"- Unique SHA-256 hashes: **{duplicates['unique_sha256_count']}**.",
              f"- Exact duplicate groups: **{duplicates['exact_duplicate_group_count']}**; images in groups: **{duplicates['exact_duplicate_image_count']}**.",
              f"- Cross-split exact duplicate groups: **{duplicates['cross_split_exact_duplicate_group_count']}**.", "",
              "## 6. Perceptual Duplicate Analysis", "", "The audit uses a dependency-free 64-bit average perceptual hash as a reproducible screening equivalent; it is not proof of product identity.",
              f"- Unique perceptual hashes: **{duplicates['unique_perceptual_hash_count']}**.", f"- Same-hash groups: **{duplicates['perceptual_duplicate_group_count']}**.",
              f"- Near-duplicate pairs at Hamming distance <= 4: **{duplicates['near_duplicate_pair_count_hamming_le_4']}**.", "",
              "## 7. Cross-Split Leakage Analysis", "", f"- Cross-split same perceptual-hash groups: **{duplicates['cross_split_perceptual_duplicate_group_count']}**.",
              f"- Cross-split near-duplicate pairs: **{duplicates['cross_split_near_duplicate_pair_count_hamming_le_4']}**.",
              f"- Cross-split filename source-stem groups: **{duplicates['cross_split_filename_source_stem_groups_gt_1']}**.", "",
              "These findings mean raw split retrieval results must be treated as potentially optimistic.", "",
              "## 8. Product Identity Analysis", "", "- Verified product ID: **No**.",
              f"- Non-image columns: `{', '.join(report['identity_assessment']['non_image_columns'])}`.", report["identity_assessment"]["limitation"], "",
              "## 9. Dataset Limitations", "", "- Multiple catalogue rows may be views, augmentations, or images of the same physical item.",
              "- Class labels identify jewellery categories, not products.", "- Filename patterns are source clues, not validated identity metadata.", "",
              "## 10. Recommended Retrieval Benchmark", "",
              "Use generated image-level IDs only, explicitly named as image identities. Build the catalogue from one leakage-screened partition, hold out query images whose exact/perceptual/source-stem groups overlap the catalogue, and report results separately for clean versus suspicious groups. For product-level claims, collect real phone queries with manually verified item provenance; do not use the raw labels as product IDs.", "",
              "## 11. Risks and Mitigations", ""]
    lines.extend(f"- {risk}" for risk in report["leakage_risks"])
    (output_dir / "dataset_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    try:
        report, duplicate_groups = audit_dataset(args.dataset)
        write_reports(report, duplicate_groups, args.output_dir)
    except Exception as exc:
        print(f"Dataset audit failed: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {args.output_dir / 'dataset_audit.json'}")
    print(f"Wrote {args.output_dir / 'dataset_audit.md'}")
    print(f"Wrote {args.output_dir / 'duplicate_report.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
