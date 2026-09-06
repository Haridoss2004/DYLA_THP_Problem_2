"""Build deterministic image-level catalogue manifests from the HF dataset."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

DATASET_NAME = "bzcasper/ai-tool-pool-jewelry-vision"
CATALOGUE_FIELDS = [
    "catalogue_id",
    "dataset_split",
    "dataset_index",
    "label",
    "image_path",
    "sha256",
    "phash",
]


@dataclass(frozen=True)
class CatalogueRecord:
    catalogue_id: str
    dataset_split: str
    dataset_index: int
    label: str
    image_path: str
    sha256: str
    phash: str


def sha256_image(image: Any) -> str:
    image = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{image.mode}:{image.width}x{image.height}".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def perceptual_hash(image: Any, hash_size: int = 8) -> str:
    """Use the same fast average hash as the dataset audit."""
    pixels = list(image.convert("L").resize((hash_size, hash_size)).getdata())
    average = sum(pixels) / len(pixels)
    return "".join("1" if pixel > average else "0" for pixel in pixels)


def _label_name(split: Any, value: Any) -> str:
    names = list(getattr(split.features.get("label"), "names", []) or [])
    return names[value] if names and isinstance(value, int) else str(value)


def iter_catalogue_records(dataset: Any, split_name: str = "train") -> Iterable[CatalogueRecord]:
    split = dataset[split_name]
    for index, row in enumerate(split):
        image = row["image"]
        source_path = str(getattr(image, "filename", "") or "")
        yield CatalogueRecord(
            catalogue_id=f"JEW_{index + 1:06d}",
            dataset_split=split_name,
            dataset_index=index,
            label=_label_name(split, row["label"]),
            image_path=source_path,
            sha256=sha256_image(image),
            phash=perceptual_hash(image),
        )


def build_catalogue(
    output_path: str | Path,
    dataset_name: str = DATASET_NAME,
    split_name: str = "train",
) -> list[CatalogueRecord]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name)
    records = list(iter_catalogue_records(dataset, split_name))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CATALOGUE_FIELDS)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    return records


def load_catalogue(path: str | Path) -> list[CatalogueRecord]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [CatalogueRecord(**row) for row in csv.DictReader(handle)]