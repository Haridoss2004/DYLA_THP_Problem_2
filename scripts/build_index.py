"""Compute catalogue embeddings and build the local FAISS index."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jewellery_retrieval.catalogue import load_catalogue
from jewellery_retrieval.embeddings import ImageEmbedder
from jewellery_retrieval.index import build_faiss_index, save_faiss_index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", default="data/catalogue/catalogue.csv")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    records = load_catalogue(args.catalogue)
    if not records:
        raise ValueError("Catalogue manifest is empty.")
    embedder = ImageEmbedder()
    vectors: list[np.ndarray] = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start:start + args.batch_size]
        vectors.append(embedder.embed_batch(record.image_path for record in batch))
        print(f"Embedded {min(start + args.batch_size, len(records))}/{len(records)}")
    embeddings = np.concatenate(vectors, axis=0)
    index = build_faiss_index(embeddings)
    output = Path(args.index_dir)
    output.mkdir(parents=True, exist_ok=True)
    ImageEmbedder.save(embeddings, output / "catalogue_embeddings.npy")
    save_faiss_index(index, output / "catalogue.faiss")
    shutil.copyfile(args.catalogue, output / "catalogue_manifest.csv")
    print(f"Wrote {output / 'catalogue_embeddings.npy'} shape={embeddings.shape}")
    print(f"Wrote {output / 'catalogue.faiss'} vectors={index.ntotal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())