"""Build the deterministic image-level catalogue manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jewellery_retrieval.catalogue import DATASET_NAME, build_catalogue


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="data/catalogue/catalogue.csv")
    args = parser.parse_args()
    records = build_catalogue(args.output, args.dataset, args.split)
    print(f"Wrote {args.output} with {len(records)} image-level catalogue records.")
    print("catalogue_id is an image-level identity, not a verified product identity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())