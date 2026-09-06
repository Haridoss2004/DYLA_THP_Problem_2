# Evaluation Protocol

This protocol evaluates **image-level retrieval**, not verified product retrieval. The source dataset has no product IDs.

## Catalogue

The catalogue is the train split with 4488 records. Catalogue IDs are image-level IDs.

## Candidate Queries

Candidate queries are validation and test rows: **642**.
Contaminated queries: **249**.
Clean queries: **393**.
Excluded from baseline metrics: **249**.

## Contamination Rule

A query is marked contaminated, and excluded from clean metrics, when it has an exact SHA-256 overlap, the same 64-bit average perceptual hash, or an average-hash Hamming distance <= 4 from any catalogue image.
No source image is deleted or modified. Every candidate remains in `clean_queries.csv` with its status and reason.

## Interpretation

The clean set is conservative and measures retrieval against rows without the configured image-level overlap. It still does not establish physical-product identity or generalization to a new photograph of a known product.
