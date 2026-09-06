# Dyla Jewellery Visual Retrieval

A local computer-vision system for **image-level visual retrieval of jewellery** using deep image embeddings and FAISS similarity search.

The project uses the public Hugging Face dataset [`bzcasper/ai-tool-pool-jewelry-vision`](https://huggingface.co/datasets/bzcasper/ai-tool-pool-jewelry-vision) and provides a reproducible pipeline for dataset auditing, catalogue construction, embedding generation, vector indexing, evaluation, and interactive visual search.

> **Important:** This project performs **image retrieval**, not verified product identification. The source dataset provides `image` and `label` fields but does not contain verified product-level identities.

---

## Overview

Given a jewellery image, the system:

1. Processes the query image using a pretrained ResNet-50 encoder.
2. Generates a 2,048-dimensional image embedding.
3. L2-normalizes the embedding.
4. Searches a FAISS vector index using inner-product similarity.
5. Returns the most visually similar catalogue images.
6. Displays the retrieved images, category labels, similarity scores, and query latency through a local Streamlit interface.

### Retrieval Pipeline

```text
                    ┌─────────────────────┐
                    │   Query Image       │
                    │ JPG / PNG / WEBP     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   ResNet-50         │
                    │  Image Encoder      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  2,048-D Embedding  │
                    │   L2 Normalized     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   FAISS Index       │
                    │   IndexFlatIP       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Top-K Similar Images│
                    └─────────────────────┘
```

---

## Current Baseline

| Component           | Configuration                               |
| ------------------- | ------------------------------------------- |
| Dataset             | `bzcasper/ai-tool-pool-jewelry-vision`      |
| Catalogue           | 4,488 training images                       |
| Categories          | Bracelet, Earrings, Necklace, Pendant, Ring |
| Encoder             | Pretrained torchvision ResNet-50            |
| Classification head | Removed                                     |
| Embedding size      | 2,048 dimensions                            |
| Normalization       | L2                                          |
| Vector index        | FAISS `IndexFlatIP`                         |
| Similarity          | Cosine similarity                           |
| Interface           | Streamlit                                   |
| Input formats       | JPG, JPEG, PNG, WEBP                        |

Because the embeddings are L2-normalized, inner-product search with `FAISS IndexFlatIP` is equivalent to cosine-similarity retrieval.

---

## Dataset Audit

Before evaluating retrieval performance, the dataset was audited for duplicate and near-duplicate images.

### Audit Results

| Metric                           | Result |
| -------------------------------- | -----: |
| Training images                  |  4,488 |
| Validation images                |    429 |
| Test images                      |    213 |
| Total images                     |  5,130 |
| Unique SHA-256 hashes            |  5,130 |
| Exact duplicates                 |      0 |
| Perceptual-hash groups           |    104 |
| Near-duplicate pairs             | 18,859 |
| Cross-split near-duplicate pairs |  3,715 |

The presence of **3,715 cross-split near-duplicate pairs** creates a significant risk of evaluation leakage.

As a result, raw train/validation/test retrieval scores can substantially overestimate real-world performance.

For this reason, the project explicitly avoids presenting those scores as evidence of genuine product-level recognition.

---

## Product Identity vs. Image Similarity

This distinction is central to the project.

The public dataset provides:

```text
image
label
```

It does **not** provide a verified product identifier linking multiple images to the same physical jewellery item.

Therefore:

```text
JEW_000001
JEW_000002
JEW_000003
...
```

should only be interpreted as **dataset image identifiers**.

They must **not** be interpreted as:

* manufacturer product IDs
* SKU numbers
* catalogue IDs
* verified jewellery identities
* unique physical products

Consequently, the current system answers:

> **"Which catalogue images look most similar to this image?"**

It does not reliably answer:

> **"Which exact product is this?"**

---

## Repository Structure

```text
Dyla-Jewellery-Visual-Retrieval/
│
├── app/
│   └── streamlit_app.py
│
├── scripts/
│   ├── download_dataset.py
│   ├── audit_dataset.py
│   ├── build_catalogue.py
│   ├── build_evaluation_split.py
│   ├── build_index.py
│   └── evaluate_baseline.py
│
├── data/
│   └── index/              # Generated locally; not committed
│
├── reports/                # Dataset audit / evaluation reports
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

### 1. Clone the repository

```powershell
git clone <YOUR_REPOSITORY_URL>
cd Dyla-Jewellery-Visual-Retrieval
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

---

## Reproduce the Pipeline

Run the following commands in order:

```powershell
python scripts/download_dataset.py
python scripts/audit_dataset.py
python scripts/build_catalogue.py
python scripts/build_evaluation_split.py
python scripts/build_index.py
python scripts/evaluate_baseline.py
```

The first execution downloads the dataset and pretrained ResNet-50 weights.

Depending on network bandwidth, CPU performance, and local storage, the initial setup may take several minutes.

Downloaded datasets, model weights, and generated vector artifacts are intentionally excluded from version control.

---

## Run the Web Demo

Start the Streamlit application with:

```powershell
python -m streamlit run app\streamlit_app.py
```

The application supports:

* JPG
* JPEG
* PNG
* WEBP

### Demo workflow

```text
Upload Image
     │
     ▼
Generate Embedding
     │
     ▼
FAISS Similarity Search
     │
     ▼
Retrieve Top-K Candidates
     │
     ▼
Display:
  • Catalogue Image
  • Category
  • Similarity Score
  • Retrieval Latency
```

The interface is intended as a **local demonstration of the retrieval pipeline**, rather than a production deployment.

---

## Generated Artifacts

The following outputs are reproducible and intentionally ignored by Git:

```text
data/index/
```

Contains generated FAISS and NumPy indexing artifacts.

Local dataset/cache files are also excluded from version control.

Catalogue and evaluation CSV files containing machine-specific absolute cache paths are not committed.

Baseline result files containing machine-specific latency measurements are also generated locally.

### Why?

This keeps the repository:

* lightweight
* reproducible
* machine-independent
* free from large model/data artifacts
* easier to clone and review

---

## Technical Approach

### Image Encoder

The baseline uses a pretrained **ResNet-50** model from torchvision with the final classification layer removed.

The resulting feature vector contains:

```text
2048 dimensions
```

The vector is then L2-normalized before indexing.

### Vector Search

FAISS is used with:

```text
IndexFlatIP
```

Since the embeddings are normalized:

```text
Inner Product ≈ Cosine Similarity
```

This provides an exact nearest-neighbour baseline without introducing approximate-index parameters that could complicate evaluation.

---

## Evaluation Considerations

A conventional random train/test split is not sufficient for this dataset.

The audit identified substantial visual overlap between splits, including thousands of cross-split near-duplicate pairs.

This means a query image may have a visually almost identical counterpart in the catalogue even when the images belong to different dataset partitions.

Therefore:

> **High retrieval similarity does not necessarily imply successful generalization to unseen jewellery products.**

The evaluation pipeline screens candidate images for exact and perceptual overlap before interpreting retrieval behaviour.

---

## Scientific Limitations

This repository should be considered a **defensible image-level visual retrieval baseline**, not a production-grade jewellery product-recognition system.

### 1. No verified product IDs

The dataset does not provide reliable product-level identity metadata.

Therefore, product-level accuracy cannot be scientifically reported from this dataset alone.

### 2. Near-duplicate leakage

The dataset contains significant visual overlap across splits.

This can inflate conventional retrieval metrics.

### 3. Domain gap

The dataset images may differ significantly from real-world smartphone photographs.

For example:

```text
Dataset Image
     ↓
Controlled / curated image
     ↓
ResNet embedding
     ↓
FAISS retrieval
```

is not necessarily representative of:

```text
Real Jewellery
     ↓
Phone Camera
     ↓
Different lighting
     ↓
Different background
     ↓
Different orientation
     ↓
Occlusion / reflections
     ↓
Retrieval
```

### 4. Image similarity ≠ product identity

Two different jewellery products can be visually similar.

Conversely, the same physical product can look substantially different under different:

* viewing angles
* lighting conditions
* backgrounds
* camera distances
* image resolutions

---

## What Is Required for Product-Level Retrieval?

A stronger production benchmark would require a verified catalogue containing information such as:

```text
product_id
sku
category
image_path
product_images[]
```

For example:

```text
SKU-001
 ├── front.jpg
 ├── side.jpg
 └── phone_photo.jpg

SKU-002
 ├── front.jpg
 ├── side.jpg
 └── phone_photo.jpg
```

This would allow evaluation using genuine product identity rather than image similarity.

A production-oriented dataset should also contain real smartphone photographs covering different:

* angles
* distances
* lighting conditions
* backgrounds
* orientations
* camera devices

---

## Future Improvements

Potential next steps include:

* [ ] Replace the generic ResNet-50 baseline with a stronger image-retrieval encoder
* [ ] Evaluate CLIP/DINO-style embeddings
* [ ] Fine-tune embeddings using jewellery-specific data
* [ ] Introduce verified product IDs
* [ ] Build a multi-image-per-product catalogue
* [ ] Add hard-negative mining
* [ ] Evaluate Recall@K and mAP using verified product identities
* [ ] Add real smartphone query images
* [ ] Improve robustness to lighting and background changes
* [ ] Add approximate FAISS indexes for larger catalogues
* [ ] Benchmark CPU/GPU inference latency
* [ ] Package the system for production deployment

---

## Reproducibility

The project is designed so that the core pipeline can be reconstructed from the repository:

```text
Dataset
   ↓
Audit
   ↓
Catalogue
   ↓
Embeddings
   ↓
FAISS Index
   ↓
Evaluation
   ↓
Streamlit Demo
```

Generated data and model artifacts are intentionally excluded from Git because they can be regenerated locally.

---

## Responsible Interpretation

This repository does **not** claim that the baseline can identify an exact jewellery product from an arbitrary photograph.

The scientifically supported claim is narrower:

> **The system provides image-level visual similarity retrieval over the available catalogue.**

Any product-level identification claim requires a verified product-identity dataset and evaluation protocol.


## Acknowledgements

* **Hugging Face** — dataset hosting
* **PyTorch / torchvision** — pretrained ResNet-50 model
* **FAISS** — vector similarity search
* **Streamlit** — local interactive demonstration

---

## Project Status

**Status:** Research / Demonstration Baseline

The current implementation establishes a reproducible image-retrieval pipeline and documents the dataset's identity and leakage limitations. Further product-level validation requires verified catalogue metadata and real-world smartphone imagery.
