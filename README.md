# Dyla Jewellery Visual Retrieval

A local computer-vision demonstration for image-level visual retrieval over the public Hugging Face dataset `bzcasper/ai-tool-pool-jewelry-vision`.

This project deliberately distinguishes **image retrieval** from **product retrieval**. The source dataset exposes only `image` and `label`; it does not provide verified product IDs. Generated IDs such as `JEW_000001` identify dataset images only and must not be interpreted as manufacturer or product identities.

## Current baseline

- Catalogue: 4,488 train images
- Encoder: pretrained torchvision ResNet-50 with the classifier removed
- Embedding: 2,048 dimensions, L2 normalized
- Search: FAISS `IndexFlatIP`, equivalent to cosine similarity for normalized vectors
- Evaluation: validation/test candidates are screened for exact and perceptual overlap with the catalogue
- UI: local Streamlit demonstration

The audit found 3,715 cross-split near-duplicate pairs. Raw train/validation/test retrieval scores would therefore be misleading. Product-level accuracy is not reported because the dataset has no verified product identity.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The dataset and model weights are downloaded into local caches. They are not committed to Git.

## Reproduce the pipeline

```powershell
python scripts\download_dataset.py
python scripts\audit_dataset.py
python scripts\build_catalogue.py
python scripts\build_evaluation_split.py
python scripts\build_index.py
python scripts\evaluate_baseline.py
```

The first run downloads the dataset and ResNet weights and can take longer than five minutes depending on network and CPU speed.

## Launch the demonstration

```powershell
python -m streamlit run app\streamlit_app.py
```

The application accepts JPG, JPEG, PNG, and WEBP uploads and displays the top catalogue candidates, category, measured similarity, and end-to-end latency.

## Repository outputs

The audit reports are committed because they document the dataset limitations and leakage risk. The following are generated locally and intentionally ignored because they are large, machine-specific, or reproducible:

- `data/index/` FAISS and NumPy artifacts
- downloaded dataset files and caches
- local catalogue/query CSVs containing absolute cache paths
- local baseline result files containing machine-specific latency

## Scientific limitations

This is a defensible image-level retrieval baseline, not a product recognition system. The public dataset has category labels but no verified item identity, and many images may represent views, augmentations, or visually related items. A product-level benchmark requires manually verified catalogue/product metadata and real phone photographs.