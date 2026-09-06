"""Pretrained image embedding extraction for the retrieval baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision.models import ResNet50_Weights, resnet50


class ImageEmbedder:
    """ResNet-50 ImageNet representation with normalized output vectors."""

    def __init__(self, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        weights = ResNet50_Weights.DEFAULT
        self.preprocess = weights.transforms()
        self.model = resnet50(weights=weights)
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(self.device)
        self.dimension = 2048

    @staticmethod
    def _load_image(image: Image.Image | str | Path) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        try:
            with Image.open(path) as loaded:
                return loaded.convert("RGB")
        except UnidentifiedImageError as exc:
            raise ValueError(f"Unsupported or corrupt image: {path}") from exc

    def embed_batch(self, images: Iterable[Image.Image | str | Path]) -> np.ndarray:
        loaded = [self._load_image(image) for image in images]
        if not loaded:
            return np.empty((0, self.dimension), dtype=np.float32)
        batch = torch.stack([self.preprocess(image) for image in loaded]).to(self.device)
        with torch.inference_mode():
            output = self.model(batch)
            output = torch.nn.functional.normalize(output, p=2, dim=1)
        return output.cpu().numpy().astype(np.float32)

    def embed(self, image: Image.Image | str | Path) -> np.ndarray:
        return self.embed_batch([image])[0]

    @staticmethod
    def save(embeddings: np.ndarray, path: str | Path) -> None:
        array = np.asarray(embeddings, dtype=np.float32)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array)

    @staticmethod
    def load(path: str | Path) -> np.ndarray:
        return np.load(path).astype(np.float32)