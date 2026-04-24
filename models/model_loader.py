# models/model_loader.py — CNN (EfficientNet-B0) loader & predictor

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms

from config import (
    PREDICTIONS_DIR,
    SUPPORTED_FRUITS,
    CNN_CUTOFF,
    CNN_DAY_BIAS,
    DEFAULT_SHELF_LIFE,
    CNN_IMG_SIZE,
    CNN_MEAN,
    CNN_STD,
)

logger = logging.getLogger(__name__)


# ─── Architecture ─────────────────────────────────────────────────────────── #

def build_model() -> nn.Module:
    """Build EfficientNet-B0 with custom freshness head (matches training)."""
    m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    # freeze base
    for p in m.parameters():
        p.requires_grad = False
    # unfreeze top blocks
    for p in m.features[5:].parameters():
        p.requires_grad = True

    in_feat = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_feat, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid(),   # sigmoid IS inside the model
    )
    return m


# ─── Transform ────────────────────────────────────────────────────────────── #

_transform = transforms.Compose([
    transforms.Resize((CNN_IMG_SIZE, CNN_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=CNN_MEAN, std=CNN_STD),
])


def preprocess(crop: np.ndarray | Image.Image) -> torch.Tensor:
    """Convert a numpy BGR crop or PIL image to a model-ready tensor."""
    if isinstance(crop, np.ndarray):
        import cv2
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil  = Image.fromarray(crop)
    else:
        pil = crop.convert("RGB")
    return _transform(pil).unsqueeze(0)   # (1, C, H, W)


# ─── Model registry ───────────────────────────────────────────────────────── #

class ModelRegistry:
    """Lazy-loads and caches one CNN model per fruit type."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device  = device or torch.device("cpu")
        self._cache: Dict[str, nn.Module] = {}

    def _load(self, fruit_type: str) -> Optional[nn.Module]:
        path = os.path.join(PREDICTIONS_DIR, f"{fruit_type}.pt")
        if not os.path.exists(path):
            logger.warning(f"Model not found: {path}")
            return None
        try:
            model = build_model()
            ckpt  = torch.load(path, map_location=self.device)
            state = ckpt["state_dict"]
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()
            logger.info(f"Loaded CNN model for {fruit_type}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model for {fruit_type}: {e}")
            return None

    def get(self, fruit_type: str) -> Optional[nn.Module]:
        if fruit_type not in self._cache:
            self._cache[fruit_type] = self._load(fruit_type)
        return self._cache[fruit_type]

    # ------------------------------------------------------------------ #
    def predict(
        self,
        fruit_type: str,
        crop: np.ndarray | Image.Image,
        learned_shelf: Dict[str, float],
    ) -> Dict:
        """
        Predict the current age (day) of a fruit crop.

        Returns
        -------
        {
            "predicted_day" : int,
            "confidence"    : float,
            "raw_norm"      : float,
            "model_found"   : bool,
        }
        """
        model = self.get(fruit_type)
        shelf = learned_shelf.get(fruit_type, DEFAULT_SHELF_LIFE.get(fruit_type, 10))
        cutoff = CNN_CUTOFF.get(fruit_type, int(shelf))

        if model is None:
            # fallback: assume day 1 with zero confidence
            return {
                "predicted_day": 1,
                "confidence":    0.0,
                "raw_norm":      0.5,
                "model_found":   False,
            }

        x    = preprocess(crop).to(self.device)
        with torch.no_grad():
            norm = model(x).item()  # sigmoid already applied inside model

        raw_day    = max(1, min(cutoff, round(norm * cutoff)))
        # apply bias correction
        biased_day = min(raw_day + CNN_DAY_BIAS, int(shelf))
        confidence = abs(norm - 0.5) * 2

        return {
            "predicted_day": biased_day,
            "confidence":    round(confidence, 4),
            "raw_norm":      round(norm, 4),
            "model_found":   True,
        }


# ─── Singleton ────────────────────────────────────────────────────────────── #
_registry: Optional[ModelRegistry] = None

def get_registry(device: Optional[torch.device] = None) -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry(device=device)
    return _registry
