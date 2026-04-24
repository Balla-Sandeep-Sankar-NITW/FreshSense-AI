# utils/image_utils.py — Image helpers

from __future__ import annotations

import io
from typing import Optional

import cv2
import numpy as np
from PIL import Image


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to a BGR numpy array (OpenCV format)."""
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert a BGR numpy array to a PIL RGB image."""
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def bytes_to_bgr(data: bytes) -> Optional[np.ndarray]:
    """Decode image bytes (JPEG / PNG / etc.) to BGR numpy array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def bgr_to_bytes(arr: np.ndarray, ext: str = ".jpg") -> bytes:
    """Encode a BGR array to image bytes."""
    ok, buf = cv2.imencode(ext, arr)
    if not ok:
        raise ValueError("cv2.imencode failed")
    return buf.tobytes()


def safe_crop(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Return a safe crop; falls back to full image if crop is empty."""
    h, w = image.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    crop = image[y1:y2, x1:x2]
    return crop if crop.size > 0 else image
