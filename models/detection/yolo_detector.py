# models/detection/yolo_detector.py

from __future__ import annotations

import logging
from typing import List, Dict

import numpy as np

from config import (
    DETECTION_MODEL,
    YOLO_CONF_THRESH,
    YOLO_IOU_THRESH,
    LABEL_MAP,
)

logger = logging.getLogger(__name__)


class YOLODetector:
    """Wraps Ultralytics YOLO for fruit/vegetable detection."""

    def __init__(self, model_path: str = DETECTION_MODEL):
        from ultralytics import YOLO
        logger.info(f"Loading YOLO model from {model_path}")
        self.model = YOLO(model_path)
        self.conf   = YOLO_CONF_THRESH
        self.iou    = YOLO_IOU_THRESH

    # ------------------------------------------------------------------ #
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Run inference on a BGR (OpenCV) or RGB numpy image.

        Returns
        -------
        list of {
            "label"     : str   — normalised CNN label (or raw if unmapped),
            "raw_label" : str   — original YOLO class name,
            "bbox"      : [x1, y1, x2, y2],
            "confidence": float,
        }
        """
        results = self.model(
            image,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )

        detections: List[Dict] = []
        for result in results:
            boxes   = result.boxes
            names   = result.names          # {id: class_name}

            for box in boxes:
                cls_id     = int(box.cls[0].item())
                raw_label  = names[cls_id].lower().strip()
                conf_score = float(box.conf[0].item())
                xyxy       = box.xyxy[0].tolist()
                x1, y1, x2, y2 = (int(v) for v in xyxy)

                norm_label = self._normalise(raw_label)
                if norm_label is None:
                    logger.debug(f"Ignoring YOLO label: {raw_label!r}")
                    continue

                detections.append({
                    "label":      norm_label,
                    "raw_label":  raw_label,
                    "bbox":       [x1, y1, x2, y2],
                    "confidence": round(conf_score, 4),
                })

        logger.debug(f"YOLO returned {len(detections)} supported detections")
        return detections

    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalise(raw: str) -> str | None:
        """Map a raw YOLO class name to a supported CNN label, or None."""
        raw = raw.lower().strip()
        # direct hit
        if raw in LABEL_MAP:
            return LABEL_MAP[raw]
        # substring search (e.g. "red bell pepper")
        for key, val in LABEL_MAP.items():
            if key in raw:
                return val
        return None
