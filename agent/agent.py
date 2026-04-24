# agent/agent.py — FreshSense AI orchestration agent

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from config import (
    ALPHA_INIT,
    DEFAULT_SHELF_LIFE,
    ALERT_ROTTEN,
    ALERT_USE_NOW,
    ALERT_EXPIRING,
)
from models.detection.yolo_detector import YOLODetector
from models.model_loader import get_registry
from agent.tracking import IdentityTracker
from agent.memory import MemoryStore
from agent.learning import level1_fuse, level2_update_shelf

logger = logging.getLogger(__name__)


def _alert_label(remaining: int) -> str:
    if remaining <= 2:
        return "Rotten"
    elif remaining <= ALERT_USE_NOW:
        return "Use Immediately"
    elif remaining <= ALERT_EXPIRING:
        return "Expiring Soon"
    else:
        return "Fresh"


class FreshSenseAgent:
    """End-to-end agentic pipeline for fruit freshness analysis."""

    def __init__(self, device=None):
        import torch
        self.device    = device or torch.device("cpu")
        self.detector  = YOLODetector()
        self.registry  = get_registry(device=self.device)
        self.tracker   = IdentityTracker()
        self.memory    = MemoryStore()
        self.global_day: int = 0
        self.events: List[str] = []

    # ------------------------------------------------------------------ #
    #  Day controls
    # ------------------------------------------------------------------ #

    def increment_day(self):
        self.global_day += 1
        self._log(f"Day incremented → Day {self.global_day}")

    def decrement_day(self):
        self.global_day = max(0, self.global_day - 1)
        self._log(f"Day decremented → Day {self.global_day}")

    def reset_day(self):
        self.global_day = 0
        self.memory.reset()
        self.tracker.reset()
        self.events.clear()
        self._log("Full reset — Day 0, memory cleared")

    # ------------------------------------------------------------------ #
    #  Main processing pipeline
    # ------------------------------------------------------------------ #

    def process(self, image: np.ndarray) -> Dict:
        """
        Run the full pipeline on a single image.

        Parameters
        ----------
        image : np.ndarray  — BGR (OpenCV) image

        Returns
        -------
        {
            "results"      : List[Dict],   ← one entry per unique fruit group
            "annotated"    : np.ndarray,   ← image with bboxes drawn
            "events"       : List[str],
            "learned_shelf": Dict,
            "global_day"   : int,
        }
        """
        self.events = []

        # ── 1. Detect ───────────────────────────────────────────────────
        raw_detections = self.detector.detect(image)
        self._log(f"Detected {len(raw_detections)} supported fruits (Day {self.global_day})")

        # ── 2. Assign / match identities ────────────────────────────────
        detections = self.tracker.assign(raw_detections)

        # ── 3. Track which IDs were present yesterday ────────────────────
        prev_ids = set(self.memory.get_all_ids())
        curr_ids = {d["fruit_id"] for d in detections}

        # ── 4. Predict day for each detection, Level-1 fusion ────────────
        for det in detections:
            fid    = det["fruit_id"]
            ftype  = det["label"]
            bbox   = det["bbox"]
            x1,y1,x2,y2 = bbox

            # crop
            crop = image[max(0,y1):y2, max(0,x1):x2]
            if crop.size == 0:
                crop = image

            pred = self.registry.predict(
                ftype, crop,
                self.memory.learned_shelf,
            )
            cnn_day    = pred["predicted_day"]
            confidence = pred["confidence"]

            # Level-1 fusion
            record = self.memory.get(fid)
            if record is not None:
                last_day = record["predicted_day"]
                alpha    = record.get("alpha", ALPHA_INIT)
            else:
                last_day = max(cnn_day - 1, 0)
                alpha    = ALPHA_INIT

            fusion = level1_fuse(cnn_day, last_day, alpha, fid)
            final_day = fusion["final_day"]
            new_alpha = fusion["alpha"]

            # remaining days
            shelf    = self.memory.learned_shelf.get(ftype, DEFAULT_SHELF_LIFE.get(ftype, 10))
            remaining = int(shelf) - final_day

            # build record
            if record is None:
                # first time we see this fruit
                predicted_life = float(remaining)
                first_seen_day = self.global_day
                self._log(f"New fruit: {fid} first seen on Day {first_seen_day}")
            else:
                predicted_life = record.get("predicted_life", float(remaining))
                first_seen_day = record.get("first_seen_day", self.global_day)

            new_record = {
                "fruit_id":       fid,
                "fruit_type":     ftype,
                "predicted_day":  final_day,
                "confidence":     confidence,
                "last_seen_day":  self.global_day,
                "first_seen_day": first_seen_day,
                "predicted_life": predicted_life,
                "bbox":           bbox,
                "alpha":          new_alpha,
                "remaining_days": remaining,
                "alert":          _alert_label(remaining),
                "model_found":    pred["model_found"],
            }
            self.memory.upsert(new_record)

        # ── 5. Missing fruit handling (Level-2 learning) ─────────────────
        missing_ids = prev_ids - curr_ids
        for fid in missing_ids:
            record = self.memory.get(fid)
            if record is None:
                continue
            ftype       = record["fruit_type"]
            first_day   = record.get("first_seen_day", 0)
            last_day    = record.get("last_seen_day", self.global_day - 1)
            pred_life   = record.get("predicted_life", 1.0)
            conf        = record.get("confidence", 0.5)
            cur_shelf   = self.memory.learned_shelf.get(
                ftype, DEFAULT_SHELF_LIFE.get(ftype, 10)
            )

            new_shelf = level2_update_shelf(
                ftype, first_day, last_day, pred_life, conf, cur_shelf
            )
            self.memory.update_shelf(ftype, new_shelf)
            self.memory.remove(fid)
            self.tracker.remove(fid)
            self._log(
                f"[Missing] {fid} consumed/expired on Day {self.global_day}. "
                f"Shelf life updated: {cur_shelf:.1f}→{new_shelf:.1f}"
            )

        # ── 6. Build results ─────────────────────────────────────────────
        results = self._build_results()

        # ── 7. Annotate image ────────────────────────────────────────────
        annotated = self._annotate(image, self.memory.all_fruits())

        # ── 8. Persist ───────────────────────────────────────────────────
        self.memory.save()

        return {
            "results":       results,
            "annotated":     annotated,
            "events":        list(self.events),
            "learned_shelf": dict(self.memory.learned_shelf),
            "global_day":    self.global_day,
        }

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _build_results(self) -> List[Dict]:
        """Aggregate individual fruit IDs into display groups (type+day+alert)."""
        from collections import defaultdict

        groups: Dict[str, Dict] = {}
        for rec in self.memory.all_fruits():
            key = f"{rec['fruit_type']}__day{rec['predicted_day']}__alert{rec['alert']}"
            if key not in groups:
                groups[key] = {
                    "fruit_type":    rec["fruit_type"],
                    "predicted_day": rec["predicted_day"],
                    "remaining_days":rec["remaining_days"],
                    "alert":         rec["alert"],
                    "confidence_sum":rec["confidence"],
                    "count":         1,
                    "ids":           [rec["fruit_id"]],
                }
            else:
                groups[key]["confidence_sum"] += rec["confidence"]
                groups[key]["count"]          += 1
                groups[key]["ids"].append(rec["fruit_id"])

        results = []
        for g in groups.values():
            g["avg_confidence"] = round(g["confidence_sum"] / g["count"], 3)
            del g["confidence_sum"]
            results.append(g)

        # sort: rotten → use now → expiring → fresh
        order = {"Rotten": 0, "Use Immediately": 1, "Expiring Soon": 2, "Fresh": 3}
        results.sort(key=lambda r: order.get(r["alert"], 4))
        return results

    def _annotate(self, image: np.ndarray, fruits: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on a copy of the image."""
        import cv2
        out = image.copy()

        colour_map = {
            "Rotten":         (0,   0,  220),   # red
            "Use Immediately":(0,  140, 255),   # orange
            "Expiring Soon":  (0,  215, 255),   # yellow
            "Fresh":          (0,  200,  80),   # green
        }

        for rec in fruits:
            bbox   = rec.get("bbox")
            if not bbox:
                continue
            x1,y1,x2,y2 = bbox
            alert  = rec.get("alert", "Fresh")
            colour = colour_map.get(alert, (200, 200, 200))

            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

            label  = (
                f"{rec['fruit_id']} | Day {rec['predicted_day']} "
                f"| {alert} | rem {rec['remaining_days']}d"
            )
            cv2.putText(
                out, label,
                (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48, colour, 1, cv2.LINE_AA,
            )
        return out

    def _log(self, msg: str):
        logger.info(msg)
        self.events.append(msg)
