# agent/tracking.py — Identity tracking via bbox centre distance

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from config import BBOX_CENTRE_THRESH

logger = logging.getLogger(__name__)


def _centre(bbox: List[int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


class IdentityTracker:
    """
    Assigns persistent IDs (e.g. apple_1, apple_2) to detected fruits.

    Matching strategy:
      - For each new detection, find the closest existing live ID of the
        same fruit type whose centre is within BBOX_CENTRE_THRESH pixels.
      - If no match, create a new ID.
    """

    def __init__(self):
        # fruit_type → next counter
        self._counters: Dict[str, int] = {}
        # fruit_id → last known bbox centre
        self._live: Dict[str, Tuple[float, float]] = {}

    # ------------------------------------------------------------------ #
    def assign(self, detections: List[Dict]) -> List[Dict]:
        """
        Add a "fruit_id" key to every detection dict (in-place).

        Parameters
        ----------
        detections : list of dicts with at least {"label", "bbox"}

        Returns
        -------
        The same list, each item now has "fruit_id".
        """
        # centres of this frame's detections
        new_centres = [_centre(d["bbox"]) for d in detections]

        # per fruit type: available live IDs sorted by distance to each new det
        matched: Dict[str, bool] = {fid: False for fid in self._live}
        assigned_live: Dict[int, str] = {}   # detection index → fruit_id

        # greedy nearest-neighbour matching
        for i, det in enumerate(detections):
            ftype   = det["label"]
            cen     = new_centres[i]
            best_id = None
            best_d  = float("inf")

            for fid, live_cen in self._live.items():
                if not fid.startswith(ftype + "_"):
                    continue
                if matched.get(fid, False):
                    continue          # already taken this frame
                d = _dist(cen, live_cen)
                if d < best_d and d <= BBOX_CENTRE_THRESH:
                    best_d  = d
                    best_id = fid

            if best_id is not None:
                matched[best_id] = True
                assigned_live[i] = best_id
            else:
                # new individual
                cnt = self._counters.get(ftype, 0) + 1
                self._counters[ftype] = cnt
                new_id = f"{ftype}_{cnt}"
                assigned_live[i] = new_id
                matched[new_id]   = True
                logger.debug(f"New fruit ID: {new_id}")

        # apply assignment and update live registry
        for i, det in enumerate(detections):
            fid = assigned_live[i]
            det["fruit_id"] = fid
            self._live[fid] = new_centres[i]

        return detections

    # ------------------------------------------------------------------ #
    def get_live_ids(self) -> List[str]:
        return list(self._live.keys())

    def remove(self, fruit_id: str):
        self._live.pop(fruit_id, None)

    def reset(self):
        self._counters.clear()
        self._live.clear()
        logger.info("IdentityTracker reset")
