# agent/memory.py — Persistent JSON memory store for tracked fruits

from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from typing import Dict, List, Optional

from config import MEMORY_PATH, DEFAULT_SHELF_LIFE, ALPHA_INIT

logger = logging.getLogger(__name__)


class MemoryStore:
    """
    Stores per-fruit-ID state and per-type learned shelf lives.

    Schema per fruit record:
    {
        "fruit_id"      : str,
        "fruit_type"    : str,
        "predicted_day" : int,      ← final_day after Level-1 fusion
        "confidence"    : float,
        "last_seen_day" : int,
        "first_seen_day": int,
        "predicted_life": float,    ← remaining_days predicted at first detection
        "bbox"          : [x1,y1,x2,y2],
        "alpha"         : float,    ← per-ID CNN reliability weight
    }
    """

    def __init__(self, path: str = MEMORY_PATH):
        self.path           = path
        self._fruits: Dict[str, Dict]  = {}         # fruit_id → record
        self._shelf:  Dict[str, float] = deepcopy(DEFAULT_SHELF_LIFE)
        self._load()

    # ─── Persistence ──────────────────────────────────────────────────── #

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
                self._fruits = data.get("fruits", {})
                saved_shelf  = data.get("learned_shelf", {})
                self._shelf.update(saved_shelf)
                logger.info(f"Memory loaded from {self.path}")
            except Exception as e:
                logger.warning(f"Could not load memory: {e}")

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(
                {"fruits": self._fruits, "learned_shelf": self._shelf},
                f, indent=2,
            )

    # ─── Fruit records ────────────────────────────────────────────────── #

    def has(self, fruit_id: str) -> bool:
        return fruit_id in self._fruits

    def get(self, fruit_id: str) -> Optional[Dict]:
        return self._fruits.get(fruit_id)

    def all_fruits(self) -> List[Dict]:
        return list(self._fruits.values())

    def upsert(self, record: Dict):
        """Insert or update a fruit record."""
        fid = record["fruit_id"]
        if fid not in self._fruits:
            record.setdefault("alpha", ALPHA_INIT)
        self._fruits[fid] = record

    def remove(self, fruit_id: str):
        self._fruits.pop(fruit_id, None)

    def get_all_ids(self) -> List[str]:
        return list(self._fruits.keys())

    # ─── Shelf lives ──────────────────────────────────────────────────── #

    @property
    def learned_shelf(self) -> Dict[str, float]:
        return self._shelf

    def update_shelf(self, fruit_type: str, new_val: float):
        self._shelf[fruit_type] = max(3.0, new_val)

    # ─── Reset ────────────────────────────────────────────────────────── #

    def reset(self):
        self._fruits.clear()
        self._shelf = deepcopy(DEFAULT_SHELF_LIFE)
        if os.path.exists(self.path):
            os.remove(self.path)
        logger.info("MemoryStore reset")
