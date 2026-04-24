# agent/learning.py — Two-level learning system

from __future__ import annotations

import logging
from typing import Dict

from config import ALPHA_INIT, ALPHA_SHELF, FRAME_LR

logger = logging.getLogger(__name__)


# ─── Level 1: Frame-level fusion ─────────────────────────────────────────── #

def level1_fuse(
    cnn_day:      int,
    last_day:     int,
    alpha:        float,
    fruit_id:     str,
) -> Dict:
    """
    Fuse CNN prediction with temporal expectation.

    expected_day = last_day + 1
    final_day    = α × cnn_day + (1-α) × expected_day

    Updates α based on error.

    Returns
    -------
    {"final_day": int, "alpha": float, "expected_day": float}
    """
    expected_day = last_day + 1
    final_day    = alpha * cnn_day + (1 - alpha) * expected_day

    # update reliability
    error = abs(final_day - expected_day)
    new_alpha = alpha - FRAME_LR * error
    new_alpha = max(0.0, min(1.0, new_alpha))

    logger.debug(
        f"[L1] {fruit_id} | cnn={cnn_day} exp={expected_day:.1f} "
        f"final={final_day:.2f} α {alpha:.3f}→{new_alpha:.3f}"
    )

    return {
        "final_day":    round(final_day),
        "alpha":        new_alpha,
        "expected_day": expected_day,
    }


# ─── Level 2: Shelf life learning ────────────────────────────────────────── #

def level2_update_shelf(
    fruit_type:     str,
    first_seen_day: int,
    last_seen_day:  int,
    predicted_life: float,
    confidence:     float,
    current_shelf:  float,
) -> float:
    """
    Update the learned shelf life when a fruit disappears (assumed consumed/expired).

    observed_life = last_seen_day - first_seen_day
    error         = observed_life - predicted_life
    new_shelf     = current_shelf + α_shelf × error × confidence

    Returns updated shelf life (≥ 3).
    """
    observed_life = last_seen_day - first_seen_day
    error         = observed_life - predicted_life
    delta         = ALPHA_SHELF * error * confidence
    new_shelf     = max(3.0, current_shelf + delta)

    logger.info(
        f"[L2] {fruit_type} | observed={observed_life} predicted={predicted_life:.1f} "
        f"error={error:.2f} Δ={delta:.3f} shelf {current_shelf:.1f}→{new_shelf:.1f}"
    )
    return new_shelf
