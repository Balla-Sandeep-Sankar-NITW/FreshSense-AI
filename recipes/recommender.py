# recipes/recommender.py — Recipe recommendation logic

from __future__ import annotations
from typing import Dict, List

from recipes.recipes import RECIPES
from config import ALERT_EXPIRING


def recommend(results: List[Dict]) -> List[Dict]:
    """
    Recommend recipes that use fruits with remaining_days <= 3 (use-first fruits).

    Parameters
    ----------
    results : list of fruit group dicts from agent._build_results()

    Returns
    -------
    list of recipe dicts:
    {
        "name"       : str,
        "emoji"      : str,
        "description": str,
        "matched"    : List[str],   ← fruit types matched
        "urgency"    : str,         ← based on most urgent matched fruit
    }
    """
    # collect fruit types with remaining_days <= 3
    urgent_types = {
        r["fruit_type"]
        for r in results
        if 2 <= r["remaining_days"] <= ALERT_EXPIRING
    }

    recommendations = []
    for recipe_name, recipe in RECIPES.items():
        ingredients = recipe["ingredients"]
        required    = recipe["required"]

        if required == "all":
            matched = [i for i in ingredients if i in urgent_types]
            if not set(ingredients).issubset(urgent_types) and matched:
                # partial match is still useful — include it
                pass
            if not matched:
                continue
        else:   # "any"
            matched = [i for i in ingredients if i in urgent_types]
            if not matched:
                continue

        # urgency = worst alert among matched fruits
        urgency = _worst_urgency(matched, results)

        recommendations.append({
            "name":        recipe_name,
            "emoji":       recipe["emoji"],
            "description": recipe["description"],
            "matched":     matched,
            "urgency":     urgency,
        })

    # sort by urgency
    order = {"Rotten": 0, "Use Immediately": 1, "Expiring Soon": 2, "Fresh": 3}
    recommendations.sort(key=lambda r: order.get(r["urgency"], 4))
    return recommendations


def _worst_urgency(fruit_types: List[str], results: List[Dict]) -> str:
    order = {"Rotten": 0, "Use Immediately": 1, "Expiring Soon": 2, "Fresh": 3}
    worst = "Fresh"
    for r in results:
        if r["fruit_type"] in fruit_types:
            if order.get(r["alert"], 3) < order.get(worst, 3):
                worst = r["alert"]
    return worst
