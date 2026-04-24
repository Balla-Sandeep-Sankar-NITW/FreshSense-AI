# recipes/recipes.py — Recipe definitions

from __future__ import annotations
from typing import Dict, List

# recipe_name → list of required fruit types (at least one must be present)
RECIPES: Dict[str, Dict] = {
    "Fruit Salad": {
        "ingredients": ["apple", "grapes"],
        "required":    "any",          # at least one ingredient present
        "emoji":       "🥗",
        "description": "Light & refreshing fruit medley. Perfect for apples and grapes.",
    },
    "Smoothie": {
        "ingredients": ["mango", "strawberry", "papaya"],
        "required":    "any",
        "emoji":       "🥤",
        "description": "Tropical vitamin boost. Blend with yogurt for extra creaminess.",
    },
    "Lemon Juice": {
        "ingredients": ["lemon"],
        "required":    "all",
        "emoji":       "🍋",
        "description": "Zesty classic. Add honey and mint for a refreshing twist.",
    },
    "Tomato Curry": {
        "ingredients": ["tomato", "paprika_pepper"],
        "required":    "any",
        "emoji":       "🍛",
        "description": "Rich & aromatic curry. Great with rice or flatbread.",
    },
    "Mixed Juice": {
        "ingredients": ["watermelon", "mango"],
        "required":    "any",
        "emoji":       "🍹",
        "description": "Sweet summer blend. Serve chilled over ice.",
    },
    "Papaya Bowl": {
        "ingredients": ["papaya"],
        "required":    "all",
        "emoji":       "🥣",
        "description": "Tropical breakfast bowl. Top with granola and coconut flakes.",
    },
    "Grape Juice": {
        "ingredients": ["grapes"],
        "required":    "all",
        "emoji":       "🍇",
        "description": "Natural antioxidant-rich juice. Serve chilled.",
    },
    "Strawberry Shake": {
        "ingredients": ["strawberry"],
        "required":    "all",
        "emoji":       "🍓",
        "description": "Creamy strawberry shake. Add vanilla ice cream for a treat.",
    },
    "Tomato Soup": {
        "ingredients": ["tomato"],
        "required":    "all",
        "emoji":       "🍲",
        "description": "Comforting classic. Best served with crusty bread.",
    },
    "Lemon Honey Drink": {
        "ingredients": ["lemon"],
        "required":    "all",
        "emoji":       "🫖",
        "description": "Soothing immune booster. Mix with warm water and honey.",
    },
    "Watermelon Juice": {
        "ingredients": ["watermelon"],
        "required":    "all",
        "emoji":       "🍉",
        "description": "Hydrating summer cooler. Add a pinch of salt for flavour.",
    },
}
