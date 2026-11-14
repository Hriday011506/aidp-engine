"""Explainable product-seasonality signals for OptiRetail's demo decision engine.

These rules are transparent heuristics, not a trained retail elasticity model.
Replace/tune them with SKU/store sales history when connected data is available.
"""
import re

CATEGORY_RULES = [
    ("ice_cream", ["ice cream", "kulfi", "frozen dessert", "gelato"]),
    ("blanket", ["blanket", "razai", "comforter", "quilt", "duvet"]),
    ("winter_wear", ["sweater", "jacket", "woollen", "woolen", "thermal", "hoodie"]),
    ("heater", ["room heater", "heating appliance", "heater"]),
    ("air_conditioner", ["air conditioner", "air cooler", "ac "]),
    ("cold_drink", ["cold drink", "soft drink", "beverage", "juice", "lassi"]),
    ("fan", ["ceiling fan", "table fan", "pedestal fan"]),
    ("sunscreen", ["sunscreen", "sun screen"]),
    ("umbrella_rainwear", ["umbrella", "raincoat", "rain coat", "gumboots"]),
    ("soup", ["soup", "hot chocolate"]),
]

def classify_product(product):
    p = re.sub(r"[^a-z0-9 ]", " ", str(product).lower())
    for category, terms in CATEGORY_RULES:
        if any(term in p for term in terms):
            return category
    return "general"

def seasonal_demand_multiplier(category, temperature_c):
    """Relative demand multiplier from temperature; intentionally conservative."""
    t = float(temperature_c)
    rules = {
        "ice_cream": (0.72 if t <= 15 else 0.86 if t <= 20 else 1.0 if t <= 27 else 1.12 if t <= 33 else 1.22),
        "blanket": (1.24 if t <= 15 else 1.14 if t <= 20 else 1.0 if t <= 27 else 0.91 if t <= 33 else 0.86),
        "winter_wear": (1.20 if t <= 15 else 1.12 if t <= 20 else 1.0 if t <= 27 else 0.90),
        "heater": (1.22 if t <= 15 else 1.12 if t <= 20 else 1.0 if t <= 27 else 0.92),
        "air_conditioner": (0.82 if t <= 20 else 0.94 if t <= 25 else 1.08 if t <= 32 else 1.18),
        "cold_drink": (0.82 if t <= 15 else 0.92 if t <= 20 else 1.0 if t <= 27 else 1.10 if t <= 33 else 1.18),
        "fan": (0.82 if t <= 15 else 0.92 if t <= 20 else 1.0 if t <= 27 else 1.10 if t <= 33 else 1.16),
        "sunscreen": (0.88 if t <= 15 else 0.96 if t <= 20 else 1.0 if t <= 27 else 1.08 if t <= 33 else 1.12),
        "umbrella_rainwear": 1.0,
        "soup": (1.18 if t <= 15 else 1.10 if t <= 20 else 1.0 if t <= 27 else 0.92),
        "general": 1.0,
    }
    return float(rules.get(category, rules["general"]))

def explain_seasonality(category, temperature_c, multiplier):
    if multiplier < 0.97:
        return f"Weather is currently less favourable for {category.replace('_', ' ')} demand ({temperature_c:.1f}°C)."
    if multiplier > 1.03:
        return f"Weather is currently favourable for {category.replace('_', ' ')} demand ({temperature_c:.1f}°C)."
    return f"No material temperature-driven demand change applied ({temperature_c:.1f}°C)."

def pricing_adjustment(market_price, demand_multiplier, category, mrp):
    """Small weather-driven competitive adjustment; final output never exceeds MRP."""
    # Keep the movement modest until SKU-level price elasticity is learned.
    weather_component = max(-0.04, min(0.04, (float(demand_multiplier) - 1.0) * 0.20))
    category_component = -0.01 if category in {"ice_cream", "blanket", "winter_wear"} and demand_multiplier < 0.95 else 0.0
    raw = float(market_price) * (1.0 + weather_component + category_component)
    return round(max(0.01, min(raw, float(mrp))), 2), weather_component + category_component
