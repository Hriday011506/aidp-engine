"""Historical price-elasticity estimation and profit-aware price optimization.

Expected CSV columns: date (optional), product (optional), units, selling_price.
Optional columns: promo (0/1), stock_available (0/1). Stockout rows are excluded
when stock_available is supplied. Estimates are observational, not causal.
"""
import numpy as np
import pandas as pd


def prepare_sales_history(frame, product=None):
    required = {"units", "selling_price"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("CSV must contain: units, selling_price.")
    data = frame.copy()
    for col in ["units", "selling_price"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["units", "selling_price"])
    data = data[(data["units"] > 0) & (data["selling_price"] > 0)]
    if product and "product" in data.columns:
        data = data[data["product"].astype(str).str.casefold() == str(product).casefold()]
    if "stock_available" in data.columns:
        available = data["stock_available"].astype(str).str.lower().isin(["1", "true", "yes", "y"])
        data = data[available]
    if "promo" not in data.columns:
        data["promo"] = 0.0
    data["promo"] = pd.to_numeric(data["promo"], errors="coerce").fillna(0).clip(0, 1)
    if "date" in data.columns:
        dates = pd.to_datetime(data["date"], errors="coerce")
        data = data.assign(_date=dates).dropna(subset=["_date"]).sort_values("_date")
        if len(data):
            data["_trend"] = np.arange(len(data), dtype=float) / max(len(data) - 1, 1)
    else:
        data["_trend"] = np.arange(len(data), dtype=float) / max(len(data) - 1, 1)
    return data


def estimate_elasticity(frame, product=None):
    data = prepare_sales_history(frame, product)
    unique_prices = int(data["selling_price"].round(2).nunique())
    if len(data) < 8 or unique_prices < 3:
        return {"reliable": False, "reason": "Need at least 8 valid observations and 3 distinct selling prices.", "observations": len(data), "unique_prices": unique_prices}
    y = np.log(data["units"].to_numpy(dtype=float))
    log_price = np.log(data["selling_price"].to_numpy(dtype=float))
    # Control for promotions and a simple time trend where available.
    x = np.column_stack([np.ones(len(data)), log_price, data["promo"].to_numpy(dtype=float), data["_trend"].to_numpy(dtype=float)])
    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ coef
    ss_total = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - float(np.sum((y - fitted) ** 2)) / ss_total if ss_total > 1e-12 else 0.0
    elasticity = float(coef[1])
    reliable = bool(np.isfinite(elasticity) and elasticity < -0.05 and r_squared >= 0.10)
    reason = "Historical price response estimated with promotion and time controls." if reliable else "Price response is weak, positive, or poorly explained; do not optimize from this sample."
    return {
        "reliable": reliable, "reason": reason, "elasticity": elasticity,
        "r_squared": max(0.0, min(1.0, r_squared)), "observations": len(data),
        "unique_prices": unique_prices, "average_units": float(data["units"].mean()),
        "average_price": float(data["selling_price"].mean()),
        "promotion_coefficient": float(coef[2]),
    }


def optimize_price(market_price, mrp, unit_cost, elasticity_result, baseline_units):
    if not elasticity_result.get("reliable"):
        raise ValueError("A reliable historical elasticity estimate is required.")
    market, cap, cost = float(market_price), float(mrp), float(unit_cost)
    if cost <= 0 or cap <= 0 or market <= 0:
        raise ValueError("Market price, MRP and unit cost must be positive.")
    ceiling = min(market, cap)
    if cost >= ceiling:
        raise ValueError("Unit cost is at or above the allowed selling-price ceiling; no positive gross margin is possible.")
    if float(baseline_units) <= 0:
        raise ValueError("Baseline forecast units must be positive.")
    floor = min(ceiling, max(cost, ceiling * 0.70))
    if floor >= ceiling:
        candidates = np.array([ceiling])
    else:
        candidates = np.linspace(floor, ceiling, 61)
    reference_price = market  # Current forecast is treated as demand at the current market reference price.
    reference_units = float(baseline_units)
    elasticity = float(elasticity_result["elasticity"])
    expected_units = reference_units * (candidates / reference_price) ** elasticity
    expected_units = np.maximum(0, expected_units)
    profit = (candidates - cost) * expected_units
    revenue = candidates * expected_units
    index = int(np.argmax(profit))
    return {
        "recommended_price": round(float(candidates[index]), 2),
        "expected_units": round(float(expected_units[index]), 1),
        "expected_revenue": round(float(revenue[index]), 2),
        "expected_gross_profit": round(float(profit[index]), 2),
        "candidate_count": len(candidates),
        "price_floor": round(float(floor), 2),
        "price_ceiling": round(float(ceiling), 2),
        "reference_price": round(reference_price, 2),
        "reference_units": round(reference_units, 1),
    }
