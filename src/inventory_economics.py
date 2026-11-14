"""Inventory-aware pricing economics for OptiRetail AI.

Uses a simple monthly demand curve derived from the estimated price elasticity.
The user supplies stock, remaining shelf life and carrying/write-off costs.
This is a planning scenario, not a replenishment or expiry guarantee.
"""
import numpy as np


def optimize_price_with_inventory(
    market_price, mrp, unit_cost, elasticity_result, baseline_units,
    on_hand_units, days_to_expiry=30, holding_cost_per_unit=0.0,
    expiry_writeoff_per_unit=0.0, horizon_days=30,
):
    if not elasticity_result.get("reliable"):
        raise ValueError("A reliable historical elasticity estimate is required.")
    market, cap, cost = float(market_price), float(mrp), float(unit_cost)
    stock = float(on_hand_units)
    expiry = float(days_to_expiry)
    holding = float(holding_cost_per_unit)
    writeoff = float(expiry_writeoff_per_unit)
    horizon = float(horizon_days)
    baseline = float(baseline_units)
    elasticity = float(elasticity_result["elasticity"])
    if min(market, cap, cost, horizon) <= 0 or stock < 0 or expiry < 0 or holding < 0 or writeoff < 0 or baseline <= 0:
        raise ValueError("Enter positive market price, MRP, unit cost, forecast and horizon; inventory and cost inputs cannot be negative.")
    ceiling = min(market, cap)
    if cost >= ceiling:
        raise ValueError("Unit cost is at or above the allowed selling-price ceiling; no positive gross margin is possible.")
    floor = min(ceiling, max(cost, ceiling * 0.70))
    candidates = np.linspace(floor, ceiling, 61) if floor < ceiling else np.array([ceiling])
    reference_price = market
    expected_demand = baseline * (candidates / reference_price) ** elasticity
    sell_window = min(expiry, horizon) if expiry > 0 else horizon
    window_demand = expected_demand * min(1.0, sell_window / horizon)
    sold = np.minimum(stock, window_demand)
    unsold = np.maximum(0.0, stock - sold)
    expires_within_horizon = expiry > 0 and expiry <= horizon
    expired_units = unsold if expires_within_horizon else np.zeros_like(unsold)
    held_units = np.zeros_like(unsold) if expires_within_horizon else unsold
    revenue = candidates * sold
    cost_of_sold = cost * sold
    writeoff_cost = writeoff * expired_units
    carrying_cost = holding * held_units
    gross_profit_after_inventory_costs = revenue - cost_of_sold - writeoff_cost - carrying_cost
    index = int(np.argmax(gross_profit_after_inventory_costs))
    return {
        "recommended_price": round(float(candidates[index]), 2),
        "expected_demand": round(float(expected_demand[index]), 1),
        "expected_units_sold": round(float(sold[index]), 1),
        "unmet_demand": round(float(max(0.0, window_demand[index] - sold[index])), 1),
        "unsold_units": round(float(unsold[index]), 1),
        "expected_expired_units": round(float(expired_units[index]), 1),
        "expected_revenue": round(float(revenue[index]), 2),
        "gross_profit_after_inventory_costs": round(float(gross_profit_after_inventory_costs[index]), 2),
        "estimated_writeoff_cost": round(float(writeoff_cost[index]), 2),
        "estimated_holding_cost": round(float(carrying_cost[index]), 2),
        "price_floor": round(float(floor), 2),
        "price_ceiling": round(float(ceiling), 2),
        "sell_window_days": round(float(sell_window), 1),
        "objective": "Revenue less cost of sold units, estimated expiry write-off and carrying cost",
    }
