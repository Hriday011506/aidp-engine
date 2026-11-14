"""Multi-store inventory balancing utilities for OptiRetail AI.

The first release uses editable demo data. The calculations are deterministic and
can later be fed from POS, ERP, WMS or MongoDB collections.
"""
from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {"Store", "SKU", "Product", "On hand", "Forecast demand", "Lead time (days)"}


def demo_inventory() -> pd.DataFrame:
    return pd.DataFrame([
        ["Jaipur Central", "SKU-1001", "Wheat Flour 5kg", 620, 420, 3],
        ["Jaipur Central", "SKU-1002", "Cooking Oil 1L", 180, 310, 4],
        ["Jaipur Central", "SKU-1003", "Rice 5kg", 510, 350, 3],
        ["Ajmer Road", "SKU-1001", "Wheat Flour 5kg", 150, 330, 5],
        ["Ajmer Road", "SKU-1002", "Cooking Oil 1L", 460, 250, 3],
        ["Ajmer Road", "SKU-1003", "Rice 5kg", 190, 280, 4],
        ["Vaishali Nagar", "SKU-1001", "Wheat Flour 5kg", 390, 280, 2],
        ["Vaishali Nagar", "SKU-1002", "Cooking Oil 1L", 130, 240, 5],
        ["Vaishali Nagar", "SKU-1003", "Rice 5kg", 420, 300, 2],
        ["Mansarovar", "SKU-1001", "Wheat Flour 5kg", 120, 260, 4],
        ["Mansarovar", "SKU-1002", "Cooking Oil 1L", 350, 220, 3],
        ["Mansarovar", "SKU-1003", "Rice 5kg", 140, 250, 5],
    ], columns=["Store", "SKU", "Product", "On hand", "Forecast demand", "Lead time (days)"])


def validate_inventory(frame: pd.DataFrame) -> tuple[bool, str]:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        return False, "Missing columns: " + ", ".join(sorted(missing))
    if frame.empty:
        return False, "Add at least one store/SKU row."
    for column in ["On hand", "Forecast demand", "Lead time (days)"]:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or (values < 0).any():
            return False, f"{column} must contain non-negative numbers."
    if frame[["Store", "SKU", "Product"]].isna().any().any():
        return False, "Store, SKU and Product cannot be blank."
    if frame.duplicated(["Store", "SKU"]).any():
        return False, "Each Store + SKU combination must be unique."
    return True, ""


def calculate_inventory_position(frame: pd.DataFrame, safety_stock_pct: float = 10) -> pd.DataFrame:
    valid, message = validate_inventory(frame)
    if not valid:
        raise ValueError(message)
    result = frame.copy()
    for column in ["On hand", "Forecast demand", "Lead time (days)"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["Safety stock"] = (result["Forecast demand"] * safety_stock_pct / 100).round().astype(int)
    result["Target stock"] = (result["Forecast demand"] + result["Safety stock"]).round().astype(int)
    result["Surplus"] = (result["On hand"] - result["Target stock"]).clip(lower=0).astype(int)
    result["Shortage"] = (result["Target stock"] - result["On hand"]).clip(lower=0).astype(int)
    result["Coverage (days)"] = result.apply(
        lambda row: round(row["On hand"] / (row["Forecast demand"] / 30), 1)
        if row["Forecast demand"] > 0 else 999.0, axis=1
    )
    result["Risk"] = result.apply(
        lambda row: "Critical" if row["Shortage"] > 0 and row["Lead time (days)"] >= 4
        else "At risk" if row["Shortage"] > 0
        else "Surplus" if row["Surplus"] > 0 else "Balanced", axis=1
    )
    return result


def recommend_transfers(position: pd.DataFrame, transfer_cost_per_unit: float = 2.5) -> pd.DataFrame:
    """Greedy same-SKU matching: largest shortage first, then available donor surplus."""
    recommendations = []
    for sku, group in position.groupby("SKU", sort=True):
        donors = group[group["Surplus"] > 0].sort_values("Surplus", ascending=False).to_dict("records")
        receivers = group[group["Shortage"] > 0].sort_values(
            ["Lead time (days)", "Shortage"], ascending=[False, False]
        ).to_dict("records")
        donor_remaining = {row["Store"]: int(row["Surplus"]) for row in donors}
        receiver_remaining = {row["Store"]: int(row["Shortage"]) for row in receivers}
        for receiver in receivers:
            for donor in donors:
                available = donor_remaining[donor["Store"]]
                needed = receiver_remaining[receiver["Store"]]
                if available <= 0 or needed <= 0 or donor["Store"] == receiver["Store"]:
                    continue
                quantity = min(available, needed)
                recommendations.append({
                    "SKU": sku, "Product": receiver["Product"],
                    "From store": donor["Store"], "To store": receiver["Store"],
                    "Transfer units": quantity,
                    "Est. transfer cost (₹)": round(quantity * transfer_cost_per_unit, 2),
                    "Receiver lead time (days)": int(receiver["Lead time (days)"]),
                    "Reason": "Cover projected shortage using available network stock",
                })
                donor_remaining[donor["Store"]] -= quantity
                receiver_remaining[receiver["Store"]] -= quantity
    return pd.DataFrame(recommendations, columns=[
        "SKU", "Product", "From store", "To store", "Transfer units",
        "Est. transfer cost (₹)", "Receiver lead time (days)", "Reason"
    ])
