import pandas as pd
import numpy as np
import os


def preprocess_data():

    print("🚀 Starting AIDP preprocessing pipeline...")

    # -------------------------------
    # Load raw datasets
    # -------------------------------
    train_path = "data/raw/train.csv"
    holiday_path = "data/raw/holidays_events.csv"

    print("📂 Loading datasets...")
    train = pd.read_csv(train_path)
    holidays = pd.read_csv(holiday_path)

    print("✅ Datasets loaded successfully")

    # -------------------------------
    # Convert date columns
    # -------------------------------
    train["date"] = pd.to_datetime(train["date"])
    holidays["date"] = pd.to_datetime(holidays["date"])

    # -------------------------------
    # Create month column
    # -------------------------------
    train["month"] = train["date"].dt.to_period("M")
    holidays["month"] = holidays["date"].dt.to_period("M")

    print("📅 Aggregating daily sales to monthly level...")

    # -------------------------------
    # Aggregate monthly sales
    # -------------------------------
    monthly_sales = (
        train
        .groupby("month")["sales"]
        .sum()
        .reset_index()
    )

    monthly_sales.rename(columns={"sales": "monthly_sales"}, inplace=True)

    # -------------------------------
    # Count holidays per month
    # -------------------------------
    holiday_count = (
        holidays
        .groupby("month")
        .size()
        .reset_index(name="holiday_count")
    )

    # Merge holiday info
    monthly = monthly_sales.merge(
        holiday_count,
        on="month",
        how="left"
    )

    monthly["holiday_count"] = monthly["holiday_count"].fillna(0)

    # -------------------------------
    # Add External Features (Simulated for v1)
    # -------------------------------
    print("🌦 Adding external signals...")

    np.random.seed(42)

    monthly["avg_temp"] = np.random.randint(20, 40, len(monthly))
    monthly["viral_score"] = np.random.randint(10, 100, len(monthly))

    # -------------------------------
    # Final formatting
    # -------------------------------
    monthly["month"] = monthly["month"].astype(str)

    # Ensure processed folder exists
    os.makedirs("data/processed", exist_ok=True)

    output_path = "data/processed/monthly_data.csv"
    monthly.to_csv(output_path, index=False)

    print("✅ Preprocessing complete!")
    print(f"📁 Processed dataset saved at: {output_path}")
    print("📊 Total months processed:", len(monthly))


if __name__ == "__main__":
    preprocess_data()
