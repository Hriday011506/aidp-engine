import pandas as pd
import numpy as np
import os


def preprocess_data():

    print("🚀 Starting AIDP preprocessing (Daily Level)...")

    train_path = "data/raw/train.csv"
    holiday_path = "data/raw/holidays_events.csv"

    train = pd.read_csv(train_path)
    holidays = pd.read_csv(holiday_path)

    # Convert date
    train["date"] = pd.to_datetime(train["date"])
    holidays["date"] = pd.to_datetime(holidays["date"])

    # Sample data (to avoid huge memory load)
    train = train.sample(n=50000, random_state=42)

    # Aggregate sales per day
    daily_sales = train.groupby("date")["sales"].sum().reset_index()

    # Holiday flag per day
    holidays["is_holiday"] = 1
    holiday_flag = holidays[["date", "is_holiday"]]

    daily = daily_sales.merge(holiday_flag, on="date", how="left")
    daily["is_holiday"] = daily["is_holiday"].fillna(0)

    # Add external features
    np.random.seed(42)
    daily["avg_temp"] = np.random.randint(20, 40, len(daily))
    daily["viral_score"] = np.random.randint(10, 100, len(daily))
    daily["local_events"] = np.random.randint(0, 5, len(daily))

    # Rename
    daily.rename(columns={"sales": "daily_sales"}, inplace=True)

    # Save
    os.makedirs("data/processed", exist_ok=True)
    daily.to_csv("data/processed/daily_data.csv", index=False)

    print("✅ Daily dataset created!")
    print("📊 Total rows:", len(daily))


if __name__ == "__main__":
    preprocess_data()
