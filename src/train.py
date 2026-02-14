import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = os.path.join("data", "processed", "monthly_data.csv")
MODEL_PATH = os.path.join("models", "demand_model.pkl")

def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        print("Dataset not found. Creating synthetic dataset...")
        np.random.seed(42)
        months = 60

        df = pd.DataFrame({
            "holiday_count": np.random.randint(0, 5, months),
            "avg_temp": np.random.randint(20, 40, months),
            "viral_score": np.random.randint(10, 100, months),
        })

        df["monthly_sales"] = (
            50000 +
            df["holiday_count"] * 2000 +
            df["avg_temp"] * 300 +
            df["viral_score"] * 150 +
            np.random.randint(-5000, 5000, months)
        )

    # Feature Engineering
    df["month"] = np.tile(np.arange(1, 13), len(df)//12 + 1)[:len(df)]
    df["prev_month_sales"] = df["monthly_sales"].shift(1)
    df = df.dropna()

    return df

def train():

    df = load_data()

    X = df[[
        "holiday_count",
        "avg_temp",
        "viral_score",
        "month",
        "prev_month_sales"
    ]]

    y = df["monthly_sales"]

    tscv = TimeSeriesSplit(n_splits=5)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        random_state=42
    )

    model.fit(X, y)

    # Evaluate on last split
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"MAE: {mae}")
    print(f"R2 Score: {r2}")

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("Model saved successfully!")

if __name__ == "__main__":
    train()
