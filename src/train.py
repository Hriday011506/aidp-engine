import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def train_model():

    print("🚀 Starting AIDP Model Training...")

    # -------------------------------
    # Load processed data
    # -------------------------------
    data_path = "data/processed/daily_data.csv"

    df = pd.read_csv(data_path)

    print("✅ Data loaded successfully")
    print("📊 Total rows:", len(df))

    # -------------------------------
    # Feature Selection
    # -------------------------------
    features = ["is_holiday", "avg_temp", "viral_score", "local_events"]
    target = "daily_sales"

    X = df[features]
    y = df[target]

    # -------------------------------
    # Train/Test Split
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("📈 Training Random Forest model...")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    # -------------------------------
    # Evaluation
    # -------------------------------
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("📊 Model Performance:")
    print("MAE:", round(mae, 2))
    print("R2 Score:", round(r2, 3))

    # -------------------------------
    # Save Model
    # -------------------------------
    os.makedirs("models", exist_ok=True)

    model_path = "models/demand_model.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("✅ Model saved successfully!")
    print("📁 Saved at:", model_path)


if __name__ == "__main__":
    train_model()
