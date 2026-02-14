import pandas as pd
import pickle
import numpy as np


# -------------------------------
# DATA FUNCTIONS
# -------------------------------

def load_raw_data(path):
    """Load raw retail dataset"""
    return pd.read_csv(path)


def save_processed_data(df, path):
    """Save cleaned dataset"""
    df.to_csv(path, index=False)


# -------------------------------
# FEATURE ENGINEERING
# -------------------------------

def add_external_features(df):
    """
    Simulate external signals:
    - Weather
    - Events
    - Social virality
    """

    np.random.seed(42)

    df["avg_temp"] = np.random.randint(20, 40, len(df))
    df["local_events"] = np.random.randint(0, 8, len(df))
    df["viral_score"] = np.random.randint(10, 100, len(df))

    return df


# -------------------------------
# MODEL SAVE / LOAD
# -------------------------------

def save_model(model, path):
    """Save trained model"""
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    """Load trained model"""
    with open(path, "rb") as f:
        return pickle.load(f)


# -------------------------------
# BUSINESS LOGIC
# -------------------------------

def calculate_reorder_quantity(predicted_demand, buffer_percent=0.10):
    """Add safety buffer to demand"""
    return predicted_demand * (1 + buffer_percent)


def calculate_optimized_price(predicted_demand, base_price=100, elasticity=0.01):
    """Simple dynamic pricing logic"""
    return base_price * (1 + elasticity * (predicted_demand / 1000))
