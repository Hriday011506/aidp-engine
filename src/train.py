import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

def train_model():
    df = pd.read_csv("data/processed/monthly_data.csv")

    X = df[['avg_temp', 'local_events', 'viral_score']]
    y = df['monthly_sales']

    model = RandomForestRegressor(n_estimators=200)
    model.fit(X, y)

    with open("models/demand_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model trained and saved.")
