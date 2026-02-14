import pickle
import numpy as np

def predict_demand(avg_temp, local_events, viral_score):

    with open("models/demand_model.pkl", "rb") as f:
        model = pickle.load(f)

    prediction = model.predict(
        np.array([[avg_temp, local_events, viral_score]])
    )[0]

    return prediction
