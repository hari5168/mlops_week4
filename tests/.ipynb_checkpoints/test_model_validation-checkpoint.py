import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

def test_model_accuracy():
    model = joblib.load("models/model.joblib")
    df = pd.read_csv("data/data.csv")

    X = df.drop("species", axis=1)
    y = df["species"]

    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    assert acc > 0.9, f"Model accuracy too low: {acc}"
