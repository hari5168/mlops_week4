import pandas as pd
import joblib, os

# Load model
clf = joblib.load("models/model.joblib")

# Load eval data
df = pd.read_csv("data/data.csv")
X = df.drop("species", axis=1)
y = df["species"]

# Inference
y_pred = clf.predict(X)
print("Sample Predictions:", y_pred[:10])
print("Accuracy:", (y_pred == y).mean())