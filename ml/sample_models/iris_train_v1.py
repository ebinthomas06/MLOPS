"""
ml/sample_models/iris_train_v1.py
Baseline v1 training script for Iris dataset
Logs parameters, metrics, and model artifact to MLflow and exports model.pkl
"""

import os
import pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow

# Point to self-hosted MLflow tracking server (default: localhost:5000)
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
try:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("iris-classification")
except Exception as e:
    print(f"MLflow connection note: {e}")

def train():
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    # Save baseline dataset for Evidently AI comparison
    os.makedirs("data", exist_ok=True)
    baseline_df = X.copy()
    baseline_df["target"] = y
    baseline_df.to_csv("data/iris_baseline.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    params = {
        "n_estimators": 20,
        "max_depth": 3,
        "random_state": 42
    }

    try:
        with mlflow.start_run(run_name="iris_v1_baseline"):
            mlflow.log_params(params)

            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            accuracy = round(float(accuracy_score(y_test, preds)), 4)
            f1 = round(float(f1_score(y_test, preds, average="weighted")), 4)
            precision = round(float(precision_score(y_test, preds, average="weighted")), 4)
            recall = round(float(recall_score(y_test, preds, average="weighted")), 4)

            metrics = {
                "accuracy": accuracy,
                "f1": f1,
                "precision": precision,
                "recall": recall
            }
            mlflow.log_metrics(metrics)

            with open("model.pkl", "wb") as f:
                pickle.dump(model, f)

            mlflow.log_artifact("model.pkl")
            print(f"METRICS:accuracy={accuracy},f1={f1},precision={precision},recall={recall}")
            print("Model saved to model.pkl and logged to MLflow successfully.")
            return model, metrics
    except Exception as e:
        # Fallback if MLflow is not yet active
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = round(float(accuracy_score(y_test, preds)), 4)
        f1 = round(float(f1_score(y_test, preds, average="weighted")), 4)
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
        print(f"METRICS:accuracy={accuracy},f1={f1}")
        return model, {"accuracy": accuracy, "f1": f1}

if __name__ == "__main__":
    train()
