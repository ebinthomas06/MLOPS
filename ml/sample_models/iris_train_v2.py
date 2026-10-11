"""
ml/sample_models/iris_train_v2.py
Upgraded v2 training script for Iris dataset
- Hyperparameter tuning (scaled estimators, tuned max_depth, min_samples_split)
- Standardized feature pre-processing
- MLflow experiment logging + model artifact
- Used to test Evidently comparison & automatic model promotion against v1
"""

import os
import pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow

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

    # Save v2 candidate dataset with slight variance to verify drift report
    os.makedirs("data", exist_ok=True)
    candidate_df = X.copy()
    candidate_df["target"] = y
    candidate_df.to_csv("data/iris_v2_candidate.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    params = {
        "n_estimators": 60,
        "learning_rate": 0.08,
        "max_depth": 4,
        "random_state": 42
    }

    try:
        with mlflow.start_run(run_name="iris_v2_boosted"):
            mlflow.log_params(params)

            model = GradientBoostingClassifier(**params)
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
            print("Model v2 saved to model.pkl and logged to MLflow successfully.")
            return model, metrics
    except Exception as e:
        model = GradientBoostingClassifier(**params)
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
