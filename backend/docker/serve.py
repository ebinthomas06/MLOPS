"""
AutoMLOps Generic Prediction Serving Server
Wraps trained model (model.pkl / model.joblib) into a FastAPI inference API.
"""

import os
import time
import pickle
import joblib
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("automlops-serving")

app = FastAPI(
    title="AutoMLOps Model Serving API",
    description="Generic model inference endpoint managed by AutoMLOps",
    version="1.0.0"
)

MODEL_PATH = os.getenv("MODEL_PATH", "/app/model.pkl")
MODEL = None
LOAD_TIME = None


def load_model():
    global MODEL, LOAD_TIME
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Model file not found at {MODEL_PATH}. Waiting for mount...")
        return False
    try:
        try:
            with open(MODEL_PATH, "rb") as f:
                MODEL = pickle.load(f)
        except Exception:
            MODEL = joblib.load(MODEL_PATH)
        LOAD_TIME = time.time()
        logger.info(f"Model successfully loaded from {MODEL_PATH}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model from {MODEL_PATH}: {e}")
        return False


@app.on_event("startup")
def startup_event():
    load_model()


class PredictRequest(BaseModel):
    # Supports list of records [{"feature1": val, ...}], list of feature vectors [[1, 2, ...]], or dict
    inputs: Union[List[Dict[str, Any]], List[List[Any]], Dict[str, Any]]


@app.get("/health")
def healthcheck():
    global MODEL
    if MODEL is None:
        # Attempt reload if mounted after start
        load_model()
    return {
        "status": "healthy" if MODEL is not None else "model_not_loaded",
        "model_path": MODEL_PATH,
        "loaded": MODEL is not None,
        "model_type": type(MODEL).__name__ if MODEL is not None else None,
        "timestamp": time.time()
    }


@app.post("/predict")
async def predict(payload: PredictRequest):
    global MODEL
    if MODEL is None:
        if not load_model():
            raise HTTPException(status_code=503, detail="Model is not loaded or unavailable")

    start_time = time.time()
    try:
        raw_inputs = payload.inputs
        
        # Format input data
        if isinstance(raw_inputs, dict):
            # Single row dict or column-oriented dict
            df = pd.DataFrame([raw_inputs])
        elif isinstance(raw_inputs, list):
            if len(raw_inputs) == 0:
                raise HTTPException(status_code=400, detail="Empty inputs provided")
            if isinstance(raw_inputs[0], dict):
                df = pd.DataFrame(raw_inputs)
            else:
                df = np.array(raw_inputs)
        else:
            df = np.array(raw_inputs)

        # Run inference
        if hasattr(MODEL, "predict"):
            preds = MODEL.predict(df)
        elif callable(MODEL):
            preds = MODEL(df)
        else:
            raise HTTPException(status_code=500, detail="Loaded model object is neither callable nor has predict method")

        # Convert numpy types to native Python types for JSON serialization
        if hasattr(preds, "tolist"):
            results = preds.tolist()
        else:
            results = list(preds)

        # Optional probabilities if supported
        probabilities = None
        if hasattr(MODEL, "predict_proba"):
            try:
                proba = MODEL.predict_proba(df)
                probabilities = proba.tolist() if hasattr(proba, "tolist") else list(proba)
            except Exception:
                probabilities = None

        latency_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "predictions": results,
            "probabilities": probabilities,
            "latency_ms": latency_ms,
            "count": len(results)
        }

    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
