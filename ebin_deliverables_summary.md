# AutoMLOps — Core Architecture & Ebin's Implementation Summary

This document details the implementation of all deliverables assigned to **Ebin** across the agreed team split and the project architecture guide.

---

## 1. Summary of Deliverables Completed

| Area | Component | Location | Description |
| :--- | :--- | :--- | :--- |
| **Evidently AI & Comparison** | `comparator.py` | [`ml/comparator.py`](file:///c:/Users/ebint/OneDrive/Documents/GitHub/MLOPS/ml/comparator.py) & [`backend/services/comparator.py`](file:///c:/Users/ebint/OneDrive/Documents/GitHub/MLOPS/backend/services/comparator.py) | Full version comparison (old vs new metric diffs), dataset distribution comparison, and live production drift detection. Outputs HTML report artifacts and structured JSON metrics for Joel's `monitor.py` and Sanjan's `diagnosis.py`. |
| **MLflow & Model Versions** | `iris_train_v1.py` & `iris_train_v2.py` | [`ml/sample_models/iris_train_v1.py`](file:///c:/Users/ebint/OneDrive/Documents/GitHub/MLOPS/ml/sample_models/iris_train_v1.py) <br> [`ml/sample_models/iris_train_v2.py`](file:///c:/Users/ebint/OneDrive/Documents/GitHub/MLOPS/ml/sample_models/iris_train_v2.py) | Baseline v1 and boosted/tuned v2 training scripts with MLflow tracking (parameters, metrics, and `model.pkl` artifact logging). |
| **Model Serving Container** | `Dockerfile.serving` & `serve.py` | [`backend/docker/Dockerfile.serving`](file:///c:/Users/ebint/OneDrive/Documents/GitHub/MLOPS/backend/docker/Dockerfile.serving) <br> [`backend/docker/serve.py`](file:///c:/Users/ebint/OneDrive/Documents/GitHub/MLOPS/backend/docker/serve.py) | Standalone microservice container running FastAPI/Uvicorn. Exposes `GET /health` and `POST /predict` (accepting flexible record lists or matrices). |
| **Dynamic Deployment Engine** | `deployer.py` | [`backend/services/deployer.py`](file:///c:/Users/ebint/OneDrive/Documents/GitHub/MLOPS/backend/services/deployer.py) | Dynamic port allocator (`8081+`), spins up serving containers, maintains registry, healthchecks, and rewrites Nginx reverse-proxy configuration. |
| **Gateway & Stable Routing** | `nginx.conf` | [`nginx/nginx.conf`](file:///c:/Users/ebint/OneDrive/Documents/GitHub/MLOPS/nginx/nginx.conf) | Nginx reverse-proxy keeping persistent URLs like `/models/{model_id}/predict` even when underlying containers are updated or swapped. |
| **Infrastructure Orchestration** | `docker-compose.yml` | [`docker-compose.yml`](file:///c:/Users/ebint/OneDrive/Documents/GitHub/MLOPS/docker-compose.yml) | Composes backend, self-hosted MLflow tracking server (Port 5000), Nginx gateway (Port 80), and Cloudflare Tunnel container. |
| **Cloudflare Tunnel** | Tunnel script & configuration | [`start_tunnel.ps1`](file:///c:/Users/ebint/OneDrive/Documents/GitHub/MLOPS/start_tunnel.ps1) & [`.env.example`](file:///c:/Users/ebint/OneDrive/Documents/GitHub/MLOPS/.env.example) | Quick one-command Cloudflare Tunnel setup (`trycloudflare` or authenticated token) exposing local port 80 to the web. |

---

## 2. API Contract & Integration Specifications

### Ebin to Joel (`monitor.py` & alerts)
Joel can directly import from `ml.comparator`:
```python
from ml.comparator import check_production_drift

# Call drift check with baseline training data and live traffic window
drift_result = check_production_drift(
    baseline_data="data/iris_baseline.csv",
    live_traffic_data=current_traffic_df,
    model_id="iris_v1"
)

if drift_result["should_retrain"]:
    # Fire Discord webhook alert and trigger GitHub repository_dispatch
    pass
```

### Ebin to Sanjan (`diagnosis.py`)
Sanjan can consume the structured output of `compare_versions`:
```python
from ml.comparator import compare_versions

report = compare_versions(old_metrics, new_metrics, old_data_path, new_data_path)
# report contains:
# - metric_diff (deltas, % changes, improvement flags)
# - data_drift_detected (boolean)
# - data_comparison (drifted features, p-values, HTML path)
```

### Ebin to Kevin / Joel (Backend to Serving / Frontend)
When a passing pipeline finishes, `deployer.py` returns:
```json
{
  "model_id": "iris_v1",
  "port": 8081,
  "status": "running",
  "gateway_url": "http://localhost/models/iris_v1/predict",
  "health_url": "http://localhost/models/iris_v1/health"
}
```
Client prediction payload:
```http
POST http://localhost/models/iris_v1/predict
Content-Type: application/json

{
  "inputs": [
    {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2}
  ]
}
```
