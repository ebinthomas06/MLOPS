"""
smoke_test_ebin.py
Verifies Ebin's work items:
1. Version comparison & metric diff calculation
2. Dataset comparison and Evidently drift check with HTML report output
3. Production drift detection for Joel's monitor
4. Deployer free-port finding and mock registration
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure root is in python path
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

from ml.comparator import (
    calculate_metric_diff,
    compare_datasets,
    compare_versions,
    check_production_drift
)
from backend.services.deployer import find_free_port, update_nginx_config

def run_tests():
    print("====================================================")
    print("Running Ebin's Work Items Smoke Test...")
    print("====================================================")

    # Test 1: Metric diff
    print("\n[1] Testing calculate_metric_diff:")
    v1_metrics = {"accuracy": 0.88, "f1": 0.86, "loss": 0.32}
    v2_metrics = {"accuracy": 0.94, "f1": 0.93, "loss": 0.18}
    diff = calculate_metric_diff(v1_metrics, v2_metrics)
    print("Metric diff calculated successfully:")
    for k, v in diff.items():
        print(f"  - {k}: {v['old']} -> {v['new']} (diff {v['change']}, improved={v['improved']})")
    assert diff["accuracy"]["improved"] is True
    assert diff["loss"]["improved"] is True

    # Test 2: Dataset comparison & HTML generation
    print("\n[2] Testing compare_datasets & HTML artifact generation:")
    np.random.seed(42)
    ref_df = pd.DataFrame({
        "feature_1": np.random.normal(0, 1, 100),
        "feature_2": np.random.normal(5, 2, 100),
        "target": np.random.choice([0, 1], 100)
    })
    # Simulated drifted current data
    curr_df = pd.DataFrame({
        "feature_1": np.random.normal(2.5, 1, 100), # significant drift
        "feature_2": np.random.normal(5, 2, 100),
        "target": np.random.choice([0, 1], 100)
    })
    
    comp_result = compare_datasets(ref_df, curr_df, report_name="test_smoke_drift")
    print(f"Dataset drift detected: {comp_result['dataset_drift']}")
    print(f"Drift share: {comp_result['drift_share']}")
    print(f"Drifted features: {comp_result['drifted_features']}")
    print(f"Report generated at: {comp_result['report_html_path']}")
    assert os.path.exists(comp_result['report_html_path'])

    # Test 3: Version comparison wrapper
    print("\n[3] Testing compare_versions:")
    ver_comp = compare_versions(v1_metrics, v2_metrics, version_label="v1_vs_v2_smoke")
    print(f"Overall candidate model improved: {ver_comp['overall_improved']}")
    assert ver_comp["overall_improved"] is True

    # Test 4: Production drift check for Joel's monitor
    print("\n[4] Testing check_production_drift (called by Joel's monitor.py):")
    drift_eval = check_production_drift(ref_df, curr_df, model_id="test_model_123")
    print(f"Should retrain trigger fire: {drift_eval['should_retrain']}")

    # Test 5: Deployer port discovery
    print("\n[5] Testing deployer port allocator:")
    free_port = find_free_port()
    print(f"Allocated free port: {free_port}")
    assert free_port >= 8081

    # Test 6: Nginx config regeneration
    print("\n[6] Testing Nginx routing generation:")
    mock_registry = {
        "iris_v1": {
            "port": 8081,
            "status": "running"
        }
    }
    nginx_ok = update_nginx_config(mock_registry)
    print(f"Nginx config updated successfully: {nginx_ok}")
    assert nginx_ok is True

    print("\n====================================================")
    print("ALL SMOKE TESTS PASSED SUCCESSFULLY!")
    print("====================================================")

if __name__ == "__main__":
    run_tests()
