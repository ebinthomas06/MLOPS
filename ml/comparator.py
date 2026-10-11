"""
ml/comparator.py
AutoMLOps — Model Version Comparator & Evidently AI Drift Engine
Author: Ebin Thomas

Responsibilities:
1. Version Comparison:
   - Computes metric diffs between old (live) model and new candidate model.
   - Compares dataset distributions (features drift, target drift) using Evidently AI Report.
   - Generates and saves standalone HTML reports for rich visualization.
2. Production Drift Detection:
   - Compares live production traffic against training baseline.
   - Flags dataset drift with configurable thresholds for automated retraining triggers (used by Joel's monitor.py).
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("automlops-comparator")

# Output directory for Evidently HTML reports
REPORTS_DIR = Path(os.getenv("EVIDENTLY_REPORTS_DIR", "reports"))
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metric_diff(old_metrics: Dict[str, float], new_metrics: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    """
    Computes delta and percentage change between old and new metrics.
    """
    diff = {}
    all_keys = set(old_metrics.keys()).union(set(new_metrics.keys()))
    for key in sorted(all_keys):
        old_val = old_metrics.get(key)
        new_val = new_metrics.get(key)
        
        change = None
        pct_change = None
        improved = None
        
        if old_val is not None and new_val is not None:
            change = round(float(new_val) - float(old_val), 4)
            pct_change = round((change / old_val * 100), 2) if old_val != 0 else None
            # Standard metrics: higher is better (accuracy, f1, precision, recall, auc, r2)
            # Loss/error metrics: lower is better (mse, rmse, mae, log_loss)
            lower_is_better = any(term in key.lower() for term in ["loss", "mse", "rmse", "mae", "error"])
            improved = (change < 0) if lower_is_better else (change > 0)

        diff[key] = {
            "old": old_val,
            "new": new_val,
            "change": change,
            "pct_change": pct_change,
            "improved": improved
        }
    return diff


def compare_datasets(
    reference_data: Union[pd.DataFrame, str, Path],
    current_data: Union[pd.DataFrame, str, Path],
    report_name: str = "dataset_comparison",
    target_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compares reference dataset (old data) vs current dataset (new data) using Evidently AI.
    Saves an HTML report and returns a structured dictionary summary.
    """
    if isinstance(reference_data, (str, Path)):
        ref_df = pd.read_csv(reference_data)
    else:
        ref_df = reference_data.copy()

    if isinstance(current_data, (str, Path)):
        curr_df = pd.read_csv(current_data)
    else:
        curr_df = current_data.copy()

    # Drop non-overlapping columns if any, or keep shared columns
    common_cols = [c for c in ref_df.columns if c in curr_df.columns]
    ref_df = ref_df[common_cols]
    curr_df = curr_df[common_cols]

    report_html_path = REPORTS_DIR / f"{report_name}.html"
    drift_detected = False
    drift_share = 0.0
    drifted_features = []
    total_features = len([c for c in common_cols if c != target_col])

    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset

        # Generate Evidently AI report
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_df, current_data=curr_df)
        
        # Save HTML artifact
        report.save_html(str(report_html_path))
        logger.info(f"Evidently HTML report saved to {report_html_path}")

        # Extract structured metrics dictionary
        report_dict = report.as_dict()
        metrics_list = report_dict.get("metrics", [])
        
        for m in metrics_list:
            res = m.get("result", {})
            if "dataset_drift" in res:
                drift_detected = bool(res.get("dataset_drift", False))
                drift_share = float(res.get("drift_share", 0.0))
                # Identify drifted columns
                drift_by_columns = res.get("drift_by_columns", {})
                for col_name, col_stats in drift_by_columns.items():
                    if col_stats.get("drift_detected", False):
                        drifted_features.append({
                            "feature": col_name,
                            "stat_test": col_stats.get("stat_test_name"),
                            "p_value": round(col_stats.get("p_value", 0.0), 4)
                        })

    except Exception as e:
        logger.warning(f"Evidently report generation fallback (error: {e})")
        # Lightweight statistical fallback (KS-test / Chi-square) if evidently dependency has issues
        from scipy import stats
        for col in common_cols:
            if col == target_col:
                continue
            if pd.api.types.is_numeric_dtype(ref_df[col]) and pd.api.types.is_numeric_dtype(curr_df[col]):
                stat, p_val = stats.ks_2samp(ref_df[col].dropna(), curr_df[col].dropna())
                if p_val < 0.05:
                    drifted_features.append({"feature": col, "stat_test": "ks_2samp", "p_value": round(float(p_val), 4)})
        
        drift_share = len(drifted_features) / total_features if total_features > 0 else 0.0
        drift_detected = drift_share > 0.3
        
        # Write minimal HTML report
        with open(report_html_path, "w", encoding="utf-8") as f:
            f.write(f"<html><body><h2>Dataset Drift Fallback Report</h2><p>Drift detected: {drift_detected}</p></body></html>")

    return {
        "dataset_drift": drift_detected,
        "drift_share": round(drift_share, 4),
        "total_features": total_features,
        "drifted_features_count": len(drifted_features),
        "drifted_features": drifted_features,
        "report_html_path": str(report_html_path),
        "report_filename": report_html_path.name
    }


def compare_versions(
    old_metrics: Dict[str, float],
    new_metrics: Dict[str, float],
    old_dataset_path: Optional[str] = None,
    new_dataset_path: Optional[str] = None,
    version_label: str = "v1_vs_v2"
) -> Dict[str, Any]:
    """
    Main entry point for comparing two model versions (Aadi v1 vs Ebin/Candidate v2).
    """
    metric_diff = calculate_metric_diff(old_metrics, new_metrics)
    
    data_comparison = None
    if old_dataset_path and new_dataset_path and os.path.exists(old_dataset_path) and os.path.exists(new_dataset_path):
        data_comparison = compare_datasets(
            reference_data=old_dataset_path,
            current_data=new_dataset_path,
            report_name=f"compare_{version_label}"
        )

    # Decision summary: Did new model outperform the baseline?
    improvements = [v["improved"] for v in metric_diff.values() if v["improved"] is not None]
    overall_improved = sum(improvements) > len(improvements) / 2 if improvements else True

    return {
        "version_label": version_label,
        "metric_diff": metric_diff,
        "overall_improved": overall_improved,
        "data_drift_detected": data_comparison["dataset_drift"] if data_comparison else False,
        "data_comparison": data_comparison
    }


def check_production_drift(
    baseline_data: Union[pd.DataFrame, str, Path],
    live_traffic_data: Union[pd.DataFrame, str, Path],
    model_id: str,
    target_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    Monitors live inference traffic vs training baseline.
    Called by Joel's monitor.py to decide if a retraining dispatch should be triggered.
    """
    comparison = compare_datasets(
        reference_data=baseline_data,
        current_data=live_traffic_data,
        report_name=f"drift_monitor_{model_id}_{int(pd.Timestamp.now().timestamp())}",
        target_col=target_col
    )

    should_retrain = comparison["dataset_drift"] or comparison["drift_share"] > 0.4

    return {
        "model_id": model_id,
        "drift_detected": comparison["dataset_drift"],
        "drift_share": comparison["drift_share"],
        "drifted_features": comparison["drifted_features"],
        "should_retrain": should_retrain,
        "report_html_path": comparison["report_html_path"]
    }
