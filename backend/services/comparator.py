"""
backend/services/comparator.py
Adapter proxy exposing ml/comparator.py functions to backend routes and Joel's monitor.
"""

import sys
from pathlib import Path

# Add project root to sys.path so ml.comparator can be imported reliably
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ml.comparator import (
    calculate_metric_diff,
    compare_datasets,
    compare_versions,
    check_production_drift
)

__all__ = [
    "calculate_metric_diff",
    "compare_datasets",
    "compare_versions",
    "check_production_drift"
]
