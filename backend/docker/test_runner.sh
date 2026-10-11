#!/bin/bash
# ============================================================
# test_runner.sh
# Quick smoke-test for Dockerfile.runner
# Run this after building the image to confirm it works.
# ============================================================

set -e

echo "=== Step 1: Building automlops-runner image ==="
docker build -t automlops-runner -f Dockerfile.runner .

echo ""
echo "=== Step 2: Checking Python and libraries inside container ==="
docker run --rm automlops-runner python -c "
import pandas; print('pandas:', pandas.__version__)
import numpy; print('numpy:', numpy.__version__)
import sklearn; print('scikit-learn:', sklearn.__version__)
import xgboost; print('xgboost:', xgboost.__version__)
import lightgbm; print('lightgbm:', lightgbm.__version__)
print('All libraries OK!')
"

echo ""
echo "=== Step 3: Simulating a training job (mounts a temp folder) ==="

# Create a tiny temp workspace
TMPDIR=$(mktemp -d)
cat > "$TMPDIR/train.py" << 'EOF'
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pickle

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
accuracy = round(accuracy_score(y_test, preds), 4)
f1 = round(f1_score(y_test, preds, average='weighted'), 4)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print(f"METRICS:accuracy={accuracy},f1={f1}")
EOF

docker run --rm \
  -v "$TMPDIR:/workspace" \
  -w /workspace \
  automlops-runner \
  python train.py

echo ""
if [ -f "$TMPDIR/model.pkl" ]; then
    echo "model.pkl created successfully in workspace!"
else
    echo "WARNING: model.pkl was NOT found — something went wrong."
fi

rm -rf "$TMPDIR"

echo ""
echo "=== All tests passed! automlops-runner is ready. ==="
echo "Share this with Joel: he needs the image name 'automlops-runner' in runner.py"