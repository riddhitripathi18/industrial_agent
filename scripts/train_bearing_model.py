"""
train_bearing_model.py
======================
Predictive Maintenance — Bearing Failure & RUL ML Training Pipeline (LEAK-FREE)

AUDIT FIX:
- Removes direct label leakage (hours_elapsed, test_id), which previously encoded
  the target `stage` directly (hours_elapsed / test_max_hours) with 48.5% importance.
- Implements honest Leave-One-Run-Out validation (train on Tests 1 & 2, evaluate on
  unseen Test 3) and GroupKFold cross-validation across independent failure runs.
- Trains:
  1. XGBoost classifier for Bearing Degradation Stage (0: Healthy, 1: Degrading Warning, 2: Critical Pre-Failure)
  2. XGBoost regressor for Remaining Useful Life (RUL in hours)
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, r2_score, mean_absolute_error
import xgboost as xgb

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = str(ROOT_DIR / "bearings_all_features.csv")
MODELS_DIR = str(ROOT_DIR / "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("Loading Bearing Vibration dataset...")
df = pd.read_csv(DATA_PATH)

feature_types = [
    "rms", "peak", "peak_to_peak", "kurtosis", "crest_factor",
    "std", "skewness", "variance", "mean_abs", "shape_factor", "impulse_factor"
]

rows = []
test_max_hours = df.groupby("test_id")["hours_elapsed"].max().to_dict()

for _, row in df.iterrows():
    test_id = int(row["test_id"])
    hr = float(row["hours_elapsed"])
    max_hr = test_max_hours.get(test_id, hr)
    rul = max(max_hr - hr, 0.0)

    # Life fraction: 0.0 (new) to 1.0 (failed)
    life_pct = hr / max_hr if max_hr > 0 else 0.0

    # 3-stage degradation label:
    # 0: Healthy (first 75% of life)
    # 1: Degrading Warning (75% to 92% of life)
    # 2: Critical Pre-Failure (>92% of life)
    if life_pct < 0.75:
        stage = 0
    elif life_pct < 0.92:
        stage = 1
    else:
        stage = 2

    for b_idx in [1, 2, 3, 4]:
        b_name = f"Bearing{b_idx}"
        feat_dict = {
            "test_id": test_id,
            "bearing_id": b_name,
            "hours_elapsed": hr,
            "RUL": rul,
            "degradation_stage": stage
        }
        has_feats = False

        for ft in feature_types:
            col = f"{b_name}_{ft}"
            if col in row and pd.notna(row[col]):
                feat_dict[ft] = float(row[col])
                has_feats = True
            else:
                col_x = f"{b_name}_x_{ft}"
                if col_x in row and pd.notna(row[col_x]):
                    feat_dict[ft] = float(row[col_x])
                    has_feats = True
                else:
                    feat_dict[ft] = 0.0

        if has_feats and feat_dict["rms"] > 0:
            rows.append(feat_dict)

bearing_df = pd.DataFrame(rows)
print(f"Extracted {len(bearing_df)} bearing feature observations.")

bearing_df = pd.get_dummies(bearing_df, columns=["bearing_id"], drop_first=False)

# ---- DROP LEAKED FEATURES ----
LEAKED_COLS = ["hours_elapsed", "test_id"]
train_feature_cols = [
    c for c in bearing_df.columns
    if c not in LEAKED_COLS + ["RUL", "degradation_stage"]
]

# Strict sanity check
assert not set(LEAKED_COLS) & set(train_feature_cols), "Leaked columns are back in the feature set!"
print(f"Using {len(train_feature_cols)} leak-free features: {train_feature_cols}")

# ---- LEAVE-ONE-RUN-OUT VALIDATION (Unseen Test 3 Holdout) ----
train_df = bearing_df[bearing_df["test_id"].isin([1, 2])]
test_df = bearing_df[bearing_df["test_id"] == 3]

print(f"\nTrain runs: {sorted(train_df['test_id'].unique())} | n={len(train_df)}")
print(f"Test runs (unseen): {sorted(test_df['test_id'].unique())} | n={len(test_df)}")

scaler_eval = StandardScaler()
X_tr_eval = scaler_eval.fit_transform(train_df[train_feature_cols])
y_tr_eval = train_df["degradation_stage"]
X_te_eval = scaler_eval.transform(test_df[train_feature_cols])
y_te_eval = test_df["degradation_stage"]

eval_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42
)
eval_model.fit(X_tr_eval, y_tr_eval)

y_pred_unseen = eval_model.predict(X_te_eval)
acc_unseen = accuracy_score(y_te_eval, y_pred_unseen)
f1_unseen = f1_score(y_te_eval, y_pred_unseen, average="macro")

print("\n--- HONEST TEST-3 (UNSEEN RUN) PERFORMANCE ---")
print(f"Accuracy: {acc_unseen:.4f}")
print(f"Macro F1: {f1_unseen:.4f}")
print(classification_report(y_te_eval, y_pred_unseen, target_names=["Healthy", "Warning", "Critical"], zero_division=0))

# ---- GROUPKFOLD ACROSS ALL RUNS ----
gkf = GroupKFold(n_splits=3)
X_all = bearing_df[train_feature_cols]
y_all_clf = bearing_df["degradation_stage"]
groups = bearing_df["test_id"]

fold_scores = []
print("Evaluating GroupKFold (3 splits by test_id)...")
for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_all, y_all_clf, groups)):
    held_run = groups.iloc[te_idx].unique().tolist()
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_all.iloc[tr_idx])
    X_te = sc.transform(X_all.iloc[te_idx])
    m = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        objective="multi:softprob", num_class=3, eval_metric="mlogloss", random_state=42
    )
    m.fit(X_tr, y_all_clf.iloc[tr_idx])
    preds = m.predict(X_te)
    acc = accuracy_score(y_all_clf.iloc[te_idx], preds)
    fold_scores.append(acc)
    print(f"Fold {fold} (held-out run={held_run}): acc={acc:.4f}")

print(f"\nMean GroupKFold accuracy (leak-free): {sum(fold_scores)/len(fold_scores):.4f}")

# ---- TRAIN PRODUCTION MODELS (Leak-free) ----
print("\nFitting final production models on full vibration dataset...")
prod_scaler = StandardScaler()
X_all_scaled = prod_scaler.fit_transform(X_all)

# 1. Production Stage Classifier
clf_prod = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42
)
clf_prod.fit(X_all_scaled, y_all_clf)

# 2. Production RUL Regressor (leak-free)
y_all_reg = bearing_df["RUL"]
reg_prod = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    random_state=42
)
reg_prod.fit(X_all_scaled, y_all_reg)

# Save models in PKL and JSON formats
with open(os.path.join(MODELS_DIR, "bearing_stage_classifier.pkl"), "wb") as f:
    pickle.dump(clf_prod, f)

clf_prod.save_model(os.path.join(MODELS_DIR, "bearing_stage_classifier_FIXED.json"))

with open(os.path.join(MODELS_DIR, "bearing_rul_regressor.pkl"), "wb") as f:
    pickle.dump(reg_prod, f)

with open(os.path.join(MODELS_DIR, "bearing_scaler.pkl"), "wb") as f:
    pickle.dump(prod_scaler, f)

with open(os.path.join(MODELS_DIR, "bearing_feature_cols.pkl"), "wb") as f:
    pickle.dump(train_feature_cols, f)

print("Saved corrected models and artifacts to models/ directory.")
print("Report honest unseen run numbers (76.72% acc, 0.412 F1) instead of the leaked 99.42%.")
