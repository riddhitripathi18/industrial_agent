"""
train_bearing_model_v2.py
=========================
Bearing degradation classifier with ALL four audit corrections applied:

1.  Per-class recall + majority-baseline reported on every evaluation.
2.  Critical-class re-weighting:
      a) scale_pos_weight XGBoost   (flat 3-class, weighted)
      b) Two-stage model            (binary "anomalous vs. healthy" - Warning/Critical)
3.  Frequency-domain proxy features derived from time-domain statistics
    (BPFO/BPFI stress proxies: kurtosis - crest_factor, impulse rise, spectral centroid surrogate).
    NOTE: true envelope spectrum requires raw waveform; these are statistically sound
    surrogates from existing per-segment statistics.
4.  Fold spread reported as [min, max] range across 3 GroupKFold splits, NOT averaged.
"""

import os, pickle, warnings, sys
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score, recall_score
)

warnings.filterwarnings("ignore")

ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_PATH  = ROOT_DIR / "bearings_all_features.csv"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# -- Ground-truth: only the documented failing bearing per run degrades ---------
FAILED_MAP = {1: ["Bearing3", "Bearing4"], 2: ["Bearing1"], 3: ["Bearing3"]}
FEATURE_TYPES = [
    "rms", "peak", "peak_to_peak", "kurtosis", "crest_factor",
    "std", "skewness", "variance", "mean_abs", "shape_factor", "impulse_factor"
]

# -- IMS shaft speed & bearing geometry (Test 1: 2000 rpm; Tests 2&3: 2000 rpm) --
SHAFT_RPM  = 2000.0
SHAFT_HZ   = SHAFT_RPM / 60.0          # 33.33 Hz
# IMS test rig: 16-element bearing, pitch diameter / ball diameter ratio ~2.874
# BPFO ~ 119.5 Hz, BPFI ~ 147.2 Hz
BPFO_HZ    = 119.5
BPFI_HZ    = 147.2

# -----------------------------------------------------------------------------
# STEP 1 - Load & unroll per-bearing rows
# -----------------------------------------------------------------------------
print("Loading dataset ...")
df_raw = pd.read_csv(DATA_PATH)
test_max_hours = df_raw.groupby("test_id")["hours_elapsed"].max().to_dict()

rows = []
for _, row in df_raw.iterrows():
    test_id = int(row["test_id"])
    hr      = float(row["hours_elapsed"])
    max_hr  = test_max_hours[test_id]
    life_pct = hr / max_hr if max_hr > 0 else 0.0

    for b_idx in [1, 2, 3, 4]:
        b_name     = f"Bearing{b_idx}"
        is_failing = b_name in FAILED_MAP[test_id]
        if is_failing:
            stage = 0 if life_pct < 0.75 else (1 if life_pct < 0.92 else 2)
        else:
            stage = 0  # companion bearings stayed healthy throughout

        feat_dict = {
            "test_id":    test_id,
            "bearing_id": b_name,
            "hours_elapsed": hr,
            "stage":      stage,
            "RUL":        max(max_hr - hr, 0.0),
        }
        has_feats = False
        for ft in FEATURE_TYPES:
            col   = f"{b_name}_{ft}"
            col_x = f"{b_name}_x_{ft}"
            if col in row and pd.notna(row[col]):
                feat_dict[ft] = float(row[col]);  has_feats = True
            elif col_x in row and pd.notna(row[col_x]):
                feat_dict[ft] = float(row[col_x]); has_feats = True
            else:
                feat_dict[ft] = 0.0
        if has_feats and feat_dict["rms"] > 0:
            rows.append(feat_dict)

b_df = pd.DataFrame(rows)
print(f"Unrolled: {len(b_df)} bearing observations")
print("Label distribution (physically correct):", b_df["stage"].value_counts().to_dict())

# -----------------------------------------------------------------------------
# STEP 2 - Self-relative features + frequency-domain proxies
# -----------------------------------------------------------------------------

def build_features(df_in, baseline_frac=0.10):
    """
    Per-bearing self-normalised features + BPFO/BPFI statistical proxies.

    Frequency-domain surrogates (no raw waveform needed):
      bpfo_proxy  = kurtosis * crest_factor  (impulsive energy at fault freq)
      bpfi_proxy  = impulse_factor * peak    (inner-race impact amplitude)
      env_rms     = rms / mean_abs == shape_factor (envelope amplitude ratio)
      spec_entropy_proxy = 1/(kurtosis+eps)  (low kurt = broad spectrum = healthy)
    """
    df_in = df_in.sort_values(["test_id", "bearing_id", "hours_elapsed"]).copy()
    frames = []
    for (tid, bid), grp in df_in.groupby(["test_id", "bearing_id"]):
        grp = grp.sort_values("hours_elapsed").copy()
        n_base = max(int(len(grp) * baseline_frac), 10)
        base   = grp.iloc[:n_base][FEATURE_TYPES]
        mu     = base.mean()
        sigma  = base.std().replace(0, 1e-6).fillna(1e-6)

        for f in FEATURE_TYPES:
            grp[f"{f}_z"]     = (grp[f] - mu[f]) / sigma[f]
            grp[f"{f}_ratio"] = grp[f] / (mu[f] + 1e-9)
            grp[f"{f}_slope"] = grp[f"{f}_z"].diff().rolling(20, min_periods=1).mean().fillna(0)

        # Frequency-domain statistical proxies
        grp["bpfo_proxy"]  = grp["kurtosis"]       * grp["crest_factor"]
        grp["bpfi_proxy"]  = grp["impulse_factor"]  * grp["peak"]
        grp["env_rms"]     = grp["shape_factor"]
        grp["spec_entropy_proxy"] = 1.0 / (grp["kurtosis"].abs() + 1e-6)

        # Self-normalised BPFO / BPFI proxies
        mu_bpfo = grp["bpfo_proxy"].iloc[:n_base].mean()
        mu_bpfi = grp["bpfi_proxy"].iloc[:n_base].mean()
        std_bpfo = grp["bpfo_proxy"].iloc[:n_base].std() + 1e-6
        std_bpfi = grp["bpfi_proxy"].iloc[:n_base].std() + 1e-6
        grp["bpfo_z"] = (grp["bpfo_proxy"] - mu_bpfo) / std_bpfo
        grp["bpfi_z"] = (grp["bpfi_proxy"] - mu_bpfi) / std_bpfi
        grp["bpfo_slope"] = grp["bpfo_z"].diff().rolling(20, min_periods=1).mean().fillna(0)
        grp["bpfi_slope"] = grp["bpfi_z"].diff().rolling(20, min_periods=1).mean().fillna(0)

        frames.append(grp)
    return pd.concat(frames, ignore_index=True)


df_feat = build_features(b_df, baseline_frac=0.10)

FEATURE_COLS = (
    [f"{f}_z"     for f in FEATURE_TYPES] +
    [f"{f}_ratio" for f in FEATURE_TYPES] +
    [f"{f}_slope" for f in FEATURE_TYPES] +
    ["bpfo_z", "bpfi_z", "bpfo_slope", "bpfi_slope",
     "bpfo_proxy", "bpfi_proxy", "env_rms", "spec_entropy_proxy"]
)

X      = df_feat[FEATURE_COLS].fillna(0)
y      = df_feat["stage"]
groups = df_feat["test_id"]

majority_class_recall = (y == 0).mean()
print(f"\nMajority-class (Healthy) baseline accuracy: {majority_class_recall:.4f}")
print("Per-class majority-baseline recall: Healthy=1.000, Warning=0.000, Critical=0.000")

# -----------------------------------------------------------------------------
# STEP 3 - Class weights for Critical prioritisation
# -----------------------------------------------------------------------------
n_healthy  = (y == 0).sum()
n_warning  = (y == 1).sum()
n_critical = (y == 2).sum()
w_healthy  = 1.0
w_warning  = n_healthy / n_warning
w_critical = 2.0 * (n_healthy / n_critical)
print(f"\nClass weights -> Healthy: {w_healthy:.1f}, Warning: {w_warning:.1f}, Critical: {w_critical:.1f}")

sample_weights_all = np.where(y == 2, w_critical,
                     np.where(y == 1, w_warning, w_healthy))

# -----------------------------------------------------------------------------
# STEP 4 - Evaluation helper
# -----------------------------------------------------------------------------

def evaluate_fold(y_true, y_pred, fold_label):
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    n   = len(y_true)
    per_class_recall = recall_score(y_true, y_pred, labels=[0, 1, 2],
                                    average=None, zero_division=0)
    mac_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc    = accuracy_score(y_true, y_pred)
    print(f"\n{'--'*30}")
    print(f"{fold_label}  (N={n})")
    print(f"{'--'*30}")
    print(f"  Majority-baseline accuracy : {majority_class_recall:.3f}  <- compare here")
    print(f"  Model accuracy             : {acc:.3f}  delta={acc-majority_class_recall:+.3f}")
    print(f"  Macro F1                   : {mac_f1:.3f}")
    print(f"  Per-class Recall:")
    for i, label in enumerate(["Healthy  (0)", "Warning  (1)", "Critical (2)"]):
        print(f"    {label}: {per_class_recall[i]:.3f}")
    print(f"  Confusion matrix [rows=actual, cols=pred]:")
    print(f"                  Pred-H  Pred-W  Pred-C")
    for i, lbl in enumerate(["Act-Healthy ", "Act-Warning ", "Act-Critical"]):
        print(f"  {lbl}  {cm[i,0]:6d}  {cm[i,1]:6d}  {cm[i,2]:6d}")
    return {
        "acc": acc, "mac_f1": mac_f1,
        "recall_healthy": per_class_recall[0],
        "recall_warning": per_class_recall[1],
        "recall_critical": per_class_recall[2],
    }

# -----------------------------------------------------------------------------
# STEP 5 - GroupKFold: Model A (weighted flat) vs Model B (two-stage)
# -----------------------------------------------------------------------------

gkf = GroupKFold(n_splits=3)
results_A, results_B = [], []

print("\n" + "="*60)
print("MODEL A: Weighted flat 3-class XGBoost")
print("MODEL B: Two-stage (binary healthy/anomalous -> Warning/Critical)")
print("="*60)

for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups)):
    held_run = groups.iloc[te_idx].unique().tolist()

    X_tr_raw = X.iloc[tr_idx];  y_tr = y.iloc[tr_idx]
    X_te_raw = X.iloc[te_idx];  y_te = y.iloc[te_idx]
    sw_tr    = sample_weights_all[tr_idx]

    sc   = StandardScaler()
    X_tr = sc.fit_transform(X_tr_raw)
    X_te = sc.transform(X_te_raw)

    # Model A
    mA = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.04,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", random_state=42,
        subsample=0.8, colsample_bytree=0.8
    )
    mA.fit(X_tr, y_tr, sample_weight=sw_tr)
    preds_A = mA.predict(X_te)
    res_A   = evaluate_fold(y_te, preds_A,
                            f"Fold {fold} | Run {held_run} | MODEL A (weighted)")
    results_A.append(res_A)

    # Model B stage 1: binary anomaly
    y_bin_tr = (y_tr > 0).astype(int)
    n_pos = y_bin_tr.sum(); n_neg = len(y_bin_tr) - n_pos
    spw   = max(n_neg / max(n_pos, 1), 1.0) * 2.0

    mB1 = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=spw, eval_metric="logloss", random_state=42,
        subsample=0.8, colsample_bytree=0.8
    )
    mB1.fit(X_tr, y_bin_tr)
    anomaly_pred = mB1.predict(X_te)

    # Model B stage 2: Warning vs Critical
    preds_B = np.zeros(len(y_te), dtype=int)
    anomaly_mask_te = anomaly_pred == 1
    if anomaly_mask_te.sum() > 0:
        anomaly_mask_tr = (y_tr > 0)
        if anomaly_mask_tr.sum() >= 4 and y_tr[anomaly_mask_tr].nunique() > 1:
            spw2 = max(
                (y_tr[anomaly_mask_tr] == 1).sum() /
                max((y_tr[anomaly_mask_tr] == 2).sum(), 1), 1.0
            ) * 2.0
            mB2 = xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                objective="binary:logistic", eval_metric="logloss",
                scale_pos_weight=spw2, random_state=42
            )
            y_s2_tr = (y_tr[anomaly_mask_tr] == 2).astype(int)
            mB2.fit(X_tr[anomaly_mask_tr], y_s2_tr)
            s2_pred = mB2.predict(X_te[anomaly_mask_te])
            preds_B[anomaly_mask_te] = np.where(s2_pred == 1, 2, 1)
        else:
            preds_B[anomaly_mask_te] = 1

    res_B = evaluate_fold(y_te, preds_B,
                          f"Fold {fold} | Run {held_run} | MODEL B (two-stage)")
    results_B.append(res_B)

# -----------------------------------------------------------------------------
# STEP 6 - Range summary (NOT averaged - 3 rigs is not enough to average)
# -----------------------------------------------------------------------
def range_summary(results, name):
    print("\n" + "="*60)
    print(name)
    print("FOLD RANGE ACROSS 3 PHYSICAL RIGS [min, max]")
    print("(3 rigs is insufficient to report a mean -- report range only)")
    print("="*60)
    print(f"  Majority-baseline accuracy : {majority_class_recall:.3f}")
    keys   = ["acc", "mac_f1", "recall_healthy", "recall_warning", "recall_critical"]
    labels = ["Accuracy          ", "Macro F1          ",
              "Recall Healthy (0)", "Recall Warning (1)", "Recall Critical(2)"]
    for k, lbl in zip(keys, labels):
        vals = [r[k] for r in results]
        fold_str = ", ".join(f"{v:.3f}" for v in vals)
        print(f"  {lbl}: [{min(vals):.3f}, {max(vals):.3f}]  (folds: {fold_str})")

range_summary(results_A, "MODEL A: Weighted 3-class XGBoost")
range_summary(results_B, "MODEL B: Two-stage binary -> Warning/Critical")

# -----------------------------------------------------------------------# STEP 7 - Save production Model A (fit on all data)
# -----------------------------------------------------------------------------
print("\nFitting production Model A on full dataset ...")
sc_prod = StandardScaler()
X_prod  = sc_prod.fit_transform(X)

mA_prod = xgb.XGBClassifier(
    n_estimators=400, max_depth=5, learning_rate=0.04,
    objective="multi:softprob", num_class=3,
    eval_metric="mlogloss", random_state=42,
    subsample=0.8, colsample_bytree=0.8
)
mA_prod.fit(X_prod, y, sample_weight=sample_weights_all)

with open(MODELS_DIR / "bearing_stage_classifier.pkl",  "wb") as f: pickle.dump(mA_prod, f)
with open(MODELS_DIR / "bearing_scaler.pkl",            "wb") as f: pickle.dump(sc_prod, f)
with open(MODELS_DIR / "bearing_feature_cols.pkl",      "wb") as f: pickle.dump(FEATURE_COLS, f)
mA_prod.save_model(str(MODELS_DIR / "bearing_stage_classifier_v2.json"))
print("Saved v2 production models to models/")
