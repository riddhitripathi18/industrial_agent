"""
train_bearing_model_v3.py
=========================
Comprehensive evaluation addressing Tasks 1, 2, 3, and 4:
- Task 1: Sensor configuration modeling (global with interaction vs conditioned)
- Task 2: Frequency-domain features (FFT harmonics, envelope BPFO/BPFI, spectral kurtosis)
- Task 3: Collapsing to 2-class problem (Healthy vs Anomalous: Warning + Critical)
- Task 4: Complete per-fold confusion matrices, per-class recall vs majority baseline,
          accuracy delta, macro F1, and range reporting.
"""

import os, sys, pickle, warnings, time
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
    confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
)

warnings.filterwarnings("ignore")

ROOT_DIR   = Path(r"c:\Users\riddh\OneDrive\Desktop\industrial_agent")
DATA_PATH  = ROOT_DIR / "bearings_all_features.csv"
FREQ_PATH  = ROOT_DIR / "bearings_freq_raw_features.csv"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

FAILED_MAP = {1: ["Bearing3", "Bearing4"], 2: ["Bearing1"], 3: ["Bearing3"]}
TIME_FEATS = [
    "rms", "peak", "peak_to_peak", "kurtosis", "crest_factor",
    "std", "skewness", "variance", "mean_abs", "shape_factor", "impulse_factor"
]
FREQ_FEATS = [
    "env_bpfo", "env_bpfo2", "env_bpfi", "env_bsf",
    "fft_bpfo", "fft_bpfi", "fft_bsf", "sk_mean", "sk_max"
]

def load_and_merge_data():
    print("Loading base features ...")
    df_raw = pd.read_csv(DATA_PATH)
    test_max_hours = df_raw.groupby("test_id")["hours_elapsed"].max().to_dict()

    has_freq = FREQ_PATH.exists()
    if has_freq:
        print(f"Loading raw frequency-domain features from {FREQ_PATH.name} ...")
        df_freq = pd.read_csv(FREQ_PATH)
        df_merged = pd.merge(df_raw, df_freq, on=["test_id", "file"], how="left")
    else:
        print("Notice: Raw frequency features CSV not found yet; proceeding with time features.")
        df_merged = df_raw

    rows = []
    for _, row in df_merged.iterrows():
        test_id = int(row["test_id"])
        hr      = float(row["hours_elapsed"])
        fname   = str(row["file"])
        max_hr  = test_max_hours[test_id]
        life_pct = hr / max_hr if max_hr > 0 else 0.0

        for b_idx in [1, 2, 3, 4]:
            b_name     = f"Bearing{b_idx}"
            is_failing = b_name in FAILED_MAP[test_id]
            if is_failing:
                stage = 0 if life_pct < 0.75 else (1 if life_pct < 0.92 else 2)
            else:
                stage = 0  # companion bearings stayed healthy

            feat_dict = {
                "test_id":       test_id,
                "bearing_id":    b_name,
                "file":          fname,
                "hours_elapsed": hr,
                "stage":         stage,
                "stage_binary":  1 if stage in [1, 2] else 0, # Task 3: 1 = Anomalous
                "is_dual_axis":  1 if test_id == 1 else 0,     # Task 1: Hardware flag
            }
            
            # Extract time features
            for ft in TIME_FEATS:
                col   = f"{b_name}_{ft}"
                col_x = f"{b_name}_x_{ft}"
                if col in row and pd.notna(row[col]):
                    feat_dict[ft] = float(row[col])
                elif col_x in row and pd.notna(row[col_x]):
                    feat_dict[ft] = float(row[col_x])
                else:
                    feat_dict[ft] = 0.0

            # Extract frequency features
            for ff in FREQ_FEATS:
                col = f"{b_name}_{ff}"
                if col in row and pd.notna(row[col]):
                    feat_dict[ff] = float(row[col])
                else:
                    feat_dict[ff] = 0.0

            if feat_dict["rms"] > 0:
                rows.append(feat_dict)

    b_df = pd.DataFrame(rows)
    print(f"Total unrolled observations: {len(b_df)}")
    return b_df

def compute_drift_features(df_in, baseline_frac=0.10):
    df_in = df_in.sort_values(["test_id", "bearing_id", "hours_elapsed"]).copy()
    frames = []
    
    all_raw_cols = TIME_FEATS + FREQ_FEATS
    
    for (tid, bid), grp in df_in.groupby(["test_id", "bearing_id"]):
        grp = grp.sort_values("hours_elapsed").copy()
        n_base = max(int(len(grp) * baseline_frac), 10)
        base   = grp.iloc[:n_base][all_raw_cols]
        mu     = base.mean()
        sigma  = base.std().replace(0, 1e-6).fillna(1e-6)

        for f in all_raw_cols:
            grp[f"{f}_z"]     = (grp[f] - mu[f]) / sigma[f]
            grp[f"{f}_ratio"] = grp[f] / (mu[f] + 1e-9)
            grp[f"{f}_slope"] = grp[f"{f}_z"].diff().rolling(20, min_periods=1).mean().fillna(0)

        # Hardware interaction features (Task 1)
        for f in ["rms", "kurtosis", "env_bpfo", "sk_mean"]:
            grp[f"dual_x_{f}_z"] = grp["is_dual_axis"] * grp[f"{f}_z"]

        frames.append(grp)
        
    return pd.concat(frames, ignore_index=True)

def print_cm_3x3(cm):
    print("                  Pred-H  Pred-W  Pred-C")
    for i, lbl in enumerate(["Act-Healthy ", "Act-Warning ", "Act-Critical"]):
        print(f"  {lbl}  {cm[i,0]:6d}  {cm[i,1]:6d}  {cm[i,2]:6d}")

def print_cm_2x2(cm):
    print("                  Pred-Healthy  Pred-Anomalous")
    for i, lbl in enumerate(["Act-Healthy  ", "Act-Anomalous"]):
        print(f"  {lbl}  {cm[i,0]:12d}  {cm[i,1]:14d}")

def run_evaluation():
    b_df = load_and_merge_data()
    df_feat = compute_drift_features(b_df)
    
    y3 = df_feat["stage"]
    y2 = df_feat["stage_binary"]
    groups = df_feat["test_id"]
    
    n_total = len(df_feat)
    maj_recall_h = (y3 == 0).mean()
    maj_recall_a = (y2 == 0).mean() # majority baseline for binary is predicting Healthy
    
    print("\n" + "="*70)
    print("GROUND TRUTH CLASS DISTRIBUTION")
    print(f"Total: {n_total}")
    print(f"3-Class: Healthy(0)={(y3==0).sum()} ({(y3==0).mean()*100:.2f}%), Warning(1)={(y3==1).sum()} ({(y3==1).mean()*100:.2f}%), Critical(2)={(y3==2).sum()} ({(y3==2).mean()*100:.2f}%)")
    print(f"2-Class: Healthy(0)={(y2==0).sum()} ({(y2==0).mean()*100:.2f}%), Anomalous(1)={(y2==1).sum()} ({(y2==1).mean()*100:.2f}%)")
    print(f"Majority Baseline Accuracy (Predict Healthy always): {maj_recall_h*100:.2f}%")
    print("="*70)
    
    # Feature sets
    time_feature_cols = (
        [f"{f}_z" for f in TIME_FEATS] +
        [f"{f}_ratio" for f in TIME_FEATS] +
        [f"{f}_slope" for f in TIME_FEATS]
    )
    freq_feature_cols = (
        [f"{f}_z" for f in FREQ_FEATS] +
        [f"{f}_ratio" for f in FREQ_FEATS] +
        [f"{f}_slope" for f in FREQ_FEATS]
    )
    interaction_cols = [
        "is_dual_axis", "dual_x_rms_z", "dual_x_kurtosis_z",
        "dual_x_env_bpfo_z", "dual_x_sk_mean_z"
    ]
    
    # -------------------------------------------------------------------------
    # Helper to train and evaluate GroupKFold
    # -------------------------------------------------------------------------
    def eval_gkf_3class(feat_cols, model_name, class_weights=True):
        print("\n" + "#"*70)
        print(f"EVALUATING 3-CLASS MODEL: {model_name}")
        print(f"Features: {len(feat_cols)} columns")
        print("#"*70)
        
        X = df_feat[feat_cols].fillna(0)
        gkf = GroupKFold(n_splits=3)
        
        fold_results = []
        for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y3, groups)):
            held_run = groups.iloc[te_idx].unique().tolist()[0]
            n_te = len(te_idx)
            
            X_tr, y_tr = X.iloc[tr_idx], y3.iloc[tr_idx]
            X_te, y_te = X.iloc[te_idx], y3.iloc[te_idx]
            
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)
            
            if class_weights:
                w_h = 1.0
                w_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
                w_c = 2.0 * ((y_tr == 0).sum() / max((y_tr == 2).sum(), 1))
                sw_tr = np.where(y_tr == 2, w_c, np.where(y_tr == 1, w_w, w_h))
            else:
                sw_tr = None
                
            clf = xgb.XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.04,
                objective="multi:softprob", num_class=3,
                eval_metric="mlogloss", random_state=42,
                subsample=0.8, colsample_bytree=0.8, n_jobs=-1
            )
            clf.fit(X_tr_s, y_tr, sample_weight=sw_tr)
            y_pred = clf.predict(X_te_s)
            
            acc = accuracy_score(y_te, y_pred)
            mac_f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_te, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
            cm = confusion_matrix(y_te, y_pred, labels=[0, 1, 2])
            
            delta = acc - maj_recall_h
            underperformed = acc < maj_recall_h
            
            print(f"\n--- Fold {fold} | Held-out: Test Run {held_run} (N={n_te}) ---")
            print(f"  Majority Baseline Acc: {maj_recall_h*100:.2f}%")
            print(f"  Model Accuracy       : {acc*100:.2f}%  (delta: {delta*100:+.2f}%) {'[UNDERPERFORMS BASELINE]' if underperformed else ''}")
            print(f"  Macro F1             : {mac_f1:.4f}")
            print(f"  Per-class Recall: Healthy(0)={rec[0]*100:.2f}%, Warning(1)={rec[1]*100:.2f}%, Critical(2)={rec[2]*100:.2f}%")
            print("  Confusion Matrix:")
            print_cm_3x3(cm)
            
            fold_results.append({
                "fold": fold, "held_run": held_run, "n": n_te,
                "acc": acc, "delta": delta, "mac_f1": mac_f1,
                "rec_h": rec[0], "rec_w": rec[1], "rec_c": rec[2],
                "cm": cm, "underperformed": underperformed
            })
            
        print(f"\n=== SUMMARY RANGE ACROSS 3 PHYSICAL RIGS for {model_name} ===")
        print(f"  Accuracy          : [{min(r['acc'] for r in fold_results)*100:.2f}%, {max(r['acc'] for r in fold_results)*100:.2f}%]")
        print(f"  Macro F1          : [{min(r['mac_f1'] for r in fold_results):.4f}, {max(r['mac_f1'] for r in fold_results):.4f}]")
        print(f"  Recall Healthy (0): [{min(r['rec_h'] for r in fold_results)*100:.2f}%, {max(r['rec_h'] for r in fold_results)*100:.2f}%]")
        print(f"  Recall Warning (1): [{min(r['rec_w'] for r in fold_results)*100:.2f}%, {max(r['rec_w'] for r in fold_results)*100:.2f}%]")
        print(f"  Recall Critical(2): [{min(r['rec_c'] for r in fold_results)*100:.2f}%, {max(r['rec_c'] for r in fold_results)*100:.2f}%]")
        return fold_results

    # -------------------------------------------------------------------------
    # Helper to evaluate binary model (Task 3)
    # -------------------------------------------------------------------------
    def eval_gkf_2class(feat_cols, model_name):
        print("\n" + "#"*70)
        print(f"EVALUATING 2-CLASS (BINARY) MODEL: {model_name}")
        print(f"Target: Healthy (0) vs. Anomalous (1 = Warning + Critical)")
        print(f"Features: {len(feat_cols)} columns")
        print("#"*70)
        
        X = df_feat[feat_cols].fillna(0)
        gkf = GroupKFold(n_splits=3)
        
        fold_results = []
        for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y2, groups)):
            held_run = groups.iloc[te_idx].unique().tolist()[0]
            n_te = len(te_idx)
            
            X_tr, y_tr = X.iloc[tr_idx], y2.iloc[tr_idx]
            X_te, y_te = X.iloc[te_idx], y2.iloc[te_idx]
            
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)
            
            # Asymmetric weight for anomalous class
            scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
            
            clf = xgb.XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.04,
                scale_pos_weight=scale_pos,
                eval_metric="logloss", random_state=42,
                subsample=0.8, colsample_bytree=0.8, n_jobs=-1
            )
            clf.fit(X_tr_s, y_tr)
            y_pred = clf.predict(X_te_s)
            
            acc = accuracy_score(y_te, y_pred)
            mac_f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_te, y_pred, labels=[0, 1], average=None, zero_division=0)
            prec_anom = precision_score(y_te, y_pred, pos_label=1, zero_division=0)
            f1_anom = f1_score(y_te, y_pred, pos_label=1, zero_division=0)
            cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
            
            delta = acc - maj_recall_a
            underperformed = acc < maj_recall_a
            
            print(f"\n--- Fold {fold} | Held-out: Test Run {held_run} (N={n_te}) ---")
            print(f"  Majority Baseline Acc: {maj_recall_a*100:.2f}%")
            print(f"  Model Accuracy       : {acc*100:.2f}%  (delta: {delta*100:+.2f}%) {'[UNDERPERFORMS BASELINE]' if underperformed else ''}")
            print(f"  Macro F1             : {mac_f1:.4f}")
            print(f"  Anomalous Precision  : {prec_anom*100:.2f}%")
            print(f"  Anomalous Recall     : {rec[1]*100:.2f}%")
            print(f"  Anomalous F1         : {f1_anom:.4f}")
            print(f"  Healthy Recall       : {rec[0]*100:.2f}%")
            print("  Confusion Matrix:")
            print_cm_2x2(cm)
            
            fold_results.append({
                "fold": fold, "held_run": held_run, "n": n_te,
                "acc": acc, "delta": delta, "mac_f1": mac_f1,
                "rec_h": rec[0], "rec_a": rec[1], "prec_a": prec_anom, "f1_a": f1_anom,
                "cm": cm, "underperformed": underperformed
            })
            
        print(f"\n=== SUMMARY RANGE ACROSS 3 PHYSICAL RIGS for {model_name} ===")
        print(f"  Accuracy          : [{min(r['acc'] for r in fold_results)*100:.2f}%, {max(r['acc'] for r in fold_results)*100:.2f}%]")
        print(f"  Macro F1          : [{min(r['mac_f1'] for r in fold_results):.4f}, {max(r['mac_f1'] for r in fold_results):.4f}]")
        print(f"  Anomalous Recall  : [{min(r['rec_a'] for r in fold_results)*100:.2f}%, {max(r['rec_a'] for r in fold_results)*100:.2f}%]")
        print(f"  Anomalous Prec    : [{min(r['prec_a'] for r in fold_results)*100:.2f}%, {max(r['prec_a'] for r in fold_results)*100:.2f}%]")
        print(f"  Anomalous F1      : [{min(r['f1_a'] for r in fold_results):.4f}, {max(r['f1_a'] for r in fold_results):.4f}]")
        print(f"  Healthy Recall    : [{min(r['rec_h'] for r in fold_results)*100:.2f}%, {max(r['rec_h'] for r in fold_results)*100:.2f}%]")
        return fold_results

    # -------------------------------------------------------------------------
    # Helper to evaluate sensor-conditioned models (Task 1b)
    # -------------------------------------------------------------------------
    def eval_sensor_conditioned(feat_cols):
        print("\n" + "#"*70)
        print("EVALUATING SENSOR-CONDITIONED MODELS (Task 1b)")
        print("Separating single-axis hardware (Runs 2 & 3) vs. dual-axis (Run 1)")
        print("#"*70)
        
        # 1. Single-axis hardware only: Runs 2 & 3
        df_single = df_feat[df_feat["is_dual_axis"] == 0].copy()
        X_s = df_single[feat_cols].fillna(0)
        y_s = df_single["stage"]
        grp_s = df_single["test_id"]
        
        print("\n--- SINGLE-AXIS RIGS ONLY (Test Run 2 vs Test Run 3) ---")
        print(f"Single-axis observations: {len(df_single)}")
        print("Distribution:", y_s.value_counts().to_dict())
        
        # 2-fold cross-rig validation (Train Run 2, Test Run 3; Train Run 3, Test Run 2)
        single_results = []
        for te_run, tr_run in [(3, 2), (2, 3)]:
            tr_m = (grp_s == tr_run)
            te_m = (grp_s == te_run)
            
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_s[tr_m])
            X_te = sc.transform(X_s[te_m])
            y_tr = y_s[tr_m]
            y_te = y_s[te_m]
            
            w_h = 1.0
            w_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
            w_c = 2.0 * ((y_tr == 0).sum() / max((y_tr == 2).sum(), 1))
            sw = np.where(y_tr == 2, w_c, np.where(y_tr == 1, w_w, w_h))
            
            clf = xgb.XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.04,
                objective="multi:softprob", num_class=3,
                eval_metric="mlogloss", random_state=42,
                subsample=0.8, colsample_bytree=0.8, n_jobs=-1
            )
            clf.fit(X_tr, y_tr, sample_weight=sw)
            y_pred = clf.predict(X_te)
            
            acc = accuracy_score(y_te, y_pred)
            mac_f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_te, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
            cm = confusion_matrix(y_te, y_pred, labels=[0, 1, 2])
            
            print(f"\nConditioned Single-Axis: Train Run {tr_run} -> Test Run {te_run} (N={len(y_te)})")
            print(f"  Accuracy: {acc*100:.2f}%, Macro F1: {mac_f1:.4f}")
            print(f"  Recall: Healthy={rec[0]*100:.2f}%, Warning={rec[1]*100:.2f}%, Critical={rec[2]*100:.2f}%")
            print("  Confusion matrix:")
            print_cm_3x3(cm)
            single_results.append({
                "tr_run": tr_run, "te_run": te_run, "acc": acc, "mac_f1": mac_f1,
                "rec_h": rec[0], "rec_w": rec[1], "rec_c": rec[2], "cm": cm
            })
            
        print("\nSingle-Axis Conditioned Range [min, max]:")
        print(f"  Accuracy          : [{min(r['acc'] for r in single_results)*100:.2f}%, {max(r['acc'] for r in single_results)*100:.2f}%]")
        print(f"  Critical Recall   : [{min(r['rec_c'] for r in single_results)*100:.2f}%, {max(r['rec_c'] for r in single_results)*100:.2f}%]")
        print(f"  Warning Recall    : [{min(r['rec_w'] for r in single_results)*100:.2f}%, {max(r['rec_w'] for r in single_results)*100:.2f}%]")
        
        # 2. Dual-axis hardware (Run 1 only)
        # Note: Run 1 is the ONLY dual-axis rig in existence.
        # We test temporal/bearing split within Run 1 to measure internal predictability:
        df_dual = df_feat[df_feat["is_dual_axis"] == 1].copy()
        print(f"\n--- DUAL-AXIS RIG (Test Run 1 Only: N={len(df_dual)}) ---")
        print("Note: Only 1 physical dual-axis rig exists in dataset.")
        print("Leave-bearing-out within Run 1: Train on Bearings 1, 2, 4 -> Test on failing Bearing 3:")
        b3_mask = (df_dual["bearing_id"] == "Bearing3")
        X_d = df_dual[feat_cols].fillna(0)
        y_d = df_dual["stage"]
        
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_d[~b3_mask])
        X_te = sc.transform(X_d[b3_mask])
        y_tr = y_d[~b3_mask]
        y_te = y_d[b3_mask]
        
        w_h = 1.0
        w_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        w_c = 2.0 * ((y_tr == 0).sum() / max((y_tr == 2).sum(), 1))
        sw = np.where(y_tr == 2, w_c, np.where(y_tr == 1, w_w, w_h))
        
        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.04,
            objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", random_state=42,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1
        )
        clf.fit(X_tr, y_tr, sample_weight=sw)
        y_pred = clf.predict(X_te)
        
        acc_d = accuracy_score(y_te, y_pred)
        mac_f1_d = f1_score(y_te, y_pred, average="macro", zero_division=0)
        rec_d = recall_score(y_te, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
        cm_d = confusion_matrix(y_te, y_pred, labels=[0, 1, 2])
        print(f"Dual-axis Intra-Rig (Test Bearing 3, N={len(y_te)}):")
        print(f"  Accuracy: {acc_d*100:.2f}%, Macro F1: {mac_f1_d:.4f}")
        print(f"  Recall: Healthy={rec_d[0]*100:.2f}%, Warning={rec_d[1]*100:.2f}%, Critical={rec_d[2]*100:.2f}%")
        print("  Confusion matrix:")
        print_cm_3x3(cm_d)
        
        return single_results, {
            "acc": acc_d, "mac_f1": mac_f1_d, "rec_h": rec_d[0], "rec_w": rec_d[1], "rec_c": rec_d[2], "cm": cm_d
        }

    # =========================================================================
    # EXECUTE ALL TASKS
    # =========================================================================
    
    # Baseline (Time features only)
    res_baseline = eval_gkf_3class(time_feature_cols, "Baseline Time-Domain 3-Class (v2)")
    
    # Task 1a: Single global model with hardware flag + interactions
    t1a_feats = time_feature_cols + interaction_cols
    res_t1a = eval_gkf_3class(t1a_feats, "Task 1a: Global + Sensor Config & Interaction Terms")
    
    # Task 1b: Conditioned models
    res_t1b_single, res_t1b_dual = eval_sensor_conditioned(time_feature_cols)
    
    # Task 2: Adding raw frequency-domain features (FFT harmonics, envelope BPFO/BPFI, spectral kurtosis)
    t2_feats = time_feature_cols + freq_feature_cols
    res_t2 = eval_gkf_3class(t2_feats, "Task 2: Time + Raw Frequency/Envelope Features")
    
    # Task 3: Collapsing to 2-class problem (Healthy vs Anomalous)
    res_t3 = eval_gkf_2class(time_feature_cols + freq_feature_cols, "Task 3: Interim Binary (Healthy vs Anomalous)")

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("="*80)

if __name__ == "__main__":
    run_evaluation()
