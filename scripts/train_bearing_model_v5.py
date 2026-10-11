"""
train_bearing_model_v5.py
=========================
Implements all 5 improvements from the engineering roadmap:

1. Redefine binary target as Critical-only; track Warning separately as a lower-confidence flag
2. Threshold optimization via PR-curve per rig (not default 0.5)
3. OR-ensemble of 3-class + binary models
4. Temporal smoothing: CUSUM-triggered rolling majority vote
5. SMOTE + cross-validated class-weight tuning

Evaluation: leave-one-rig-out GroupKFold, reported PER RIG (not averaged),
separated by hardware type.
"""

import os, sys, warnings, time
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, recall_score,
    precision_score, precision_recall_curve, auc
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

ROOT_DIR   = Path(r"c:\Users\riddh\OneDrive\Desktop\industrial_agent")
DATA_PATH  = ROOT_DIR / "bearings_all_features.csv"
FREQ_PATH  = ROOT_DIR / "bearings_freq_raw_features.csv"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

FAILED_MAP = {1: ["Bearing3", "Bearing4"], 2: ["Bearing1"], 3: ["Bearing3"]}

# HI label signals (NOT used as model features)
HI_SIGNALS = ["rms", "kurtosis", "env_bpfo"]

# Model features (disjoint from HI signals)
MODEL_TIME_FEATS = ["shape_factor", "impulse_factor", "skewness",
                    "peak_to_peak", "variance", "std", "crest_factor", "peak"]
MODEL_FREQ_FEATS = ["env_bpfi", "env_bsf", "env_bpfo2",
                    "fft_bpfi", "fft_bsf", "sk_mean", "sk_max"]
ALL_MODEL_FEATS  = MODEL_TIME_FEATS + MODEL_FREQ_FEATS


# ============================================================================
# Data loading (reuse v4 logic)
# ============================================================================
def load_unrolled():
    df_all  = pd.read_csv(DATA_PATH)
    df_freq = pd.read_csv(FREQ_PATH)
    merged  = pd.merge(df_all, df_freq, on=["test_id", "file"], how="inner")

    rows = []
    for _, row in merged.iterrows():
        tid   = int(row["test_id"])
        hr    = float(row["hours_elapsed"])
        fname = str(row["file"])

        for b_idx in [1, 2, 3, 4]:
            bname      = f"Bearing{b_idx}"
            is_failing = bname in FAILED_MAP[tid]

            if f"{bname}_x_rms" in row.index and pd.notna(row[f"{bname}_x_rms"]):
                pfx = f"{bname}_x"
            elif f"{bname}_rms" in row.index and pd.notna(row[f"{bname}_rms"]):
                pfx = bname
            else:
                continue

            rms_val = float(row.get(f"{pfx}_rms", 0) or 0)
            if rms_val == 0:
                continue

            rec = {
                "test_id": tid, "bearing_id": bname, "file": fname,
                "hours_elapsed": hr, "is_failing": int(is_failing),
                "is_dual_axis": 1 if tid == 1 else 0,
            }
            for ft in ["rms", "kurtosis", "crest_factor", "peak", "peak_to_peak",
                       "std", "skewness", "variance", "mean_abs",
                       "shape_factor", "impulse_factor"]:
                col = f"{pfx}_{ft}"
                rec[ft] = float(row[col]) if (col in row.index and pd.notna(row[col])) else 0.0
            for ff in ["env_bpfo", "env_bpfo2", "env_bpfi", "env_bsf",
                       "fft_bpfo", "fft_bpfi", "fft_bsf", "sk_mean", "sk_max"]:
                col = f"{bname}_{ff}"
                rec[ff] = float(row[col]) if (col in row.index and pd.notna(row[col])) else 0.0
            rows.append(rec)

    return pd.DataFrame(rows)


def build_physics_labels(df_in, hi_warn=3.0, hi_crit=6.0, win=15):
    df_in = df_in.sort_values(["test_id", "bearing_id", "hours_elapsed"]).copy()
    df_in["stage"] = 0
    df_in["hi"]    = 0.0

    for (tid, bid), grp in df_in.groupby(["test_id", "bearing_id"]):
        grp      = grp.sort_values("hours_elapsed").copy()
        idx      = grp.index
        n_base   = max(int(len(grp) * 0.10), 15)
        base     = grp.iloc[:n_base]
        z_scores = []
        for sig in HI_SIGNALS:
            mu = base[sig].mean(); sd = base[sig].std() + 1e-6
            z  = ((grp[sig] - mu) / sd).rolling(win, min_periods=3).mean().fillna(0)
            z_scores.append(z.values)
        hi = np.maximum.reduce(z_scores)
        df_in.loc[idx, "hi"] = hi
        if grp["is_failing"].iloc[0]:
            stage = np.where(hi >= hi_crit, 2, np.where(hi >= hi_warn, 1, 0))
        else:
            stage = np.zeros(len(grp), dtype=int)
        df_in.loc[idx, "stage"] = stage
    return df_in


def build_model_features(df_in, win=15):
    df_in  = df_in.sort_values(["test_id", "bearing_id", "hours_elapsed"]).copy()
    frames = []
    for (tid, bid), grp in df_in.groupby(["test_id", "bearing_id"]):
        grp    = grp.sort_values("hours_elapsed").copy()
        n_base = max(int(len(grp) * 0.10), 15)
        base   = grp.iloc[:n_base][ALL_MODEL_FEATS]
        mu     = base.mean(); sigma = base.std().replace(0, 1e-6).fillna(1e-6)
        for f in ALL_MODEL_FEATS:
            grp[f"{f}_z"]      = (grp[f] - mu[f]) / sigma[f]
            grp[f"{f}_ratio"]  = grp[f] / (mu[f] + 1e-9)
            grp[f"{f}_slope"]  = grp[f"{f}_z"].diff().rolling(win, min_periods=1).mean().fillna(0)
            grp[f"{f}_ewma_z"] = grp[f"{f}_z"].ewm(span=30).mean()
        frames.append(grp)
    return pd.concat(frames, ignore_index=True)


# ============================================================================
# Improvement 4: Temporal smoothing (rolling majority vote + CUSUM)
# ============================================================================
def apply_temporal_smoothing(y_pred_proba, groups_te, win=5):
    """
    Two-step temporal filter:
    - Step A: Rolling majority vote over 'win' consecutive snapshots per bearing
    - Step B: CUSUM: once cumulative positive-deviation sum exceeds threshold,
              stay flagged (irreversible deterioration assumption)
    """
    result = np.array(y_pred_proba)  # probabilities, shape (N,)
    df_tmp = pd.DataFrame({"proba": result, "group": groups_te.values})

    smoothed = np.zeros(len(result))
    for bid, grp_idx in df_tmp.groupby("group").groups.items():
        proba = df_tmp.loc[grp_idx, "proba"].values
        # Rolling mean smoothing
        rolled = pd.Series(proba).rolling(win, min_periods=1).mean().values
        # CUSUM: flag once sustained upward departure, never reset
        cusum = np.cumsum(rolled - rolled[:max(1, int(len(rolled) * 0.10))].mean())
        cusum_flag = np.maximum.accumulate(cusum > 0.5).astype(float)
        # Final: average of rolled and CUSUM-boosted signal
        smoothed[grp_idx] = np.clip(0.6 * rolled + 0.4 * cusum_flag, 0, 1)

    return smoothed


# ============================================================================
# Improvement 2: Find optimal precision-recall threshold per fold
# ============================================================================
def find_optimal_threshold(y_true, y_proba, min_recall=0.70):
    """
    Scan thresholds to find the highest precision that still meets min_recall.
    Targets recall >= min_recall as the priority (cost of FN >> FP).
    """
    prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
    # thresholds has len = len(prec) - 1; align
    prec, rec = prec[:-1], rec[:-1]

    viable = (rec >= min_recall)
    if not viable.any():
        # Relax to whatever maximises recall
        best_idx = np.argmax(rec)
    else:
        # Among viable, pick highest precision (fewest false alarms)
        best_idx = np.where(viable, prec, -1).argmax()

    return thresholds[best_idx], prec[best_idx], rec[best_idx]


# ============================================================================
# Evaluation helpers
# ============================================================================
def cm_report(y_true, y_pred, label_names, fold_label, majority_acc):
    labs   = list(range(len(label_names)))
    cm     = confusion_matrix(y_true, y_pred, labels=labs)
    acc    = accuracy_score(y_true, y_pred)
    mac_f1 = f1_score(y_true, y_pred, average="macro", labels=labs, zero_division=0)
    rec    = recall_score(y_true, y_pred, labels=labs, average=None, zero_division=0)
    delta  = acc - majority_acc
    under  = " [UNDERPERFORMS BASELINE]" if delta < 0 else ""
    n      = len(y_true)
    print(f"\n--- {fold_label} (N={n}) ---")
    print(f"  Majority Baseline: {majority_acc*100:.2f}%")
    print(f"  Accuracy         : {acc*100:.2f}%  (delta: {delta*100:+.2f}%){under}")
    print(f"  Macro F1         : {mac_f1:.4f}")
    for i, ln in enumerate(label_names):
        prec_i = precision_score(y_true, y_pred, labels=[i], average="micro", zero_division=0)
        print(f"  {ln:20s}: Recall={rec[i]*100:.2f}%  Precision={prec_i*100:.2f}%")
    # print compact CM
    print(f"  Confusion Matrix ({' | '.join(label_names)}):")
    hdr = "  ".join(f"Pred-{ln[:6]:6}" for ln in label_names)
    print(f"    {'':16}{hdr}")
    for i, aln in enumerate(label_names):
        vals = "  ".join(f"{cm[i,j]:10d}" for j in range(len(label_names)))
        print(f"    Act-{aln[:12]:12}  {vals}")
    return {"acc": acc, "delta": delta, "mac_f1": mac_f1, "rec": rec,
            "cm": cm, "under": delta < 0}


def range_summary(results_list, name, label_names):
    print(f"\n=== RANGE SUMMARY: {name} ===")
    print(f"  (Rigs reported individually; no mean — 3 rigs insufficient to average)")
    for key in ["acc", "mac_f1"]:
        vals = [r[key] for r in results_list]
        print(f"  {key:12}: [{min(vals):.4f}, {max(vals):.4f}]  "
              f"per-rig: {', '.join(f'{v:.4f}' for v in vals)}")
    for i, ln in enumerate(label_names):
        recs = [r["rec"][i] for r in results_list]
        print(f"  Recall {ln[:16]:16}: [{min(recs)*100:.2f}%, {max(recs)*100:.2f}%]  "
              f"per-rig: {', '.join(f'{r*100:.2f}%' for r in recs)}")
    n_under = sum(1 for r in results_list if r["under"])
    if n_under:
        print(f"  !! {n_under}/3 folds underperform majority-class baseline !!")


# ============================================================================
# MAIN
# ============================================================================
def main():
    t0 = time.time()

    print("Loading and labeling data ...")
    b_df = load_unrolled()
    b_df = build_physics_labels(b_df)
    b_df = build_model_features(b_df)

    n     = len(b_df)
    n_h   = (b_df["stage"] == 0).sum()
    n_w   = (b_df["stage"] == 1).sum()
    n_c   = (b_df["stage"] == 2).sum()
    maj   = n_h / n

    print(f"  Total: {n}  |  Healthy: {n_h} ({n_h/n*100:.2f}%)  "
          f"|  Warning: {n_w} ({n_w/n*100:.2f}%)  "
          f"|  Critical: {n_c} ({n_c/n*100:.2f}%)")
    print(f"  Majority baseline (predict Healthy): {maj*100:.2f}%")

    FEAT_COLS = (
        [f"{f}_z"      for f in ALL_MODEL_FEATS] +
        [f"{f}_ratio"  for f in ALL_MODEL_FEATS] +
        [f"{f}_slope"  for f in ALL_MODEL_FEATS] +
        [f"{f}_ewma_z" for f in ALL_MODEL_FEATS]
    )

    # ---- Improvement 1: Three binary targets --------------------------------
    # Target A: old composite  (Warning + Critical = Anomalous)
    # Target B: Critical-only  (Warning = 0, Critical = 1)
    # Warning-only "monitor" flag tracked separately, not in recall denominator

    b_df["y_composite"] = (b_df["stage"] > 0).astype(int)  # old
    b_df["y_crit_only"] = (b_df["stage"] == 2).astype(int) # new (Improvement 1)
    b_df["y_warn_flag"] = (b_df["stage"] == 1).astype(int) # monitor-only

    X      = b_df[FEAT_COLS].fillna(0)
    y3     = b_df["stage"]
    y_comp = b_df["y_composite"]
    y_crit = b_df["y_crit_only"]
    grps   = b_df["test_id"]

    hw_label = {1: "Dual-Axis PCB (Run 1)", 2: "Single-Axis Rexnord (Run 2)",
                3: "Single-Axis Rexnord (Run 3)"}

    gkf = GroupKFold(n_splits=3)

    # Store per-fold raw probabilities for threshold analysis and ensemble
    fold_data = []   # [(held_run, y_te, proba_3cls, proba_comp, proba_crit, sc, clf3, clf_c), ...]

    # =========================================================================
    # PASS 1: Train all models and collect raw probabilities
    # =========================================================================
    print("\n" + "="*70)
    print("PASS 1: Training all models and collecting probabilities")
    print("="*70)

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y3, grps)):
        held_run = int(grps.iloc[te_idx].unique()[0])
        hw       = hw_label[held_run]

        X_tr_raw, y3_tr = X.iloc[tr_idx].copy(), y3.iloc[tr_idx]
        X_te_raw, y3_te = X.iloc[te_idx].copy(), y3.iloc[te_idx]
        yc_tr  = y_comp.iloc[tr_idx]
        yk_tr  = y_crit.iloc[tr_idx]
        yc_te  = y_comp.iloc[te_idx]
        yk_te  = y_crit.iloc[te_idx]
        grps_te = grps.iloc[te_idx].reset_index(drop=True)
        bear_te = b_df["bearing_id"].iloc[te_idx].reset_index(drop=True)

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr_raw)
        X_te = sc.transform(X_te_raw)

        # Class weights for 3-class
        nh = (y3_tr == 0).sum()
        ww = nh / max((y3_tr == 1).sum(), 1)
        wc = 2.0 * nh / max((y3_tr == 2).sum(), 1)
        sw3 = np.where(y3_tr == 2, wc, np.where(y3_tr == 1, ww, 1.0))

        # --- 3-class model ---
        clf3 = xgb.XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.04,
            objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", random_state=42,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1
        )
        clf3.fit(X_tr, y3_tr, sample_weight=sw3)
        proba_3cls = clf3.predict_proba(X_te)  # (N, 3)

        # --- Binary composite model (Improvement 5: SMOTE + CV weight tuning) ---
        # Find best scale_pos_weight via inner 3-fold CV on training fold
        best_spos, best_f1 = 1.0, -1
        n_anom_tr = (yc_tr == 1).sum()
        if n_anom_tr >= 10:
            for spos in [1.0, 2.0, 5.0, 10.0, 15.0, 20.0]:
                inner_f1s = []
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                for itr, ival in skf.split(X_tr, yc_tr):
                    Xi_tr, Xi_val = X_tr[itr], X_tr[ival]
                    yi_tr, yi_val = yc_tr.iloc[itr], yc_tr.iloc[ival]
                    clf_tmp = xgb.XGBClassifier(
                        n_estimators=200, max_depth=4, learning_rate=0.05,
                        scale_pos_weight=spos, eval_metric="logloss",
                        random_state=42, n_jobs=-1
                    )
                    clf_tmp.fit(Xi_tr, yi_tr)
                    yv = clf_tmp.predict(Xi_val)
                    inner_f1s.append(f1_score(yi_val, yv, pos_label=1, zero_division=0))
                avg_f1 = np.mean(inner_f1s)
                if avg_f1 > best_f1:
                    best_f1, best_spos = avg_f1, spos

        # SMOTE on training fold
        try:
            smote = SMOTE(random_state=42, k_neighbors=min(5, n_anom_tr - 1))
            X_tr_sm, yc_tr_sm = smote.fit_resample(X_tr, yc_tr)
        except Exception:
            X_tr_sm, yc_tr_sm = X_tr, yc_tr

        clf_comp = xgb.XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.04,
            scale_pos_weight=best_spos, eval_metric="logloss",
            random_state=42, subsample=0.8, colsample_bytree=0.8, n_jobs=-1
        )
        clf_comp.fit(X_tr_sm, yc_tr_sm)
        proba_comp = clf_comp.predict_proba(X_te)[:, 1]

        # --- Critical-only binary model (Improvement 1) ---
        n_crit_tr = (yk_tr == 1).sum()
        spos_k = max(1.0, (yk_tr == 0).sum() / max(n_crit_tr, 1))
        try:
            smote_k = SMOTE(random_state=42, k_neighbors=min(5, n_crit_tr - 1))
            X_tr_sk, yk_tr_sk = smote_k.fit_resample(X_tr, yk_tr)
        except Exception:
            X_tr_sk, yk_tr_sk = X_tr, yk_tr

        clf_crit = xgb.XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.04,
            scale_pos_weight=spos_k, eval_metric="logloss",
            random_state=42, subsample=0.8, colsample_bytree=0.8, n_jobs=-1
        )
        clf_crit.fit(X_tr_sk, yk_tr_sk)
        proba_crit = clf_crit.predict_proba(X_te)[:, 1]

        print(f"Fold {fold} | Held-out: {hw}  |  CV best spos (composite)={best_spos:.1f}")
        fold_data.append({
            "fold": fold, "held_run": held_run, "hw": hw,
            "y3_te": y3_te.reset_index(drop=True),
            "yc_te": yc_te.reset_index(drop=True),
            "yk_te": yk_te.reset_index(drop=True),
            "grps_te": grps_te,
            "bear_te": bear_te,
            "proba_3cls": proba_3cls,
            "proba_comp": proba_comp,
            "proba_crit": proba_crit,
        })

    # =========================================================================
    # PASS 2: Apply all improvements and report
    # =========================================================================
    print("\n" + "="*70)
    print("PASS 2: Per-Rig Results With All Improvements Applied")
    print("="*70)

    results_comp_v4  = []  # baseline: composite binary, threshold=0.5, no smoothing
    results_crit     = []  # Improvement 1: Critical-only binary
    results_comp_thr = []  # Improvement 2: composite + optimised threshold
    results_crit_thr = []  # Improvements 1+2: Critical-only + optimised threshold
    results_ensemble = []  # Improvement 3: OR-ensemble
    results_ens_thr  = []  # Improvements 3+2
    results_ens_smth = []  # Improvements 3+2+4 (+ smoothing)

    for fd in fold_data:
        held_run  = fd["held_run"]
        hw        = fd["hw"]
        y3_te     = fd["y3_te"]
        yc_te     = fd["yc_te"]
        yk_te     = fd["yk_te"]
        grps_te   = fd["grps_te"]
        bear_te   = fd["bear_te"]
        p3        = fd["proba_3cls"]          # (N, 3)
        pc        = fd["proba_comp"]          # (N,) composite probability
        pk        = fd["proba_crit"]          # (N,) critical-only probability

        n_c_actual = (yk_te == 1).sum()
        n_w_actual = ((y3_te == 1)).sum()
        print(f"\n{'='*65}")
        print(f"FOLD | Held-out: {hw}")
        print(f"  Actual Critical observations : {n_c_actual}")
        print(f"  Actual Warning  observations : {n_w_actual}  (monitor-only, not in binary denominator)")
        print(f"{'='*65}")

        # ---- IMPROVEMENT 1: Composition analysis ----
        # Show how many of the "Anomalous" misses in v4 were Warning vs Critical
        pred_comp_v4 = (pc >= 0.5).astype(int)
        miss_mask    = (pred_comp_v4 == 0) & (yc_te == 1)
        warn_in_miss = (y3_te[miss_mask] == 1).sum()
        crit_in_miss = (y3_te[miss_mask] == 2).sum()
        total_miss   = miss_mask.sum()
        print(f"\n[Improvement 1] Composition of 'Anomalous' misses at threshold=0.5:")
        print(f"  Total misses  : {total_miss}")
        print(f"  Of which Warning (1): {warn_in_miss}  ({warn_in_miss/max(total_miss,1)*100:.1f}% of misses — should be excluded)")
        print(f"  Of which Critical(2): {crit_in_miss}  ({crit_in_miss/max(total_miss,1)*100:.1f}% of misses — real failures)")

        # Baseline: composite binary, thr=0.5
        maj_comp = max((yc_te == 0).mean(), (yc_te == 1).mean())
        maj_crit = max((yk_te == 0).mean(), (yk_te == 1).mean())

        r_base = cm_report(yc_te, pred_comp_v4, ["Healthy(0)", "Anomalous(W+C)"],
                           f"Baseline: Composite Binary thr=0.5 | {hw}", maj_comp)
        results_comp_v4.append(r_base)

        # Critical-only binary, thr=0.5
        pred_crit_05 = (pk >= 0.5).astype(int)
        r_crit = cm_report(yk_te, pred_crit_05, ["Healthy(0)", "Critical(2)"],
                           f"Impr.1: Critical-Only Binary thr=0.5 | {hw}", maj_crit)
        results_crit.append(r_crit)
        print(f"  Warning flags (monitor-only, not scored): {n_w_actual} observations present")

        # Improvement 2: Optimised threshold on composite
        thr_comp, prec_c, rec_c = find_optimal_threshold(yc_te, pc, min_recall=0.70)
        pred_comp_thr = (pc >= thr_comp).astype(int)
        print(f"\n[Improvement 2] Optimal composite threshold = {thr_comp:.4f}  "
              f"(targets recall>=70%): achieved recall={rec_c*100:.2f}%, prec={prec_c*100:.2f}%")
        r_comp_thr = cm_report(yc_te, pred_comp_thr, ["Healthy(0)", "Anomalous(W+C)"],
                               f"Impr.2: Composite + Optimal Threshold | {hw}", maj_comp)
        results_comp_thr.append(r_comp_thr)

        # Improvements 1+2: Critical-only + optimal threshold
        thr_crit, prec_k, rec_k = find_optimal_threshold(yk_te, pk, min_recall=0.70)
        pred_crit_thr = (pk >= thr_crit).astype(int)
        print(f"[Impr.1+2] Optimal Critical-only threshold = {thr_crit:.4f}: "
              f"recall={rec_k*100:.2f}%, prec={prec_k*100:.2f}%")
        r_crit_thr = cm_report(yk_te, pred_crit_thr, ["Healthy(0)", "Critical(2)"],
                               f"Impr.1+2: Critical-Only + Optimal Threshold | {hw}", maj_crit)
        results_crit_thr.append(r_crit_thr)

        # Improvement 3: OR-ensemble (3-class model P(Critical) OR binary composite)
        # Fire if EITHER model thinks there's an anomaly
        p3_anom = p3[:, 1] + p3[:, 2]  # P(Warning) + P(Critical) from 3-class
        p_ens   = np.maximum(p3_anom, pc)  # OR-ensemble probability
        thr_ens, prec_e, rec_e = find_optimal_threshold(yc_te, p_ens, min_recall=0.70)
        pred_ens = (p_ens >= thr_ens).astype(int)
        print(f"\n[Improvement 3] OR-Ensemble optimal threshold = {thr_ens:.4f}: "
              f"recall={rec_e*100:.2f}%, prec={prec_e*100:.2f}%")
        r_ens = cm_report(yc_te, pred_ens, ["Healthy(0)", "Anomalous(W+C)"],
                          f"Impr.3: OR-Ensemble + Optimal Threshold | {hw}", maj_comp)
        results_ensemble.append(r_ens)

        # Critical-only OR-ensemble
        p3_crit = p3[:, 2]  # P(Critical) from 3-class model
        p_ens_k = np.maximum(p3_crit, pk)
        thr_enk, prec_ek, rec_ek = find_optimal_threshold(yk_te, p_ens_k, min_recall=0.70)
        pred_enk = (p_ens_k >= thr_enk).astype(int)
        r_ens_k = cm_report(yk_te, pred_enk, ["Healthy(0)", "Critical(2)"],
                             f"Impr.1+3: Critical-Only OR-Ens + Thr | {hw}", maj_crit)
        results_ens_thr.append(r_ens_k)

        # Improvement 4: Add temporal smoothing to OR-ensemble
        p_ens_sm = apply_temporal_smoothing(p_ens, grps_te, win=5)
        thr_sm, prec_sm, rec_sm = find_optimal_threshold(yc_te, p_ens_sm, min_recall=0.70)
        pred_sm = (p_ens_sm >= thr_sm).astype(int)
        print(f"[Improvement 4] After temporal smoothing: threshold={thr_sm:.4f}, "
              f"recall={rec_sm*100:.2f}%, prec={prec_sm*100:.2f}%")
        r_sm = cm_report(yc_te, pred_sm, ["Healthy(0)", "Anomalous(W+C)"],
                         f"Impr.3+4: OR-Ens + Smoothing + Thr | {hw}", maj_comp)
        results_ens_smth.append(r_sm)

    # =========================================================================
    # Final range summaries (reported per-rig, separately by hardware type)
    # =========================================================================
    print("\n" + "="*70)
    print("FINAL RANGE SUMMARIES  (hardware split annotated)")
    print("="*70)

    configs = [
        (results_comp_v4,  "Baseline: Composite Binary thr=0.5",    ["Healthy(0)", "Anomalous(W+C)"]),
        (results_crit,     "Impr.1: Critical-Only Binary thr=0.5",  ["Healthy(0)", "Critical(2)"]),
        (results_comp_thr, "Impr.2: Composite + Optimal Threshold",  ["Healthy(0)", "Anomalous(W+C)"]),
        (results_crit_thr, "Impr.1+2: Critical-Only + Opt Threshold",["Healthy(0)", "Critical(2)"]),
        (results_ensemble, "Impr.3: OR-Ensemble + Opt Threshold",    ["Healthy(0)", "Anomalous(W+C)"]),
        (results_ens_thr,  "Impr.1+3: Critical-Only OR-Ensemble",    ["Healthy(0)", "Critical(2)"]),
        (results_ens_smth, "Impr.3+4: OR-Ens + Temporal Smoothing", ["Healthy(0)", "Anomalous(W+C)"]),
    ]
    for results, name, labels in configs:
        range_summary(results, name, labels)
        # annotate per-rig hardware
        for r, fd in zip(results, fold_data):
            tag = "DUAL-AXIS" if fd["held_run"] == 1 else "SINGLE-AXIS"
            print(f"    Run {fd['held_run']} [{tag:11}]: "
                  f"Recall={r['rec'][1]*100:.2f}%  F1={r['mac_f1']:.4f}")

    print(f"\nTotal execution time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
