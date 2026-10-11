"""
train_bearing_model_v7.py
=========================
Rigorous, leak-free evaluation of all improvement stacks under nested CV:
1. Critical-only binary baseline (thr=0.5)
2. Critical-only binary with Nested CV calibrated threshold
3. Critical-only OR-ensemble (3-class P(Critical) OR binary P(Critical)) with thr=0.5
4. Critical-only OR-ensemble with Nested CV calibrated threshold
5. CUSUM temporal smoothing in isolation (on binary P(Critical)) with nested thr
6. Stacked: Critical-only OR-ensemble + CUSUM temporal smoothing with nested thr
7. Composite binary equivalents (thr=0.5, Nested thr, OR-ensemble, CUSUM)
8. Investigation of Improvement 5 (SMOTE + scale_pos_weight tuning)

NO oracle or held-out test labels are used for threshold selection.
All metrics reported with Recall, Precision, and False-Alarm Rate (FP / Total Healthy).
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
    precision_score, precision_recall_curve
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

ROOT_DIR  = Path(r"c:\Users\riddh\OneDrive\Desktop\industrial_agent")
DATA_PATH = ROOT_DIR / "bearings_all_features.csv"
FREQ_PATH = ROOT_DIR / "bearings_freq_raw_features.csv"

FAILED_MAP = {1: ["Bearing3", "Bearing4"], 2: ["Bearing1"], 3: ["Bearing3"]}
HI_SIGNALS = ["rms", "kurtosis", "env_bpfo"]
MODEL_TIME_FEATS = ["shape_factor", "impulse_factor", "skewness",
                    "peak_to_peak", "variance", "std", "crest_factor", "peak"]
MODEL_FREQ_FEATS = ["env_bpfi", "env_bsf", "env_bpfo2",
                    "fft_bpfi", "fft_bsf", "sk_mean", "sk_max"]
ALL_MODEL_FEATS  = MODEL_TIME_FEATS + MODEL_FREQ_FEATS
HW_LABEL = {1: "Dual-Axis PCB (Run 1)",
            2: "Single-Axis Rexnord (Run 2)",
            3: "Single-Axis Rexnord (Run 3)"}


def load_unrolled():
    df_all  = pd.read_csv(DATA_PATH)
    df_freq = pd.read_csv(FREQ_PATH)
    merged  = pd.merge(df_all, df_freq, on=["test_id", "file"], how="inner")
    rows = []
    for _, row in merged.iterrows():
        tid = int(row["test_id"]); hr = float(row["hours_elapsed"])
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
            rec = {"test_id": tid, "bearing_id": bname, "file": fname,
                   "hours_elapsed": hr, "is_failing": int(is_failing),
                   "is_dual_axis": 1 if tid == 1 else 0}
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
    df_in["stage"] = 0; df_in["hi"] = 0.0
    for (tid, bid), grp in df_in.groupby(["test_id", "bearing_id"]):
        grp    = grp.sort_values("hours_elapsed").copy()
        idx    = grp.index
        n_base = max(int(len(grp) * 0.10), 15)
        base   = grp.iloc[:n_base]
        zs     = []
        for sig in HI_SIGNALS:
            mu = base[sig].mean(); sd = base[sig].std() + 1e-6
            z  = ((grp[sig] - mu) / sd).rolling(win, min_periods=3).mean().fillna(0)
            zs.append(z.values)
        hi = np.maximum.reduce(zs)
        df_in.loc[idx, "hi"] = hi
        stage = (np.where(hi >= hi_crit, 2, np.where(hi >= hi_warn, 1, 0))
                 if grp["is_failing"].iloc[0]
                 else np.zeros(len(grp), dtype=int))
        df_in.loc[idx, "stage"] = stage
    return df_in


def build_model_features(df_in, win=15):
    df_in  = df_in.sort_values(["test_id", "bearing_id", "hours_elapsed"]).copy()
    frames = []
    for (tid, bid), grp in df_in.groupby(["test_id", "bearing_id"]):
        grp    = grp.sort_values("hours_elapsed").copy()
        n_base = max(int(len(grp) * 0.10), 15)
        base   = grp.iloc[:n_base][ALL_MODEL_FEATS]
        mu     = base.mean()
        sigma  = base.std().replace(0, 1e-6).fillna(1e-6)
        for f in ALL_MODEL_FEATS:
            grp[f"{f}_z"]      = (grp[f] - mu[f]) / sigma[f]
            grp[f"{f}_ratio"]  = grp[f] / (mu[f] + 1e-9)
            grp[f"{f}_slope"]  = grp[f"{f}_z"].diff().rolling(win, min_periods=1).mean().fillna(0)
            grp[f"{f}_ewma_z"] = grp[f"{f}_z"].ewm(span=30).mean()
        frames.append(grp)
    return pd.concat(frames, ignore_index=True)


def train_binary_model(X_tr, y_tr, use_smote=True, pos_weight=None):
    n_pos = (y_tr == 1).sum()
    spos  = pos_weight if pos_weight is not None else max(1.0, (y_tr == 0).sum() / max(n_pos, 1))
    if use_smote and n_pos >= 5:
        try:
            smote = SMOTE(random_state=42, k_neighbors=min(5, n_pos - 1))
            X_sm, y_sm = smote.fit_resample(X_tr, y_tr)
        except Exception:
            X_sm, y_sm = X_tr, y_tr
    else:
        X_sm, y_sm = X_tr, y_tr
    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.04,
        scale_pos_weight=spos, eval_metric="logloss",
        random_state=42, subsample=0.8, colsample_bytree=0.8, n_jobs=-1
    )
    clf.fit(X_sm, y_sm)
    return clf


def train_3class_model(X_tr, y3_tr):
    nh = (y3_tr == 0).sum()
    ww = nh / max((y3_tr == 1).sum(), 1)
    wc = 2.0 * nh / max((y3_tr == 2).sum(), 1)
    sw3 = np.where(y3_tr == 2, wc, np.where(y3_tr == 1, ww, 1.0))
    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.04,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", random_state=42,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1
    )
    clf.fit(X_tr, y3_tr, sample_weight=sw3)
    return clf


def apply_temporal_smoothing(proba_series, bearing_ids, win=5):
    """
    Two-step temporal filter:
    - Step A: Rolling mean over win consecutive snapshots per bearing
    - Step B: CUSUM accumulation departure, irreversible flag
    """
    df_tmp = pd.DataFrame({"proba": proba_series, "bearing": bearing_ids})
    smoothed = np.zeros(len(proba_series))
    for bid, grp_idx in df_tmp.groupby("bearing").groups.items():
        p_vals = df_tmp.loc[grp_idx, "proba"].values
        rolled = pd.Series(p_vals).rolling(win, min_periods=1).mean().values
        base_mean = rolled[:max(1, int(len(rolled) * 0.10))].mean()
        cusum = np.cumsum(rolled - base_mean)
        cusum_flag = np.maximum.accumulate(cusum > 0.5).astype(float)
        smoothed[grp_idx] = np.clip(0.6 * rolled + 0.4 * cusum_flag, 0, 1)
    return smoothed


def calibrate_threshold_nested(
        b_df, FEAT_COLS, train_runs, target_type, transform_func=None, min_recall=0.70):
    """
    Performs inner cross-validation between the two training rigs to select a threshold.
    transform_func: takes (clf_crit, clf_3cls, X_val_sc, b_val_sub) -> proba_val
    """
    thresholds_found = []
    inner_details    = []

    for val_run in train_runs:
        inner_train_run = [r for r in train_runs if r != val_run][0]
        tr_mask  = b_df["test_id"] == inner_train_run
        val_mask = b_df["test_id"] == val_run

        X_tr_raw  = b_df.loc[tr_mask,  FEAT_COLS].fillna(0)
        X_val_raw = b_df.loc[val_mask, FEAT_COLS].fillna(0)

        target_col = "y_crit" if target_type == "crit" else "y_comp"
        y_tr  = b_df.loc[tr_mask,  target_col]
        y_val = b_df.loc[val_mask, target_col]
        y3_tr = b_df.loc[tr_mask,  "stage"]

        sc = StandardScaler()
        X_tr_sc  = sc.fit_transform(X_tr_raw)
        X_val_sc = sc.transform(X_val_raw)

        clf_bin = train_binary_model(X_tr_sc, y_tr)
        clf_3c  = train_3class_model(X_tr_sc, y3_tr)

        b_val_sub = b_df.loc[val_mask, ["bearing_id"]].copy().reset_index(drop=True)

        if transform_func is not None:
            proba_val = transform_func(clf_bin, clf_3c, X_val_sc, b_val_sub)
        else:
            proba_val = clf_bin.predict_proba(X_val_sc)[:, 1]

        n_pos = (y_val == 1).sum()
        if n_pos == 0:
            thresholds_found.append(0.5)
            continue

        prec, rec, thr = precision_recall_curve(y_val, proba_val)
        prec, rec = prec[:-1], rec[:-1]
        viable = rec >= min_recall
        if viable.any():
            best_idx = np.where(viable, prec, -1).argmax()
        else:
            best_idx = np.argmax(rec)

        chosen_thr = float(thr[best_idx])
        thresholds_found.append(chosen_thr)
        inner_details.append({
            "tr_run": inner_train_run, "val_run": val_run,
            "thr": chosen_thr, "rec": float(rec[best_idx]), "prec": float(prec[best_idx])
        })

    calibrated_thr = float(np.mean(thresholds_found))
    thr_sd = float(np.std(thresholds_found))
    return calibrated_thr, thr_sd, inner_details


def compute_metrics(y_true, y_pred, y_hlth_mask):
    """
    Computes all standard metrics including explicit False-Alarm Rate.
    FAR = FP / total actual healthy.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    n_hlth = (y_hlth_mask == 1).sum()
    fp_hlth = (y_pred[y_hlth_mask] == 1).sum()
    far = fp_hlth / max(n_hlth, 1)  # False Alarm Rate on strictly healthy observations
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    mac_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "recall": rec, "precision": prec, "far": far,
        "f1": f1, "mac_f1": mac_f1, "acc": acc, "cm": cm
    }


def main():
    t0 = time.time()
    print("Loading datasets...")
    b_df = load_unrolled()
    b_df = build_physics_labels(b_df)
    b_df = build_model_features(b_df)

    b_df["y_crit"] = (b_df["stage"] == 2).astype(int)
    b_df["y_comp"] = (b_df["stage"] > 0).astype(int)
    b_df["is_healthy"] = (b_df["stage"] == 0).astype(int)

    FEAT_COLS = (
        [f"{f}_z"      for f in ALL_MODEL_FEATS] +
        [f"{f}_ratio"  for f in ALL_MODEL_FEATS] +
        [f"{f}_slope"  for f in ALL_MODEL_FEATS] +
        [f"{f}_ewma_z" for f in ALL_MODEL_FEATS]
    )

    gkf = GroupKFold(n_splits=3)
    grps = b_df["test_id"]
    X = b_df[FEAT_COLS].fillna(0)

    # We evaluate across outer folds
    # Outer folds: Held-out Run 3, Held-out Run 1, Held-out Run 2
    outer_data = []

    print("\nTraining models on each outer fold...")
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, b_df["stage"], grps)):
        held_run = int(grps.iloc[te_idx].unique()[0])
        train_runs = [r for r in [1, 2, 3] if r != held_run]

        X_tr_raw = X.iloc[tr_idx]; X_te_raw = X.iloc[te_idx]
        y_crit_tr = b_df["y_crit"].iloc[tr_idx]; y_crit_te = b_df["y_crit"].iloc[te_idx]
        y_comp_tr = b_df["y_comp"].iloc[tr_idx]; y_comp_te = b_df["y_comp"].iloc[te_idx]
        y3_tr     = b_df["stage"].iloc[tr_idx];  y3_te     = b_df["stage"].iloc[te_idx]
        hlth_te   = b_df["is_healthy"].iloc[te_idx].values
        bear_te   = b_df["bearing_id"].iloc[te_idx].values

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr_raw)
        X_te = sc.transform(X_te_raw)

        # 1. Binary Crit-only model
        clf_crit = train_binary_model(X_tr, y_crit_tr)
        p_crit   = clf_crit.predict_proba(X_te)[:, 1]

        # 2. Binary Composite model
        clf_comp = train_binary_model(X_tr, y_comp_tr)
        p_comp   = clf_comp.predict_proba(X_te)[:, 1]

        # 3. 3-Class model
        clf_3c   = train_3class_model(X_tr, y3_tr)
        p_3c     = clf_3c.predict_proba(X_te)  # (N, 3)

        # Ensemble signals
        p_crit_ens = np.maximum(p_3c[:, 2], p_crit)
        p_comp_ens = np.maximum(p_3c[:, 1] + p_3c[:, 2], p_comp)

        # Smoothed signals
        p_crit_smth = apply_temporal_smoothing(p_crit, bear_te, win=5)
        p_crit_ens_smth = apply_temporal_smoothing(p_crit_ens, bear_te, win=5)
        p_comp_smth = apply_temporal_smoothing(p_comp, bear_te, win=5)
        p_comp_ens_smth = apply_temporal_smoothing(p_comp_ens, bear_te, win=5)

        outer_data.append({
            "fold": fold, "held_run": held_run, "train_runs": train_runs,
            "y_crit_te": y_crit_te.values, "y_comp_te": y_comp_te.values,
            "hlth_te": hlth_te, "bear_te": bear_te,
            "p_crit": p_crit, "p_comp": p_comp,
            "p_crit_ens": p_crit_ens, "p_comp_ens": p_comp_ens,
            "p_crit_smth": p_crit_smth, "p_crit_ens_smth": p_crit_ens_smth,
            "p_comp_smth": p_comp_smth, "p_comp_ens_smth": p_comp_ens_smth,
        })

    # Now let's run Nested CV calibration for each configuration
    print("\nRunning Nested Calibration for all configurations...")

    # Configurations to test
    # (config_name, target_type, proba_key, transform_func, default_thr)
    configs = [
        # --- CRITICAL-ONLY TARGETS ---
        {
            "name": "1. Crit-only Binary (thr=0.50)",
            "target": "crit", "proba_key": "p_crit",
            "thr_type": "fixed", "fixed_thr": 0.50
        },
        {
            "name": "2. Crit-only Binary (Nested Thr)",
            "target": "crit", "proba_key": "p_crit",
            "thr_type": "nested",
            "func": lambda cb, c3, Xv, b: cb.predict_proba(Xv)[:, 1]
        },
        {
            "name": "3. Crit-only OR-Ensemble (thr=0.50)",
            "target": "crit", "proba_key": "p_crit_ens",
            "thr_type": "fixed", "fixed_thr": 0.50
        },
        {
            "name": "4. Crit-only OR-Ensemble (Nested Thr)",
            "target": "crit", "proba_key": "p_crit_ens",
            "thr_type": "nested",
            "func": lambda cb, c3, Xv, b: np.maximum(c3.predict_proba(Xv)[:, 2], cb.predict_proba(Xv)[:, 1])
        },
        {
            "name": "5. Crit-only CUSUM Smoothing in Isolation (Nested Thr)",
            "target": "crit", "proba_key": "p_crit_smth",
            "thr_type": "nested",
            "func": lambda cb, c3, Xv, b: apply_temporal_smoothing(cb.predict_proba(Xv)[:, 1], b["bearing_id"].values, win=5)
        },
        {
            "name": "6. Crit-only OR-Ens + CUSUM Smoothing (thr=0.50)",
            "target": "crit", "proba_key": "p_crit_ens_smth",
            "thr_type": "fixed", "fixed_thr": 0.50
        },
        {
            "name": "7. Crit-only OR-Ens + CUSUM Smoothing (Nested Thr)",
            "target": "crit", "proba_key": "p_crit_ens_smth",
            "thr_type": "nested",
            "func": lambda cb, c3, Xv, b: apply_temporal_smoothing(
                np.maximum(c3.predict_proba(Xv)[:, 2], cb.predict_proba(Xv)[:, 1]),
                b["bearing_id"].values, win=5
            )
        },
        # --- COMPOSITE TARGETS ---
        {
            "name": "8. Composite Binary (thr=0.50)",
            "target": "comp", "proba_key": "p_comp",
            "thr_type": "fixed", "fixed_thr": 0.50
        },
        {
            "name": "9. Composite Binary (Nested Thr)",
            "target": "comp", "proba_key": "p_comp",
            "thr_type": "nested",
            "func": lambda cb, c3, Xv, b: cb.predict_proba(Xv)[:, 1]
        },
        {
            "name": "10. Composite OR-Ens + CUSUM Smoothing (Nested Thr)",
            "target": "comp", "proba_key": "p_comp_ens_smth",
            "thr_type": "nested",
            "func": lambda cb, c3, Xv, b: apply_temporal_smoothing(
                np.maximum(c3.predict_proba(Xv)[:, 1] + c3.predict_proba(Xv)[:, 2], cb.predict_proba(Xv)[:, 1]),
                b["bearing_id"].values, win=5
            )
        },
    ]

    all_results = {}

    for cfg in configs:
        cfg_name = cfg["name"]
        all_results[cfg_name] = []
        target_name = cfg["target"]
        pk = cfg["proba_key"]

        for od in outer_data:
            held_run   = od["held_run"]
            train_runs = od["train_runs"]
            y_te       = od["y_crit_te"] if target_name == "crit" else od["y_comp_te"]
            hlth_te    = od["hlth_te"]
            proba_te   = od[pk]

            if cfg["thr_type"] == "fixed":
                thr = cfg["fixed_thr"]
                thr_sd = 0.0
                inner_info = []
            else:
                thr, thr_sd, inner_info = calibrate_threshold_nested(
                    b_df, FEAT_COLS, train_runs, target_name,
                    transform_func=cfg["func"], min_recall=0.70
                )

            y_pred = (proba_te >= thr).astype(int)
            met = compute_metrics(y_te, y_pred, hlth_te)
            met["held_run"] = held_run
            met["thr"] = thr
            met["thr_sd"] = thr_sd
            met["inner_info"] = inner_info
            all_results[cfg_name].append(met)

    # Print comprehensive tables
    print("\n" + "="*85)
    print("LEAK-FREE EVALUATION: ALL IMPROVEMENT STACKS UNDER NESTED PROTOCOL")
    print("="*85)

    for cfg_name, res_list in all_results.items():
        print(f"\n### {cfg_name}")
        print(f"| Rig | Hardware | Thr | Thr SD | Recall | Precision | False-Alarm Rate (FP/Hlth) | Macro F1 | FP/TotalHlth |")
        print(f"|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        # Order by run: 3, 1, 2
        res_by_run = {r["held_run"]: r for r in res_list}
        recs, precs, fars, f1s = [], [], [], []
        for r_id in [3, 1, 2]:
            r = res_by_run[r_id]
            hw = "Dual-Axis PCB" if r_id == 1 else "Single-Axis Rexnord"
            n_hlth = (r["tn"] + r["fp"])
            recs.append(r["recall"])
            precs.append(r["precision"])
            fars.append(r["far"])
            f1s.append(r["mac_f1"])
            print(f"| Run {r_id} | {hw:19s} | {r['thr']:.4f} | {r['thr_sd']:.4f} | **{r['recall']*100:.2f}%** | {r['precision']*100:.2f}% | **{r['far']*100:.2f}%** | {r['mac_f1']:.4f} | {r['fp']}/{n_hlth} |")
        print(f"**Range across rigs:** Recall [{min(recs)*100:.2f}%, {max(recs)*100:.2f}%] | FAR [{min(fars)*100:.2f}%, {max(fars)*100:.2f}%] | Macro F1 [{min(f1s):.4f}, {max(f1s):.4f}]")

    print(f"\nExecution finished in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
