"""
train_bearing_clean_correlation.py
==================================
Correlation-based exclusion audit and re-run of the nested-CV OR-ensemble.
Replaces literal name-based exclusion with empirical correlation screening against
all labeling signals (rms, kurtosis, env_bpfo, and the composite HI).
"""

import sys, time, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score

warnings.filterwarnings("ignore")

from train_bearing_model_v6 import (
    load_unrolled, build_physics_labels, build_model_features,
    ALL_MODEL_FEATS, HI_SIGNALS, HW_LABEL
)

CORR_THRESHOLD = 0.90  # Strict threshold: >=0.90 correlation excluded

def main():
    t0 = time.time()
    print("Loading unrolled dataset...")
    b_df = load_unrolled()
    b_df = build_physics_labels(b_df)
    b_df = build_model_features(b_df)

    b_df["y_crit"] = (b_df["stage"] == 2).astype(int)
    b_df["is_healthy"] = (b_df["stage"] == 0).astype(int)

    all_candidate_feats = (
        [f"{f}_z"      for f in ALL_MODEL_FEATS] +
        [f"{f}_ratio"  for f in ALL_MODEL_FEATS] +
        [f"{f}_slope"  for f in ALL_MODEL_FEATS] +
        [f"{f}_ewma_z" for f in ALL_MODEL_FEATS]
    )

    print(f"\nTotal candidate features before correlation audit: {len(all_candidate_feats)}")
    print(f"Checking correlation against HI signals: {HI_SIGNALS} and composite 'hi'...")

    # Correlation check
    corr_records = []
    excluded_feats = set()

    signals_to_check = HI_SIGNALS + ["hi"]

    for feat in all_candidate_feats:
        max_corr = 0.0
        max_sig = None
        for sig in signals_to_check:
            s1 = b_df[feat].fillna(0)
            s2 = b_df[sig].fillna(0)
            c = abs(s1.corr(s2))
            if c > max_corr:
                max_corr = c
                max_sig = sig
        corr_records.append({"feature": feat, "max_corr": max_corr, "worst_against": max_sig})
        if max_corr >= CORR_THRESHOLD:
            excluded_feats.add(feat)

    df_corr = pd.DataFrame(corr_records).sort_values("max_corr", ascending=False)
    print("\n=== Top 20 Highest Correlated Features with Labeling Signals ===")
    print(df_corr.head(20).to_string(index=False))

    # Also exclude base features and all their children if base >= CORR_THRESHOLD
    for base in ALL_MODEL_FEATS:
        base_s = b_df[base].fillna(0)
        for sig in signals_to_check:
            c = abs(base_s.corr(b_df[sig].fillna(0)))
            if c >= CORR_THRESHOLD:
                print(f"\n[BASE EXCLUSION] Base feature '{base}' has corr={c:.4f} with '{sig}'! Purging all its derived variants...")
                for suffix in ["_z", "_ratio", "_slope", "_ewma_z"]:
                    excluded_feats.add(f"{base}{suffix}")

    print(f"\n=== Features Excluded (|corr| >= {CORR_THRESHOLD}) ===")
    for f in sorted(excluded_feats):
        c_val = df_corr.loc[df_corr["feature"] == f, "max_corr"].values
        sig_val = df_corr.loc[df_corr["feature"] == f, "worst_against"].values
        c_str = f"{c_val[0]:.4f}" if len(c_val) else "base"
        s_str = f"{sig_val[0]}" if len(sig_val) else "base"
        print(f"  {f:25s}: corr = {c_str} against {s_str}")

    clean_features = [f for f in all_candidate_feats if f not in excluded_feats]
    print(f"\nClean feature count: {len(clean_features)} / {len(all_candidate_feats)} survive")
    assert len(clean_features) >= 5, "Too few features survived!"

    print("\nSurviving Clean Feature Groups:")
    surviving_bases = set(f.split("_")[0] for f in clean_features)
    print(f"  Surviving base feature groups: {sorted(surviving_bases)}")

    # =========================================================================
    # RE-EVALUATION: Nested CV OR-Ensemble on CLEAN features
    # =========================================================================
    print("\n" + "="*80)
    print("RE-RUN: NESTED-CV OR-ENSEMBLE ON CLEAN (CORRELATION-FILTERED) FEATURES")
    print("="*80)

    gkf_rigs = [3, 1, 2]
    groups = b_df["test_id"]
    X = b_df[clean_features].fillna(0)
    y_binary = b_df["y_crit"]
    y_multi = b_df["stage"]

    def nested_threshold_clean(X_data, y_data, train_rigs, min_recall=0.70):
        thresholds = []
        for i in range(2):
            inner_train = train_rigs[i]
            inner_val = train_rigs[1 - i]
            tr_idx = groups[groups == inner_train].index
            va_idx = groups[groups == inner_val].index
            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_data.loc[tr_idx])
            X_va_sc = sc.transform(X_data.loc[va_idx])
            m = xgb.XGBClassifier(n_estimators=300, max_depth=4, eval_metric="logloss", random_state=42, n_jobs=-1)
            m.fit(X_tr_sc, y_data.loc[tr_idx])
            proba = m.predict_proba(X_va_sc)[:, 1]
            prec, rec, thr = precision_recall_curve(y_data.loc[va_idx], proba)
            prec, rec = prec[:-1], rec[:-1]
            valid = np.where(rec >= min_recall)[0]
            if len(valid):
                bi = np.where(rec >= min_recall, prec, -1).argmax()
                thresholds.append(float(thr[bi]))
            else:
                thresholds.append(float(thr[np.argmax(rec)]))
        return float(np.mean(thresholds)), float(np.std(thresholds))

    results_table = []

    for held_out in gkf_rigs:
        hw = HW_LABEL[held_out]
        train_rigs = [r for r in [1, 2, 3] if r != held_out]
        tr_idx = groups[groups.isin(train_rigs)].index
        te_idx = groups[groups == held_out].index

        sc = StandardScaler()
        X_tr = sc.fit_transform(X.loc[tr_idx])
        X_te = sc.transform(X.loc[te_idx])

        # Binary Critical-Only Model
        binary_model = xgb.XGBClassifier(n_estimators=300, max_depth=4, eval_metric="logloss", random_state=42, n_jobs=-1)
        binary_model.fit(X_tr, y_binary.loc[tr_idx])

        # 3-Class Model
        multi_model = xgb.XGBClassifier(n_estimators=300, max_depth=5, objective="multi:softprob",
                                         num_class=3, eval_metric="mlogloss", random_state=42, n_jobs=-1)
        multi_model.fit(X_tr, y_multi.loc[tr_idx])

        # Nested threshold calculation on training rigs only
        thr_nested, thr_sd = nested_threshold_clean(X, y_binary, train_rigs, min_recall=0.70)

        p_binary = binary_model.predict_proba(X_te)[:, 1]
        p_multi_crit = multi_model.predict_proba(X_te)[:, 2]
        p_ensemble = np.maximum(p_binary, p_multi_crit)

        # 1. Evaluate at default thr = 0.50
        for name, p_eval, thr_val in [
            ("Binary (thr=0.50)", p_binary, 0.50),
            ("OR-Ensemble (thr=0.50)", p_ensemble, 0.50),
            ("OR-Ensemble (Nested Thr)", p_ensemble, thr_nested),
        ]:
            preds = (p_eval >= thr_val).astype(int)
            y_true = y_binary.loc[te_idx]
            hlth_mask = (b_df.loc[te_idx, "stage"] == 0)

            tp = int(((preds == 1) & (y_true == 1)).sum())
            fp = int(((preds == 1) & (y_true == 0)).sum())
            fn = int(((preds == 0) & (y_true == 1)).sum())
            tn = int(((preds == 0) & (y_true == 0)).sum())

            rec = tp / max(tp + fn, 1)
            prec = tp / max(tp + fp, 1)
            far = fp / max(fp + tn, 1)
            f1 = f1_score(y_true, preds, average="macro", zero_division=0)

            results_table.append({
                "model": name, "rig": held_out, "hw": hw,
                "thr": thr_val, "thr_sd": thr_sd if "Nested" in name else 0.0,
                "recall": rec, "precision": prec, "far": far, "f1": f1,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn
            })

    df_res = pd.DataFrame(results_table)

    print("\n" + "="*85)
    print("RESULTS ON CLEAN SURVIVING FEATURES (AFTER CORRELATION EXCLUSION)")
    print("="*85)

    for m_name in ["Binary (thr=0.50)", "OR-Ensemble (thr=0.50)", "OR-Ensemble (Nested Thr)"]:
        print(f"\n### {m_name}")
        print(f"| Rig | Hardware | Threshold | Recall | Precision | False-Alarm Rate | FP / Total Healthy | Macro F1 |")
        print(f"|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
        sub = df_res[df_res["model"] == m_name]
        for _, row in sub.iterrows():
            total_h = row["fp"] + row["tn"]
            print(f"| Run {row['rig']} | {row['hw']:20s} | {row['thr']:.4f} | **{row['recall']*100:.2f}%** | {row['precision']*100:.2f}% | **{row['far']*100:.2f}%** | {row['fp']:5d} / {total_h:5d} | {row['f1']:.4f} |")
        print(f"**Recall Range:** [{sub['recall'].min()*100:.2f}%, {sub['recall'].max()*100:.2f}%] | **FAR Range:** [{sub['far'].min()*100:.2f}%, {sub['far'].max()*100:.2f}%]")

    print(f"\nExecution finished in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
