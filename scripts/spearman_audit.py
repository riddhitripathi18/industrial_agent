"""
spearman_audit.py — Three-part audit:
1. Spearman vs Pearson on suspects (variance, mean_abs, peak_to_peak)
2. Full Spearman-based exclusion + side-by-side thr=0.50 vs nested
3. Feature importances on the clean model
"""
import sys, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, f1_score
from train_bearing_model_v6 import (
    load_unrolled, build_physics_labels, build_model_features,
    ALL_MODEL_FEATS, HI_SIGNALS, HW_LABEL
)

t0 = time.time()
b_df = load_unrolled()
b_df = build_physics_labels(b_df)
b_df = build_model_features(b_df)
b_df['y_crit'] = (b_df['stage'] == 2).astype(int)

ALL_CAND = (
    [f'{f}_z'      for f in ALL_MODEL_FEATS] +
    [f'{f}_ratio'  for f in ALL_MODEL_FEATS] +
    [f'{f}_slope'  for f in ALL_MODEL_FEATS] +
    [f'{f}_ewma_z' for f in ALL_MODEL_FEATS]
)
signals_to_check = [s for s in HI_SIGNALS + ['hi'] if s in b_df.columns]

# ── CHECK 1: Spearman vs Pearson on known suspects ──────────────────────────
print('='*70)
print('CHECK 1: SPEARMAN vs PEARSON — variance / mean_abs / peak_to_peak family')
print('='*70)
suspects = ['variance','variance_z','variance_ewma_z','variance_ratio','variance_slope',
            'mean_abs','peak_to_peak','peak_to_peak_z','peak_to_peak_ewma_z','peak_to_peak_ratio']
for feat in suspects:
    if feat not in b_df.columns:
        print(f'  {feat:30s}: NOT IN DATAFRAME (derived — see below)')
        continue
    worst_p, worst_s, worst_sig_p, worst_sig_s = 0.0, 0.0, '', ''
    for sig in signals_to_check:
        for tid in [None, 1, 2, 3]:
            sub = b_df if tid is None else b_df[b_df['test_id']==tid]
            if len(sub) == 0:
                continue
            p = abs(sub[feat].fillna(0).corr(sub[sig].fillna(0), method='pearson'))
            s = abs(sub[feat].fillna(0).corr(sub[sig].fillna(0), method='spearman'))
            if p > worst_p: worst_p, worst_sig_p = p, f'{sig}({"pool" if tid is None else f"R{tid}"})'
            if s > worst_s: worst_s, worst_sig_s = s, f'{sig}({"pool" if tid is None else f"R{tid}"})'
    verdict = 'NEWLY EXCLUDED' if worst_s >= 0.85 and worst_p < 0.85 else ('EXCLUDED BOTH' if worst_s >= 0.85 else 'CLEAN')
    print(f'  {feat:30s}  maxPearson={worst_p:.4f}({worst_sig_p})  maxSpearman={worst_s:.4f}({worst_sig_s})  -> {verdict}')

# ── Build SPEARMAN exclusion set ─────────────────────────────────────────────
print()
print('='*70)
print('FULL SPEARMAN EXCLUSION (thr=0.85, pooled + per-rig)')
print('='*70)
excl = set()
excl_log = {}

for base in ALL_MODEL_FEATS:
    for tid in [None, 1, 2, 3]:
        sub = b_df if tid is None else b_df[b_df['test_id']==tid]
        for sig in signals_to_check:
            c = abs(sub[base].fillna(0).corr(sub[sig].fillna(0), method='spearman'))
            if c >= 0.85:
                for sfx in ['_z','_ratio','_slope','_ewma_z']:
                    key = f'{base}{sfx}'
                    if key not in excl:
                        excl.add(key)
                        excl_log[key] = (c, sig, f'base_R{tid}' if tid else 'base_pool')

for feat in ALL_CAND:
    if feat in excl: continue
    for tid in [None, 1, 2, 3]:
        sub = b_df if tid is None else b_df[b_df['test_id']==tid]
        for sig in signals_to_check:
            c = abs(sub[feat].fillna(0).corr(sub[sig].fillna(0), method='spearman'))
            if c >= 0.85:
                if feat not in excl:
                    excl.add(feat)
                    excl_log[feat] = (c, sig, f'derived_R{tid}' if tid else 'derived_pool')

clean_feats = [f for f in ALL_CAND if f not in excl]
print(f'Excluded (Spearman>=0.85): {len(excl)} | Surviving: {len(clean_feats)}/60')
print()
print('All excluded features:')
for f in sorted(excl):
    c, sig, ctx = excl_log.get(f, (0,'?','?'))
    print(f'  {f:30s}: maxS={c:.4f} vs {sig} ({ctx})')

print()
surviving_bases = sorted(set(f.rsplit('_z',1)[0].rsplit('_ratio',1)[0].rsplit('_slope',1)[0].rsplit('_ewma_z',1)[0] for f in clean_feats))
print(f'Surviving feature families: {surviving_bases}')

# ── CHECK 3: Feature importances ─────────────────────────────────────────────
print()
print('='*70)
print('CHECK 3: FEATURE IMPORTANCES (Spearman-clean features; Fold: held-out Run 3)')
print('='*70)
groups = b_df['test_id']
X = b_df[clean_feats].fillna(0)
y_crit = b_df['y_crit']
y3 = b_df['stage']

tr_idx = groups[groups.isin([1,2])].index
te_idx = groups[groups==3].index
sc = StandardScaler()
X_tr = sc.fit_transform(X.loc[tr_idx])
X_te = sc.transform(X.loc[te_idx])

clf_bin = xgb.XGBClassifier(n_estimators=300, max_depth=4, eval_metric='logloss', random_state=42, n_jobs=-1)
clf_bin.fit(X_tr, y_crit.loc[tr_idx])
clf_mul = xgb.XGBClassifier(n_estimators=300, max_depth=5, objective='multi:softprob', num_class=3, eval_metric='mlogloss', random_state=42, n_jobs=-1)
clf_mul.fit(X_tr, y3.loc[tr_idx])

imp_b = pd.Series(clf_bin.feature_importances_, index=clean_feats).sort_values(ascending=False)
imp_m = pd.Series(clf_mul.feature_importances_, index=clean_feats).sort_values(ascending=False)

print()
print('Binary Critical detector — Top 15 (gain importance):')
print(f'  {"Feature":30s}  Importance   Cumulative%')
cum = 0
for feat, imp in imp_b.head(15).items():
    cum += imp
    flag = '  <<< >40% SINGLE FEATURE' if imp > 0.40 else ''
    print(f'  {feat:30s}  {imp:.5f}      {cum*100:.1f}%{flag}')

print()
print('3-class model — Top 15 (gain importance):')
print(f'  {"Feature":30s}  Importance   Cumulative%')
cum = 0
for feat, imp in imp_m.head(15).items():
    cum += imp
    flag = '  <<< >40% SINGLE FEATURE' if imp > 0.40 else ''
    print(f'  {feat:30s}  {imp:.5f}      {cum*100:.1f}%{flag}')

# Spearman of top-5 binary features vs labeling signals
print()
print('Spearman of top-5 binary model features vs ALL labeling signals:')
for feat in imp_b.head(5).index:
    row_parts = []
    for sig in signals_to_check:
        c_pool = abs(b_df[feat].fillna(0).corr(b_df[sig].fillna(0), method='spearman'))
        c_max = max(
            abs(b_df[b_df['test_id']==t][feat].fillna(0).corr(b_df[b_df['test_id']==t][sig].fillna(0), method='spearman'))
            for t in [1,2,3]
        )
        row_parts.append(f'{sig}: pool={c_pool:.3f} max_rig={c_max:.3f}')
    print(f'  {feat:30s}: {chr(10)+"  ".join(["  "+r for r in row_parts])}')

# ── CHECK 2: thr=0.50 vs Nested, side by side ────────────────────────────────
print()
print('='*70)
print('CHECK 2: thr=0.50 vs NESTED THRESHOLD — Spearman-clean features')
print('  (Matrices from previous report used thr=0.50. Nested adds comparison.)')
print('='*70)

def nested_threshold_fn(X_data, y_data, tr_rigs, groups, min_r=0.70):
    thrs = []
    for i in range(2):
        itr_rig = tr_rigs[i]; ival_rig = tr_rigs[1-i]
        tri = groups[groups==itr_rig].index; vai = groups[groups==ival_rig].index
        sc2 = StandardScaler()
        Xtr2 = sc2.fit_transform(X_data.loc[tri]); Xva2 = sc2.transform(X_data.loc[vai])
        m = xgb.XGBClassifier(n_estimators=300, max_depth=4, eval_metric='logloss', random_state=42, n_jobs=-1)
        m.fit(Xtr2, y_data.loc[tri])
        p = m.predict_proba(Xva2)[:,1]
        prec, rec, thr = precision_recall_curve(y_data.loc[vai], p)
        prec, rec = prec[:-1], rec[:-1]
        valid = rec >= min_r
        bi = np.where(valid, prec, -1).argmax() if valid.any() else np.argmax(rec)
        thrs.append(float(thr[bi]))
    return float(np.mean(thrs)), float(np.std(thrs))

print()
print(f'  {"Rig+HW":<12} {"Threshold":<26} {"Recall":>8} {"Precision":>10} {"FAR":>8} {"FP/Hlth":>10} {"MacroF1":>8}')
print('  ' + '-'*78)
for held in [3,1,2]:
    tr_rigs2 = [r for r in [1,2,3] if r != held]
    tri = groups[groups.isin(tr_rigs2)].index
    tei = groups[groups==held].index
    sc3 = StandardScaler()
    Xtr3 = sc3.fit_transform(X.loc[tri]); Xte3 = sc3.transform(X.loc[tei])
    cb = xgb.XGBClassifier(n_estimators=300, max_depth=4, eval_metric='logloss', random_state=42, n_jobs=-1)
    cb.fit(Xtr3, y_crit.loc[tri])
    cm3 = xgb.XGBClassifier(n_estimators=300, max_depth=5, objective='multi:softprob', num_class=3, eval_metric='mlogloss', random_state=42, n_jobs=-1)
    cm3.fit(Xtr3, y3.loc[tri])
    pb = cb.predict_proba(Xte3)[:,1]
    pm = cm3.predict_proba(Xte3)[:,2]
    p_ens = np.maximum(pb, pm)
    y_te_v = y_crit.loc[tei].values
    n_hlth = (y_te_v==0).sum()
    thr_n, thr_sd = nested_threshold_fn(X, y_crit, tr_rigs2, groups)
    hw = 'DUAL' if held==1 else 'SNGL'
    for thr_v, label in [(0.50, 'thr=0.50 (Config 3)'), (thr_n, f'nested={thr_n:.4f} SD={thr_sd:.4f}')]:
        preds = (p_ens >= thr_v).astype(int)
        tp = int(((preds==1)&(y_te_v==1)).sum()); fp = int(((preds==1)&(y_te_v==0)).sum())
        fn = int(((preds==0)&(y_te_v==1)).sum()); tn = int(((preds==0)&(y_te_v==0)).sum())
        rec_v = tp/max(tp+fn,1); prec_v = tp/max(tp+fp,1); far_v = fp/max(fp+tn,1)
        f1_v = f1_score(y_te_v, preds, average='macro', zero_division=0)
        print(f'  Run{held}[{hw}]  {label:<26} {rec_v*100:>7.2f}% {prec_v*100:>9.2f}% {far_v*100:>7.2f}%  {fp:4d}/{n_hlth}  {f1_v:.4f}')
    print()

print(f'Total elapsed: {time.time()-t0:.1f}s')
