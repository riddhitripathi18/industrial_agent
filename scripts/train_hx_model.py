"""
train_hx_model.py
Predictive Maintenance — Heat Exchanger ML Training Pipeline
Trains:
  1. XGBoost classifier for TEMA Fouling Breach (Rf >= 0.0005 m^2*K/W)
  2. XGBoost regressor for Fouling Resistance Rf (m^2*K/W)
"""
import os
import math
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, r2_score
import xgboost as xgb

BASE_DIR = r"c:\Users\riddh\OneDrive\Desktop\predictive_maintenance"
DATA_PATH = os.path.join(BASE_DIR, "heat_exchanger_fouling_clean.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("Loading Heat Exchanger dataset...")
df = pd.read_csv(DATA_PATH)

exchangers = ["E01", "E02", "E03", "E04", "E05"]
shell_fluids = {
    "E01": "HeavyNaphtha",
    "E02": "Kero",
    "E03": "LightDiesel",
    "E04": "LVGO",
    "E05": "HeavyDiesel"
}

AREAS = {"E01": 105.0, "E02": 110.0, "E03": 120.0, "E04": 95.0, "E05": 85.0}
CP_TUBE = 2.1  # kJ/kg*K crude

records = []
for hx in exchangers:
    shell = shell_fluids[hx]
    t_in_col = f"{hx}_Crude_Tube_T_In_degC"
    t_out_col = f"{hx}_Crude_Tube_T_Out_degC"
    t_m_col = f"{hx}_Crude_Tube_m_kg_s"
    
    s_in_col = f"{hx}_{shell}_Shell_T_In_degC"
    s_out_col = f"{hx}_{shell}_Shell_T_Out_degC"
    s_m_col = f"{hx}_{shell}_Shell_m_kg_s"
    
    sub = df.iloc[::5].copy()
    
    for _, row in sub.iterrows():
        t_in = row[t_in_col]
        t_out = row[t_out_col]
        t_flow = row[t_m_col]
        s_in = row[s_in_col]
        s_out = row[s_out_col]
        s_flow = row[s_m_col]
        time_hr = row["Time_hr"]
        api = row["Crude_API"]
        tan = row["Crude_TAN"]
        chlorides = row["Crude_Chlorides"]
        
        delta_t_tube = abs(t_out - t_in)
        delta_t_shell = abs(s_in - s_out)
        
        q_kw = t_flow * CP_TUBE * delta_t_tube
        
        dt1 = abs(s_in - t_out)
        dt2 = abs(s_out - t_in)
        if dt1 > 0 and dt2 > 0 and dt1 != dt2:
            lmtd = (dt1 - dt2) / math.log(dt1 / dt2)
        else:
            lmtd = max((dt1 + dt2) / 2.0, 1.0)
            
        area = AREAS[hx]
        u_w_m2k = (q_kw * 1000.0) / (area * max(lmtd, 1.0)) if lmtd > 0 else 1000.0
        
        records.append({
            "exchanger_id": hx,
            "time_hr": time_hr,
            "tube_t_in": t_in,
            "tube_t_out": t_out,
            "tube_flow": t_flow,
            "shell_t_in": s_in,
            "shell_t_out": s_out,
            "shell_flow": s_flow,
            "delta_t_tube": delta_t_tube,
            "delta_t_shell": delta_t_shell,
            "q_duty_kw": q_kw,
            "lmtd": lmtd,
            "u_overall": u_w_m2k,
            "crude_api": api,
            "crude_tan": tan,
            "crude_chlorides": chlorides,
        })

data_df = pd.DataFrame(records)

clean_u = data_df[data_df["time_hr"] < 50].groupby("exchanger_id")["u_overall"].median().to_dict()

def calc_rf(row):
    u_c = clean_u.get(row["exchanger_id"], 1200.0)
    u_curr = max(row["u_overall"], 100.0)
    rf = (1.0 / u_curr) - (1.0 / u_c)
    return max(rf, 0.0)

data_df["Rf"] = data_df.apply(calc_rf, axis=1)
data_df["tema_breach"] = (data_df["Rf"] >= 0.0005).astype(int)

data_df = pd.get_dummies(data_df, columns=["exchanger_id"], drop_first=False)

feature_cols = [c for c in data_df.columns if c not in ["Rf", "tema_breach", "u_overall"]]
print(f"Training on {len(data_df)} records with {len(feature_cols)} features...")

X = data_df[feature_cols]
y_clf = data_df["tema_breach"]
y_reg = data_df["Rf"]

X_train, X_test, y_train_clf, y_test_clf, y_train_reg, y_test_reg = train_test_split(
    X, y_clf, y_reg, test_size=0.2, random_state=42, stratify=y_clf if y_clf.nunique() > 1 else None
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pos_count = max(int(y_train_clf.sum()), 1)
neg_count = len(y_train_clf) - pos_count
scale_pos_weight = neg_count / pos_count

clf = xgb.XGBClassifier(
    n_estimators=120,
    max_depth=5,
    learning_rate=0.08,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss"
)
clf.fit(X_train_scaled, y_train_clf)
y_pred_clf = clf.predict(X_test_scaled)
y_prob_clf = clf.predict_proba(X_test_scaled)[:, 1] if len(np.unique(y_train_clf)) > 1 else y_pred_clf

roc = roc_auc_score(y_test_clf, y_prob_clf) if len(np.unique(y_test_clf)) > 1 else 1.0
print(f"HX Fouling Classifier ROC-AUC: {roc:.4f}")

reg = xgb.XGBRegressor(
    n_estimators=120,
    max_depth=5,
    learning_rate=0.08,
    random_state=42
)
reg.fit(X_train_scaled, y_train_reg)
y_pred_reg = reg.predict(X_test_scaled)
r2 = r2_score(y_test_reg, y_pred_reg)
print(f"HX Fouling Regressor R^2: {r2:.4f}")

with open(os.path.join(MODELS_DIR, "hx_fouling_classifier.pkl"), "wb") as f:
    pickle.dump(clf, f)
with open(os.path.join(MODELS_DIR, "hx_fouling_regressor.pkl"), "wb") as f:
    pickle.dump(reg, f)
with open(os.path.join(MODELS_DIR, "hx_scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(MODELS_DIR, "hx_feature_cols.pkl"), "wb") as f:
    pickle.dump(feature_cols, f)

print("Heat Exchanger models successfully saved to models/")
