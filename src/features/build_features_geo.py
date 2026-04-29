"""
src/features/build_features_geo.py
====================================
Prépare les features LSTM pour chaque quartier d'Antibes.

Pipeline (équivalent build_features.py mais par quartier) :
  1. Charge antibes_geo_monthly.csv
  2. Pour chaque quartier (sauf "Autre" — non fiable, 0.1% des transactions) :
       - Construit les features (avec encodage cyclique du mois)
       - Split chronologique 70/15/15
       - MinMaxScaler fit sur train uniquement
       - Sliding windows N_STEPS=12, HORIZON=1
       - Sauvegarde arrays + scaler dans data/features/geo/{quartier}/

Usage :
  python src/features/build_features_geo.py
"""

import numpy as np
import pandas as pd
import pickle
import os
import re
from sklearn.preprocessing import MinMaxScaler

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR    = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR  = os.path.join(BASE_DIR, "data", "features", "geo")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_FILE  = os.path.join(DATA_DIR, "antibes_geo_monthly.csv")

# ── Hyperparamètres (alignés sur build_features.py) ──────────────────────────
N_STEPS     = 12
HORIZON     = 1
TARGET_COL  = "prix_m2_median"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

FEATURE_COLS = [
    TARGET_COL,
    "volume",
    "surface_median",
    "nb_pieces_median",
    "mois_sin",
    "mois_cos",
]

# Quartier exclu : trop peu de transactions, target NaN par moments
EXCLUDED_QUARTIERS = {"Autre"}


# ── Helpers ──────────────────────────────────────────────────────────────────
def slugify(name: str) -> str:
    """Vieille Ville → vieille_ville ; Cap d'Antibes → cap_d_antibes"""
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()
    feat["mois_sin"] = np.sin(2 * np.pi * feat["mois"] / 12)
    feat["mois_cos"] = np.cos(2 * np.pi * feat["mois"] / 12)

    missing = [c for c in FEATURE_COLS if c not in feat.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    feat = feat[FEATURE_COLS].copy()
    return feat


def chronological_split(feat: pd.DataFrame):
    n = len(feat)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    train = feat.iloc[:n_train]
    val   = feat.iloc[n_train : n_train + n_val]
    test  = feat.iloc[n_train + n_val :]
    return train, val, test


def make_windows(data: np.ndarray, n_steps: int, horizon: int):
    X, y = [], []
    for i in range(len(data) - n_steps - horizon + 1):
        X.append(data[i : i + n_steps, :])
        y.append(data[i + n_steps + horizon - 1, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ── Pipeline par quartier ────────────────────────────────────────────────────
def process_quartier(df_q: pd.DataFrame, quartier: str) -> dict:
    df_q = df_q.sort_values("periode").reset_index(drop=True)

    # Volume manquant signifie 0 transactions ce mois-ci → on remplit
    df_q["volume"] = df_q["volume"].fillna(0)

    # Forward-fill puis backward-fill pour les médianes (mois sans transaction)
    for col in ["prix_m2_median", "surface_median", "nb_pieces_median"]:
        df_q[col] = df_q[col].ffill().bfill()

    feat = build_features(df_q)
    n_nan = feat.isna().sum().sum()
    if n_nan > 0:
        feat = feat.dropna().reset_index(drop=True)

    train_df, val_df, test_df = chronological_split(feat)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df.values)

    train_sc = scaler.transform(train_df.values)
    val_sc   = scaler.transform(val_df.values)
    test_sc  = scaler.transform(test_df.values)

    X_train, y_train = make_windows(train_sc, N_STEPS, HORIZON)
    X_val,   y_val   = make_windows(val_sc,   N_STEPS, HORIZON)
    X_test,  y_test  = make_windows(test_sc,  N_STEPS, HORIZON)

    out_dir = os.path.join(OUTPUT_DIR, slugify(quartier))
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "X_val.npy"),   X_val)
    np.save(os.path.join(out_dir, "y_val.npy"),   y_val)
    np.save(os.path.join(out_dir, "X_test.npy"),  X_test)
    np.save(os.path.join(out_dir, "y_test.npy"),  y_test)

    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    return {
        "quartier": quartier,
        "slug": slugify(quartier),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "n_nan_filled": int(n_nan),
        "out_dir": out_dir,
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("📂 Chargement antibes_geo_monthly.csv...")
    df = pd.read_csv(INPUT_FILE, parse_dates=["periode"])
    print(f"   {len(df)} lignes  ×  {df['quartier'].nunique()} quartiers")

    quartiers = [q for q in sorted(df["quartier"].unique())
                 if q not in EXCLUDED_QUARTIERS]
    print(f"   Quartiers traités ({len(quartiers)}) : {quartiers}\n")

    summary = []
    for q in quartiers:
        df_q = df[df["quartier"] == q].copy()
        info = process_quartier(df_q, q)
        summary.append(info)
        print(f"  ✓ {q:25s}  train={info['n_train']:3d}  val={info['n_val']:3d}  "
              f"test={info['n_test']:3d}  → {os.path.relpath(info['out_dir'], BASE_DIR)}")

    print(f"\n✅  build_features_geo terminé. Prochaine étape : src/models/lstm_geo.py")


if __name__ == "__main__":
    main()
