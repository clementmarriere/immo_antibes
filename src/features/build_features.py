"""
src/features/build_features.py
================================
Prépare les données pour le modèle LSTM.

Pipeline :
  1. Charge le CSV mensuel agrégé
  2. Construit les features (dont encodage cyclique du mois)
  3. Normalise avec MinMaxScaler (fit sur train only → pas de data leakage)
  4. Crée des fenêtres glissantes (X, y)
  5. Split chronologique : 70% train / 15% val / 15% test
  6. Sauvegarde les arrays numpy + le scaler

Usage :
  python src/features/build_features.py
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import MinMaxScaler

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR    = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR  = os.path.join(BASE_DIR, "data", "features")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_FILE  = os.path.join(DATA_DIR, "antibes_appart_monthly.csv")

# ── Hyperparamètres ───────────────────────────────────────────────────────────
N_STEPS     = 12    # longueur de la fenêtre glissante (12 mois d'historique)
HORIZON     = 1     # nb de mois à prédire (1 = next step)
TARGET_COL  = "prix_m2_median"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test_ratio  = 1 - TRAIN_RATIO - VAL_RATIO  = 0.15 (implicite)


# ── 1. Chargement ─────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",", parse_dates=["periode"])
    df = df.sort_values("periode").reset_index(drop=True)
    print(f"[load]  {len(df)} mois chargés  ({df['periode'].min().date()} → {df['periode'].max().date()})")
    return df


# ── 2. Feature Engineering ────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features retenues :
      - prix_m2_median   → cible ET feature décalée
      - volume           → signal de liquidité du marché
      - surface_median   → proxy de la composition de l'offre
      - nb_pieces_median → proxy de la taille des biens
      - mois_sin / mois_cos → encodage cyclique de la saisonnalité
    """
    feat = df.copy()

    # Encodage cyclique du mois (évite la rupture 12→1)
    feat["mois_sin"] = np.sin(2 * np.pi * feat["mois"] / 12)
    feat["mois_cos"] = np.cos(2 * np.pi * feat["mois"] / 12)

    # Features sélectionnées (ordre important : la target doit être en 1ère colonne)
    FEATURE_COLS = [
        TARGET_COL,         # index 0 → utilisé comme target dans les windows
        "volume",
        "surface_median",
        "nb_pieces_median",
        "mois_sin",
        "mois_cos",
    ]

    # Vérification colonnes disponibles
    missing = [c for c in FEATURE_COLS if c not in feat.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")

    feat = feat[FEATURE_COLS].copy()

    # Supprimer d'éventuelles lignes NaN résiduelles
    before = len(feat)
    feat = feat.dropna().reset_index(drop=True)
    if len(feat) < before:
        print(f"[features]  {before - len(feat)} lignes supprimées (NaN)")

    print(f"[features]  {feat.shape[1]} features  ×  {len(feat)} timesteps")
    print(f"            {list(feat.columns)}")
    return feat


# ── 3. Split chronologique ────────────────────────────────────────────────────
def chronological_split(feat: pd.DataFrame):
    n = len(feat)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    n_test  = n - n_train - n_val

    train = feat.iloc[:n_train]
    val   = feat.iloc[n_train : n_train + n_val]
    test  = feat.iloc[n_train + n_val :]

    print(f"[split]  train={len(train)}  val={len(val)}  test={len(test)}  (total={n})")
    return train, val, test


# ── 4. Normalisation (fit sur train uniquement) ───────────────────────────────
def fit_scaler(train: pd.DataFrame):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train.values)
    return scaler

def scale(df: pd.DataFrame, scaler: MinMaxScaler) -> np.ndarray:
    return scaler.transform(df.values)


# ── 5. Sliding Windows ────────────────────────────────────────────────────────
def make_windows(data: np.ndarray, n_steps: int, horizon: int):
    """
    Transforme un array (T, F) en tenseur (N, n_steps, F) pour X
    et vecteur (N,) pour y (valeur cible = TARGET_COL = colonne 0).

    Args:
        data     : array normalisé  shape (T, nb_features)
        n_steps  : longueur de la fenêtre d'entrée
        horizon  : décalage de la prédiction (1 = next step)

    Returns:
        X : (N, n_steps, nb_features)
        y : (N,)  — valeur normalisée du TARGET_COL au pas suivant
    """
    X, y = [], []
    for i in range(len(data) - n_steps - horizon + 1):
        X.append(data[i : i + n_steps, :])          # fenêtre complète
        y.append(data[i + n_steps + horizon - 1, 0]) # target = col 0 (prix_m2_median)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ── 6. Helpers de dénormalisation ─────────────────────────────────────────────
def inverse_target(y_scaled: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """
    Dénormalise uniquement la colonne target (col 0).
    Reconstruit un array fictif à nb_features colonnes, applique inverse_transform,
    puis extrait la colonne 0.
    """
    nb_features = scaler.n_features_in_
    dummy = np.zeros((len(y_scaled), nb_features), dtype=np.float32)
    dummy[:, 0] = y_scaled
    return scaler.inverse_transform(dummy)[:, 0]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. Chargement
    df = load_data(INPUT_FILE)

    # 2. Features
    feat = build_features(df)

    # 3. Split
    train_df, val_df, test_df = chronological_split(feat)

    # 4. Normalisation
    scaler = fit_scaler(train_df)
    train_sc = scale(train_df, scaler)
    val_sc   = scale(val_df,   scaler)
    test_sc  = scale(test_df,  scaler)

    # 5. Sliding windows
    X_train, y_train = make_windows(train_sc, N_STEPS, HORIZON)
    X_val,   y_val   = make_windows(val_sc,   N_STEPS, HORIZON)
    X_test,  y_test  = make_windows(test_sc,  N_STEPS, HORIZON)

    print(f"\n[windows]")
    print(f"  X_train : {X_train.shape}   y_train : {y_train.shape}")
    print(f"  X_val   : {X_val.shape}     y_val   : {y_val.shape}")
    print(f"  X_test  : {X_test.shape}    y_test  : {y_test.shape}")

    # 6. Sauvegarde
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"),   X_val)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"),  y_test)

    scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n[save]  Arrays → {OUTPUT_DIR}/")
    print(f"        Scaler → {scaler_path}")
    print("\n✅  build_features terminé. Prochaine étape : src/models/lstm.py")


if __name__ == "__main__":
    main()