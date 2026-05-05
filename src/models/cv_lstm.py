"""
src/models/cv_lstm.py
======================
Cross-validation temporelle (TimeSeriesSplit) du LSTM global.

Pour chaque fold :
  - Scaler refitté sur train uniquement (no leakage)
  - LSTM(64,32) ré-entraîné from scratch (mêmes hyperparams, seed=42)
  - Évaluation sur 12 mois de test (1 an glissant)
  - Comparaison vs baseline MovAvg(k=3) sur exactement les mêmes folds

Donne des intervalles de confiance sur les métriques (mean ± std sur 5 folds)
plutôt qu'un point unique sur 11 mois bruités.

Sortie :
  - models/cv_metrics.csv  (1 ligne par fold pour chaque modèle)
  - reports/figures/13_timeseries_cv.png

Usage :
  python src/models/cv_lstm.py
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lstm import (
    build_lstm,
    MovingAverageBaseline,
    inverse_target,
    EPOCHS,
    BATCH_SIZE,
    PATIENCE,
)

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR    = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
FIG_DIR     = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

INPUT_FILE  = os.path.join(DATA_DIR, "antibes_appart_monthly.csv")

# ── Hyperparamètres ───────────────────────────────────────────────────────────
N_STEPS    = 12
HORIZON    = 1
N_SPLITS   = 5
TEST_SIZE  = 12       # 1 an de test par fold
VAL_RATIO  = 0.15     # 15% du train fold sert de val pour l'early stopping
SEED       = 42

FEATURE_COLS = [
    "prix_m2_median",
    "volume",
    "surface_median",
    "nb_pieces_median",
    "mois_sin",
    "mois_cos",
]


# ── Préparation features ─────────────────────────────────────────────────────
def load_features() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE, parse_dates=["periode"])
    df = df.sort_values("periode").reset_index(drop=True)
    df["mois_sin"] = np.sin(2 * np.pi * df["mois"] / 12)
    df["mois_cos"] = np.cos(2 * np.pi * df["mois"] / 12)
    feat = df[FEATURE_COLS].copy().dropna().reset_index(drop=True)
    return df, feat


# ── Création de fenêtres pour des indices de target donnés ───────────────────
def windows_for_indices(data_sc: np.ndarray, target_indices: np.ndarray,
                        n_steps: int = N_STEPS):
    """Construit X, y où chaque y correspond à un index de target_indices.
    Filtre les indices < n_steps (pas assez d'historique pour la fenêtre)."""
    Xs, ys = [], []
    for i in target_indices:
        if i < n_steps:
            continue
        Xs.append(data_sc[i - n_steps : i, :])
        ys.append(data_sc[i, 0])     # target = col 0 (prix_m2_median)
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ── Métriques ────────────────────────────────────────────────────────────────
def compute_metrics(y_true_sc, y_pred_sc, scaler):
    y_true = inverse_target(y_true_sc, scaler)
    y_pred = inverse_target(y_pred_sc, scaler)
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    return mae, rmse, mape


# ── Callbacks ────────────────────────────────────────────────────────────────
def make_callbacks():
    return [
        callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=10, min_lr=1e-6, verbose=0),
    ]


# ── Boucle de CV ─────────────────────────────────────────────────────────────
def run_cv(feat: pd.DataFrame, df: pd.DataFrame):
    data = feat.values.astype(np.float32)
    T = len(data)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)

    rows = []
    fold_predictions = []  # pour la figure

    for fold, (train_idx, test_idx) in enumerate(tscv.split(np.arange(T)), 1):
        period_test_start = df["periode"].iloc[test_idx[0]].strftime("%Y-%m")
        period_test_end   = df["periode"].iloc[test_idx[-1]].strftime("%Y-%m")
        print(f"\n── Fold {fold}/{N_SPLITS}  test = {period_test_start} → {period_test_end} "
              f"(train n={len(train_idx)}, test n={len(test_idx)})")

        # Reproductibilité par fold
        np.random.seed(SEED)
        tf.random.set_seed(SEED)
        keras.utils.set_random_seed(SEED)

        # Scaler fit sur train uniquement
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data[train_idx])
        data_sc = scaler.transform(data)

        # Split train fold en train/val (15% val pour early stopping)
        n_val = max(int(len(train_idx) * VAL_RATIO), N_STEPS + 1)
        train_main_idx = train_idx[:-n_val]
        val_idx        = train_idx[-n_val:]

        X_train, y_train = windows_for_indices(data_sc, train_main_idx)
        X_val,   y_val   = windows_for_indices(data_sc, val_idx)
        X_test,  y_test  = windows_for_indices(data_sc, test_idx)

        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            print(f"  ⚠ Fold {fold} : pas assez de fenêtres, skip")
            continue

        # ── Baseline MovAvg(k=3) ──
        ma = MovingAverageBaseline(k=3)
        y_pred_ma = ma.predict(X_test)
        mae_ma, rmse_ma, mape_ma = compute_metrics(y_test, y_pred_ma, scaler)

        # ── LSTM ──
        model = build_lstm((N_STEPS, X_train.shape[2]))
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=make_callbacks(),
            verbose=0,
        )
        y_pred_lstm = model.predict(X_test, verbose=0).ravel()
        mae_l, rmse_l, mape_l = compute_metrics(y_test, y_pred_lstm, scaler)

        rows.append({
            "fold": fold,
            "test_start": period_test_start,
            "test_end": period_test_end,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "epochs": len(history.history["loss"]),
            "lstm_mae": mae_l, "lstm_rmse": rmse_l, "lstm_mape": mape_l,
            "ma_mae": mae_ma,  "ma_rmse": rmse_ma,  "ma_mape": mape_ma,
        })

        # Pour le plot : prédictions dénormalisées
        y_test_denorm  = inverse_target(y_test, scaler)
        y_lstm_denorm  = inverse_target(y_pred_lstm, scaler)
        y_ma_denorm    = inverse_target(y_pred_ma, scaler)
        dates_test     = df["periode"].iloc[test_idx[N_STEPS - len(y_test):]].values \
                         if len(y_test) < len(test_idx) else df["periode"].iloc[test_idx].values

        fold_predictions.append({
            "fold": fold,
            "dates": dates_test,
            "y_true": y_test_denorm,
            "y_lstm": y_lstm_denorm,
            "y_ma": y_ma_denorm,
        })

        print(f"  LSTM   MAE={mae_l:5.0f}   RMSE={rmse_l:5.0f}   MAPE={mape_l:4.1f}%   "
              f"({len(history.history['loss'])} epochs)")
        print(f"  MovAvg MAE={mae_ma:5.0f}   RMSE={rmse_ma:5.0f}   MAPE={mape_ma:4.1f}%")

    return pd.DataFrame(rows), fold_predictions


# ── Visualisation ────────────────────────────────────────────────────────────
def plot_cv(df_cv: pd.DataFrame, fold_predictions: list):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # ── Panel 1 : MAE par fold ──
    ax = axes[0]
    folds = df_cv["fold"].values
    width = 0.35
    x = np.arange(len(folds))
    ax.bar(x - width/2, df_cv["lstm_mae"], width,
           label="LSTM",   color="#2563EB", alpha=0.85)
    ax.bar(x + width/2, df_cv["ma_mae"],   width,
           label="MovAvg(k=3)", color="#16A34A", alpha=0.85)

    # Lignes mean ± std en horizontal
    lstm_mean = df_cv["lstm_mae"].mean()
    lstm_std  = df_cv["lstm_mae"].std()
    ma_mean   = df_cv["ma_mae"].mean()
    ma_std    = df_cv["ma_mae"].std()
    ax.axhline(lstm_mean, color="#2563EB", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axhline(ma_mean,   color="#16A34A", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.fill_between([-0.5, len(folds) - 0.5],
                    lstm_mean - lstm_std, lstm_mean + lstm_std,
                    alpha=0.08, color="#2563EB")
    ax.fill_between([-0.5, len(folds) - 0.5],
                    ma_mean - ma_std, ma_mean + ma_std,
                    alpha=0.08, color="#16A34A")

    ax.set_xticks(x)
    ax.set_xticklabels([f"F{f}\n{r['test_start']}" for f, r in zip(folds, df_cv.to_dict("records"))],
                       fontsize=9)
    ax.set_ylabel("MAE (€/m²)")
    ax.set_title(f"MAE par fold (TimeSeriesSplit n={N_SPLITS}, test=12 mois)\n"
                 f"LSTM : {lstm_mean:.0f} ± {lstm_std:.0f}    "
                 f"MovAvg : {ma_mean:.0f} ± {ma_std:.0f}",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xlim(-0.5, len(folds) - 0.5)

    # ── Panel 2 : prédictions vs réalité (fold le plus récent) ──
    ax = axes[1]
    last = fold_predictions[-1]
    ax.plot(last["dates"], last["y_true"],  color="#111827", linewidth=2.2,
            marker="o", markersize=5, label="Observé")
    ax.plot(last["dates"], last["y_lstm"],  color="#2563EB", linewidth=2.0,
            marker="s", markersize=4, linestyle="--", label="LSTM")
    ax.plot(last["dates"], last["y_ma"],    color="#16A34A", linewidth=2.0,
            marker="^", markersize=4, linestyle=":",  label="MovAvg(k=3)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f} €"))
    ax.set_title(f"Fold {last['fold']} — prédictions vs observé sur 12 mois",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Période")
    ax.set_ylabel("Prix/m² médian")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "13_timeseries_cv.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("📂 Chargement antibes_appart_monthly.csv...")
    df, feat = load_features()
    print(f"   {len(df)} mois  ×  {feat.shape[1]} features\n")

    print(f"🔁 Cross-validation TimeSeriesSplit  (n_splits={N_SPLITS}, test_size={TEST_SIZE})")
    print(f"   Hyperparams LSTM : (64,32), dropout=0.2, epochs<={EPOCHS}, "
          f"patience={PATIENCE}, seed={SEED}")

    df_cv, fold_predictions = run_cv(feat, df)

    # Récap
    print("\n📊 Métriques agrégées (mean ± std sur folds) :")
    summary = pd.DataFrame({
        "model":  ["LSTM", "MovAvg(k=3)"],
        "MAE":    [f"{df_cv['lstm_mae'].mean():.0f} ± {df_cv['lstm_mae'].std():.0f}",
                   f"{df_cv['ma_mae'].mean():.0f} ± {df_cv['ma_mae'].std():.0f}"],
        "RMSE":   [f"{df_cv['lstm_rmse'].mean():.0f} ± {df_cv['lstm_rmse'].std():.0f}",
                   f"{df_cv['ma_rmse'].mean():.0f} ± {df_cv['ma_rmse'].std():.0f}"],
        "MAPE":   [f"{df_cv['lstm_mape'].mean():.2f} ± {df_cv['lstm_mape'].std():.2f}%",
                   f"{df_cv['ma_mape'].mean():.2f} ± {df_cv['ma_mape'].std():.2f}%"],
    })
    print(summary.to_string(index=False))

    # Sauvegarde
    cv_path = os.path.join(MODELS_DIR, "cv_metrics.csv")
    df_cv.to_csv(cv_path, index=False)
    fig_path = plot_cv(df_cv, fold_predictions)

    print(f"\n💾 Sauvegarde :")
    print(f"   {os.path.relpath(cv_path, BASE_DIR)}")
    print(f"   {os.path.relpath(fig_path, BASE_DIR)}")
    print("\n✅  cv_lstm.py terminé.")


if __name__ == "__main__":
    main()
