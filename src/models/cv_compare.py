"""
src/models/cv_compare.py
=========================
Comparaison de 4 architectures sur exactement les mêmes folds TimeSeriesSplit :

  - LSTM(64, 32)        — 31 k params
  - GRU(64, 32)         — ~24 k params
  - Transformer(d=32)   — ~9 k params
  - MovAvg(k=3)         — baseline statistique

Hyperparams identiques (epochs, patience, lr, dropout, seed=42) pour assurer
une comparaison non biaisée. Folds TimeSeriesSplit (n=5, test=12 mois) :
fold 1 = 2021, fold 2 = 2022, ..., fold 5 = 2025.

Sortie :
  - models/cv_compare_metrics.csv  (1 ligne par (modèle × fold))
  - models/cv_compare_summary.csv  (mean ± std par modèle)
  - reports/figures/14_cv_compare.png

Usage :
  python src/models/cv_compare.py
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
from gru import build_gru
from transformer import build_transformer

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR    = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
FIG_DIR     = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

INPUT_FILE  = os.path.join(DATA_DIR, "antibes_appart_monthly.csv")

# ── Hyperparamètres CV ────────────────────────────────────────────────────────
N_STEPS    = 12
N_SPLITS   = 5
TEST_SIZE  = 12
VAL_RATIO  = 0.15
SEED       = 42

FEATURE_COLS = [
    "prix_m2_median",
    "volume",
    "surface_median",
    "nb_pieces_median",
    "mois_sin",
    "mois_cos",
]

# Ordre des modèles — important pour les figures
MODEL_NAMES = ["LSTM", "GRU", "Transformer", "MovAvg(k=3)"]
MODEL_COLORS = {
    "LSTM":         "#2563EB",
    "GRU":          "#9333EA",
    "Transformer":  "#DC2626",
    "MovAvg(k=3)":  "#16A34A",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_features():
    df = pd.read_csv(INPUT_FILE, parse_dates=["periode"])
    df = df.sort_values("periode").reset_index(drop=True)
    df["mois_sin"] = np.sin(2 * np.pi * df["mois"] / 12)
    df["mois_cos"] = np.cos(2 * np.pi * df["mois"] / 12)
    feat = df[FEATURE_COLS].copy().dropna().reset_index(drop=True)
    return df, feat


def windows_for_indices(data_sc, target_indices, n_steps=N_STEPS):
    Xs, ys = [], []
    for i in target_indices:
        if i < n_steps:
            continue
        Xs.append(data_sc[i - n_steps : i, :])
        ys.append(data_sc[i, 0])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def compute_metrics(y_true_sc, y_pred_sc, scaler):
    y_true = inverse_target(y_true_sc, scaler)
    y_pred = inverse_target(y_pred_sc, scaler)
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    return mae, rmse, mape


def make_callbacks():
    return [
        callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=10, min_lr=1e-6, verbose=0),
    ]


def train_one_model(builder, X_train, y_train, X_val, y_val, X_test, scaler):
    """Entraîne un modèle Keras et retourne y_pred + n_epochs."""
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    keras.utils.set_random_seed(SEED)

    model = builder((X_train.shape[1], X_train.shape[2]))
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=make_callbacks(),
        verbose=0,
    )
    y_pred = model.predict(X_test, verbose=0).ravel()
    return y_pred, len(history.history["loss"]), model.count_params()


# ── Boucle CV ─────────────────────────────────────────────────────────────────
def run_cv(feat, df):
    data = feat.values.astype(np.float32)
    T = len(data)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)

    rows = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(np.arange(T)), 1):
        period_test_start = df["periode"].iloc[test_idx[0]].strftime("%Y-%m")
        period_test_end   = df["periode"].iloc[test_idx[-1]].strftime("%Y-%m")
        print(f"\n── Fold {fold}/{N_SPLITS}  test = {period_test_start} → {period_test_end}")

        # Scaler fit sur train fold
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data[train_idx])
        data_sc = scaler.transform(data)

        # Split train/val pour early stopping
        n_val = max(int(len(train_idx) * VAL_RATIO), N_STEPS + 1)
        train_main_idx = train_idx[:-n_val]
        val_idx        = train_idx[-n_val:]

        X_train, y_train = windows_for_indices(data_sc, train_main_idx)
        X_val,   y_val   = windows_for_indices(data_sc, val_idx)
        X_test,  y_test  = windows_for_indices(data_sc, test_idx)

        if min(len(X_train), len(X_val), len(X_test)) == 0:
            print(f"  ⚠ Fold {fold} : pas assez de fenêtres, skip")
            continue

        # ── MovAvg ──
        ma = MovingAverageBaseline(k=3)
        y_pred_ma = ma.predict(X_test)
        mae_ma, rmse_ma, mape_ma = compute_metrics(y_test, y_pred_ma, scaler)
        rows.append({
            "fold": fold, "test_start": period_test_start, "test_end": period_test_end,
            "model": "MovAvg(k=3)", "n_params": 0, "epochs": 0,
            "mae": mae_ma, "rmse": rmse_ma, "mape": mape_ma,
        })
        print(f"  MovAvg(k=3)  MAE={mae_ma:5.0f}   RMSE={rmse_ma:5.0f}   MAPE={mape_ma:4.1f}%")

        # ── Modèles deep ──
        for name, builder in [("LSTM", build_lstm),
                              ("GRU", build_gru),
                              ("Transformer", build_transformer)]:
            y_pred, n_ep, n_params = train_one_model(
                builder, X_train, y_train, X_val, y_val, X_test, scaler
            )
            mae, rmse, mape = compute_metrics(y_test, y_pred, scaler)
            rows.append({
                "fold": fold, "test_start": period_test_start, "test_end": period_test_end,
                "model": name, "n_params": n_params, "epochs": n_ep,
                "mae": mae, "rmse": rmse, "mape": mape,
            })
            print(f"  {name:11s}  MAE={mae:5.0f}   RMSE={rmse:5.0f}   MAPE={mape:4.1f}%   "
                  f"({n_ep:3d} epochs, {n_params:,} params)")

    return pd.DataFrame(rows)


# ── Visualisation ────────────────────────────────────────────────────────────
def plot_cv_compare(df_cv: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Panel 1 : MAE par fold (grouped bars 4 modèles × 5 folds) ──
    ax = axes[0]
    folds = sorted(df_cv["fold"].unique())
    n_models = len(MODEL_NAMES)
    width = 0.8 / n_models
    x = np.arange(len(folds))

    for i, model in enumerate(MODEL_NAMES):
        sub = df_cv[df_cv["model"] == model].sort_values("fold")
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, sub["mae"], width,
               label=model, color=MODEL_COLORS[model], alpha=0.85)

    test_starts = df_cv.drop_duplicates("fold").sort_values("fold")["test_start"].tolist()
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{f}\n{ts}" for f, ts in zip(folds, test_starts)], fontsize=9)
    ax.set_ylabel("MAE (€/m²)")
    ax.set_title(f"MAE par fold — comparaison de 4 architectures\n"
                 f"TimeSeriesSplit n={N_SPLITS}, test=12 mois, seed={SEED}",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 2 : boxplot MAE par modèle (distribution sur folds) ──
    ax = axes[1]
    data_box = [df_cv[df_cv["model"] == m]["mae"].values for m in MODEL_NAMES]
    bp = ax.boxplot(data_box, labels=MODEL_NAMES, patch_artist=True,
                    widths=0.5, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=7))
    for patch, model in zip(bp["boxes"], MODEL_NAMES):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_alpha(0.6)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    # Annotations mean ± std
    for i, model in enumerate(MODEL_NAMES, 1):
        sub = df_cv[df_cv["model"] == model]["mae"]
        ax.annotate(f"{sub.mean():.0f}±{sub.std():.0f}",
                    xy=(i, sub.max()), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold",
                    color=MODEL_COLORS[model])

    ax.set_ylabel("MAE (€/m²)")
    ax.set_title("Distribution de la MAE sur 5 folds (boîte à moustaches)\n"
                 "♦ = moyenne, ─ = médiane",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "14_cv_compare.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("📂 Chargement antibes_appart_monthly.csv...")
    df, feat = load_features()
    print(f"   {len(df)} mois  ×  {feat.shape[1]} features")

    print(f"\n🔁 Comparaison de {len(MODEL_NAMES)} modèles sur {N_SPLITS} folds")
    print(f"   Hyperparams partagés : epochs<={EPOCHS}, batch={BATCH_SIZE}, "
          f"patience={PATIENCE}, seed={SEED}")

    df_cv = run_cv(feat, df)

    # Récap par modèle
    print("\n📊 Métriques agrégées (mean ± std sur folds) :")
    summary_rows = []
    for model in MODEL_NAMES:
        sub = df_cv[df_cv["model"] == model]
        if sub.empty:
            continue
        n_params = int(sub["n_params"].iloc[0])
        summary_rows.append({
            "model": model,
            "n_params": f"{n_params:,}" if n_params > 0 else "—",
            "MAE_mean": sub["mae"].mean(),
            "MAE_std":  sub["mae"].std(),
            "RMSE_mean": sub["rmse"].mean(),
            "RMSE_std":  sub["rmse"].std(),
            "MAPE_mean": sub["mape"].mean(),
            "MAPE_std":  sub["mape"].std(),
        })
    df_sum = pd.DataFrame(summary_rows)
    print(df_sum.round(1).to_string(index=False))

    # Sauvegarde
    cv_path  = os.path.join(MODELS_DIR, "cv_compare_metrics.csv")
    sum_path = os.path.join(MODELS_DIR, "cv_compare_summary.csv")
    df_cv.to_csv(cv_path, index=False)
    df_sum.to_csv(sum_path, index=False)
    fig_path = plot_cv_compare(df_cv)

    # Best model
    best = df_sum.sort_values("MAE_mean").iloc[0]
    print(f"\n🏆 Meilleur modèle : {best['model']}  "
          f"(MAE = {best['MAE_mean']:.0f} ± {best['MAE_std']:.0f} €/m²)")

    print(f"\n💾 Sauvegarde :")
    print(f"   {os.path.relpath(cv_path, BASE_DIR)}")
    print(f"   {os.path.relpath(sum_path, BASE_DIR)}")
    print(f"   {os.path.relpath(fig_path, BASE_DIR)}")
    print("\n✅  cv_compare.py terminé.")


if __name__ == "__main__":
    main()
