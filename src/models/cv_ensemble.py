"""
src/models/cv_ensemble.py
==========================
Forecast ensemble : combinaison des 4 modèles (LSTM, GRU, Transformer, MovAvg)
évaluée sur exactement les mêmes folds TimeSeriesSplit que cv_compare.py.

Trois stratégies de combinaison testées :

  - Moyenne équipondérée  : (LSTM + GRU + Transformer + MovAvg) / 4
  - Médiane               : robuste aux prédictions extrêmes d'un modèle isolé
  - NNLS optimal          : poids non-négatifs appris sur le val set par moindres
                            carrés contraints (somme des poids ≤ 1)

Hypothèse : si les modèles ont des biais décorrélés, l'ensemble réduit la MAE.

Sortie :
  - models/cv_ensemble_metrics.csv  (1 ligne par modèle × fold)
  - models/cv_ensemble_summary.csv  (mean ± std par modèle)
  - reports/figures/17_cv_ensemble.png

Usage :
  python src/models/cv_ensemble.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy.optimize import nnls

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

# ── Hyperparams (alignés sur cv_compare.py) ──────────────────────────────────
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

BASE_MODELS = ["LSTM", "GRU", "Transformer", "MovAvg"]
ENSEMBLES   = ["Ensemble_mean", "Ensemble_median", "Ensemble_NNLS"]
ALL_MODELS  = BASE_MODELS + ENSEMBLES

MODEL_COLORS = {
    "LSTM":            "#2563EB",
    "GRU":             "#9333EA",
    "Transformer":     "#DC2626",
    "MovAvg":          "#16A34A",
    "Ensemble_mean":   "#F59E0B",
    "Ensemble_median": "#EC4899",
    "Ensemble_NNLS":   "#0EA5E9",
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


def metrics(y_true_sc, y_pred_sc, scaler):
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


def train_one_model(builder, X_train, y_train, X_val, y_val, X_test):
    """Reproductibilité : seed reset à chaque appel."""
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    keras.utils.set_random_seed(SEED)

    model = builder((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=EPOCHS, batch_size=BATCH_SIZE,
              callbacks=make_callbacks(), verbose=0)
    y_pred_val  = model.predict(X_val,  verbose=0).ravel()
    y_pred_test = model.predict(X_test, verbose=0).ravel()
    return y_pred_val, y_pred_test


# ── Stratégies d'ensemble ────────────────────────────────────────────────────
def fit_nnls_weights(P_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    """
    P_val : (n_val, n_models) — prédictions sur le val set
    y_val : (n_val,)
    Retourne des poids w >= 0, normalisés à somme 1 (NNLS + normalisation).
    Si tous les poids sont 0 (cas dégénéré), fallback à équipondéré.
    """
    w, _ = nnls(P_val, y_val)
    if w.sum() < 1e-9:
        return np.ones(P_val.shape[1]) / P_val.shape[1]
    return w / w.sum()


def ensemble_predictions(P_test: np.ndarray, weights: np.ndarray = None):
    """P_test : (n_test, n_models). Retourne dict de prédictions."""
    return {
        "Ensemble_mean":   P_test.mean(axis=1),
        "Ensemble_median": np.median(P_test, axis=1),
        "Ensemble_NNLS":   P_test @ weights,
    }


# ── Boucle CV ────────────────────────────────────────────────────────────────
def run_cv(feat, df):
    data = feat.values.astype(np.float32)
    T = len(data)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)

    rows = []
    weights_log = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(np.arange(T)), 1):
        period_test_start = df["periode"].iloc[test_idx[0]].strftime("%Y-%m")
        period_test_end   = df["periode"].iloc[test_idx[-1]].strftime("%Y-%m")
        print(f"\n── Fold {fold}/{N_SPLITS}  test = {period_test_start} → {period_test_end}")

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data[train_idx])
        data_sc = scaler.transform(data)

        n_val = max(int(len(train_idx) * VAL_RATIO), N_STEPS + 1)
        train_main_idx = train_idx[:-n_val]
        val_idx        = train_idx[-n_val:]

        X_train, y_train = windows_for_indices(data_sc, train_main_idx)
        X_val,   y_val   = windows_for_indices(data_sc, val_idx)
        X_test,  y_test  = windows_for_indices(data_sc, test_idx)

        if min(len(X_train), len(X_val), len(X_test)) == 0:
            print(f"  ⚠ Fold {fold} : pas assez de fenêtres, skip")
            continue

        # ── Prédictions des modèles de base ──
        # MovAvg ne s'entraîne pas
        ma = MovingAverageBaseline(k=3)
        y_ma_val  = ma.predict(X_val)
        y_ma_test = ma.predict(X_test)

        preds_val  = {"MovAvg": y_ma_val}
        preds_test = {"MovAvg": y_ma_test}

        for name, builder in [("LSTM", build_lstm),
                              ("GRU", build_gru),
                              ("Transformer", build_transformer)]:
            y_val_pred, y_test_pred = train_one_model(
                builder, X_train, y_train, X_val, y_val, X_test
            )
            preds_val[name]  = y_val_pred
            preds_test[name] = y_test_pred

        # ── Métriques modèles de base ──
        for name in BASE_MODELS:
            mae, rmse, mape = metrics(y_test, preds_test[name], scaler)
            rows.append({
                "fold": fold, "test_start": period_test_start, "test_end": period_test_end,
                "model": name, "mae": mae, "rmse": rmse, "mape": mape,
            })
            print(f"  {name:11s}  MAE={mae:5.0f}   RMSE={rmse:5.0f}   MAPE={mape:4.1f}%")

        # ── NNLS sur val ──
        P_val  = np.column_stack([preds_val[m]  for m in BASE_MODELS])
        P_test = np.column_stack([preds_test[m] for m in BASE_MODELS])

        weights = fit_nnls_weights(P_val, y_val)
        weights_log.append({
            "fold": fold,
            **{f"w_{m}": round(float(w), 3) for m, w in zip(BASE_MODELS, weights)},
        })

        # ── Ensembles ──
        y_ens = ensemble_predictions(P_test, weights)
        for name, y_pred in y_ens.items():
            mae, rmse, mape = metrics(y_test, y_pred, scaler)
            rows.append({
                "fold": fold, "test_start": period_test_start, "test_end": period_test_end,
                "model": name, "mae": mae, "rmse": rmse, "mape": mape,
            })
            tag = name.replace("Ensemble_", "ENS-")
            print(f"  {tag:11s}  MAE={mae:5.0f}   RMSE={rmse:5.0f}   MAPE={mape:4.1f}%")

        # Trace des poids NNLS du fold
        w_str = "  ".join(f"{m}={w:.2f}" for m, w in zip(BASE_MODELS, weights))
        print(f"  Poids NNLS  → {w_str}")

    return pd.DataFrame(rows), pd.DataFrame(weights_log)


# ── Visualisation ────────────────────────────────────────────────────────────
def plot_ensemble(df_cv, df_weights):
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))

    # ── Panel 1 : MAE par fold (toutes méthodes) ──
    ax = axes[0]
    folds = sorted(df_cv["fold"].unique())
    n_models = len(ALL_MODELS)
    width = 0.8 / n_models
    x = np.arange(len(folds))

    for i, model in enumerate(ALL_MODELS):
        sub = df_cv[df_cv["model"] == model].sort_values("fold")
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, sub["mae"], width,
               label=model, color=MODEL_COLORS[model], alpha=0.85)

    test_starts = df_cv.drop_duplicates("fold").sort_values("fold")["test_start"].tolist()
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{f}\n{ts}" for f, ts in zip(folds, test_starts)], fontsize=9)
    ax.set_ylabel("MAE (€/m²)")
    ax.set_title("MAE par fold — modèles de base + 3 stratégies d'ensemble",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 2 : boxplot par modèle (distribution sur folds) ──
    ax = axes[1]
    data_box = [df_cv[df_cv["model"] == m]["mae"].values for m in ALL_MODELS]
    bp = ax.boxplot(data_box, labels=ALL_MODELS, patch_artist=True,
                    widths=0.6, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=6))
    for patch, model in zip(bp["boxes"], ALL_MODELS):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_alpha(0.6)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    for i, model in enumerate(ALL_MODELS, 1):
        sub = df_cv[df_cv["model"] == model]["mae"]
        ax.annotate(f"{sub.mean():.0f}",
                    xy=(i, sub.max()), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, fontweight="bold",
                    color=MODEL_COLORS[model])

    ax.set_ylabel("MAE (€/m²)")
    ax.set_title("Distribution MAE sur 5 folds — base vs ensembles\n♦ = moyenne",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right", fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "17_cv_ensemble.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("📂 Chargement antibes_appart_monthly.csv...")
    df, feat = load_features()
    print(f"   {len(df)} mois  ×  {feat.shape[1]} features")

    print(f"\n🔁 Cross-validation Ensemble  (n_splits={N_SPLITS}, test=12 mois)")
    print(f"   3 modèles deep + MovAvg → 3 stratégies d'ensemble")

    df_cv, df_weights = run_cv(feat, df)

    # Récap
    print("\n📊 Métriques agrégées (mean ± std sur folds) :")
    summary_rows = []
    for model in ALL_MODELS:
        sub = df_cv[df_cv["model"] == model]
        if sub.empty:
            continue
        summary_rows.append({
            "model": model,
            "MAE_mean":  sub["mae"].mean(),
            "MAE_std":   sub["mae"].std(),
            "RMSE_mean": sub["rmse"].mean(),
            "RMSE_std":  sub["rmse"].std(),
            "MAPE_mean": sub["mape"].mean(),
            "MAPE_std":  sub["mape"].std(),
        })
    df_sum = pd.DataFrame(summary_rows).sort_values("MAE_mean").reset_index(drop=True)
    print(df_sum.round(1).to_string(index=False))

    # Best
    best = df_sum.iloc[0]
    print(f"\n🏆 Meilleur modèle (MAE moyenne) : {best['model']}  "
          f"(MAE = {best['MAE_mean']:.0f} ± {best['MAE_std']:.0f} €/m²)")

    print("\n📐 Poids NNLS appris par fold :")
    print(df_weights.to_string(index=False))

    # Sauvegarde
    cv_path  = os.path.join(MODELS_DIR, "cv_ensemble_metrics.csv")
    sum_path = os.path.join(MODELS_DIR, "cv_ensemble_summary.csv")
    w_path   = os.path.join(MODELS_DIR, "cv_ensemble_weights.csv")
    df_cv.to_csv(cv_path, index=False)
    df_sum.to_csv(sum_path, index=False)
    df_weights.to_csv(w_path, index=False)
    fig_path = plot_ensemble(df_cv, df_weights)

    print(f"\n💾 Sauvegarde :")
    print(f"   {os.path.relpath(cv_path,  BASE_DIR)}")
    print(f"   {os.path.relpath(sum_path, BASE_DIR)}")
    print(f"   {os.path.relpath(w_path,   BASE_DIR)}")
    print(f"   {os.path.relpath(fig_path, BASE_DIR)}")
    print("\n✅  cv_ensemble.py terminé.")


if __name__ == "__main__":
    main()
