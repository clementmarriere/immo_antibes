"""
src/models/transformer_geo.py
==============================
Entraîne un Transformer encoder par quartier d'Antibes (6 modèles).

Motivation : la CV comparative (figure 14) montre que le Transformer survit
beaucoup mieux que LSTM/GRU au choc de régime BCE 2023 (MAE 313 vs 603).
On l'applique donc à la modélisation par quartier pour le forecast 2026.

Pour chaque quartier :
  - charge les arrays préparés par build_features_geo.py
  - entraîne Transformer (mêmes hyperparams que transformer.py, seed=42)
  - évalue sur test set : MAE / RMSE / MAPE
  - compare à la baseline MovAvg(k=3)
  - sauvegarde poids + prédictions + métriques

Sortie :
  - models/geo/{slug}_transformer_best.keras
  - models/geo/transformer_metrics.csv

Usage :
  python src/models/transformer_geo.py
"""

import numpy as np
import pandas as pd
import pickle
import os
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lstm import (
    MovingAverageBaseline,
    inverse_target,
    EPOCHS,
    BATCH_SIZE,
    PATIENCE,
)
from transformer import build_transformer

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features", "geo")
MODELS_DIR   = os.path.join(BASE_DIR, "models", "geo")
os.makedirs(MODELS_DIR, exist_ok=True)

SEED = 42


# ── Callbacks ────────────────────────────────────────────────────────────────
def get_callbacks_geo(slug: str):
    ckpt = os.path.join(MODELS_DIR, f"{slug}_transformer_best.keras")
    return [
        callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                restore_best_weights=True, verbose=0),
        callbacks.ModelCheckpoint(filepath=ckpt, monitor="val_loss",
                                  save_best_only=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=10, min_lr=1e-6, verbose=0),
    ]


# ── Évaluation ───────────────────────────────────────────────────────────────
def metrics(y_true_sc, y_pred_sc, scaler):
    y_true = inverse_target(y_true_sc, scaler)
    y_pred = inverse_target(y_pred_sc, scaler)
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    return mae, rmse, mape


# ── Entraînement par quartier ────────────────────────────────────────────────
def train_quartier(slug: str):
    feat_dir = os.path.join(FEATURES_DIR, slug)
    X_train = np.load(os.path.join(feat_dir, "X_train.npy"))
    y_train = np.load(os.path.join(feat_dir, "y_train.npy"))
    X_val   = np.load(os.path.join(feat_dir, "X_val.npy"))
    y_val   = np.load(os.path.join(feat_dir, "y_val.npy"))
    X_test  = np.load(os.path.join(feat_dir, "X_test.npy"))
    y_test  = np.load(os.path.join(feat_dir, "y_test.npy"))

    with open(os.path.join(feat_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    keras.utils.set_random_seed(SEED)

    input_shape = (X_train.shape[1], X_train.shape[2])

    # MovAvg baseline (sur même test que LSTM par quartier — comparable)
    ma = MovingAverageBaseline(k=3)
    y_pred_ma = ma.predict(X_test)
    mae_ma, rmse_ma, mape_ma = metrics(y_test, y_pred_ma, scaler)

    # Transformer
    model = build_transformer(input_shape)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks_geo(slug),
        verbose=0,
    )

    y_pred = model.predict(X_test, verbose=0).ravel()
    mae_t, rmse_t, mape_t = metrics(y_test, y_pred, scaler)

    np.save(os.path.join(MODELS_DIR, f"{slug}_y_pred_transformer.npy"), y_pred)

    return {
        "slug": slug,
        "n_test": len(y_test),
        "transformer_mae": mae_t, "transformer_rmse": rmse_t, "transformer_mape": mape_t,
        "ma_mae": mae_ma,  "ma_rmse": rmse_ma,  "ma_mape": mape_ma,
        "epochs_trained": len(history.history["loss"]),
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    quartiers = sorted([d for d in os.listdir(FEATURES_DIR)
                        if os.path.isdir(os.path.join(FEATURES_DIR, d))])
    print(f"🧠 Entraînement Transformer par quartier ({len(quartiers)} quartiers)")
    print(f"   d_model=32, num_heads=4, key_dim=8, FFN=64, "
          f"epochs<={EPOCHS}, patience={PATIENCE}, seed={SEED}\n")

    results = []
    for slug in quartiers:
        print(f"  → {slug}...", flush=True)
        info = train_quartier(slug)
        results.append(info)
        print(f"     Transformer MAE={info['transformer_mae']:5.0f}  "
              f"RMSE={info['transformer_rmse']:5.0f}  "
              f"MAPE={info['transformer_mape']:4.1f}%   "
              f"({info['epochs_trained']} epochs)")
        print(f"     MovAvg     MAE={info['ma_mae']:5.0f}  "
              f"RMSE={info['ma_rmse']:5.0f}  MAPE={info['ma_mape']:4.1f}%")

    df_metrics = pd.DataFrame(results)
    df_metrics["transformer_beats_ma"] = df_metrics["transformer_mae"] < df_metrics["ma_mae"]
    metrics_path = os.path.join(MODELS_DIR, "transformer_metrics.csv")
    df_metrics.to_csv(metrics_path, index=False)

    print(f"\n📊 Récap (MAE en €/m²) :")
    print(df_metrics[["slug", "transformer_mae", "ma_mae", "transformer_mape",
                      "transformer_beats_ma"]].to_string(index=False))
    n_win = df_metrics['transformer_beats_ma'].sum()
    print(f"\n   Transformer bat la baseline sur {n_win}/{len(df_metrics)} quartiers")

    print(f"\n💾 Sauvegarde :")
    print(f"   {os.path.relpath(MODELS_DIR, BASE_DIR)}/{{slug}}_transformer_best.keras")
    print(f"   {os.path.relpath(metrics_path, BASE_DIR)}")
    print("\n✅  transformer_geo.py terminé.")


if __name__ == "__main__":
    main()
