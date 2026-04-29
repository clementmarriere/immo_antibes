"""
src/models/lstm_geo.py
=======================
Entraîne un LSTM par quartier d'Antibes (6 modèles indépendants).

Pour chaque quartier :
  - charge les arrays préparés par build_features_geo.py
  - entraîne LSTM(64→32) avec mêmes hyperparams que lstm.py
  - évalue sur test set : MAE / RMSE / MAPE
  - compare à baseline MovingAverage(k=3)
  - sauvegarde poids + prédictions + métriques

Sortie :
  - models/geo/{quartier}_best.keras
  - models/geo/metrics.csv
  - models/geo/{quartier}_y_pred_lstm.npy

Usage :
  python src/models/lstm_geo.py
"""

import numpy as np
import pandas as pd
import pickle
import os
import sys

import tensorflow as tf
from tensorflow import keras

# Import des helpers du LSTM global (même architecture)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lstm import (
    build_lstm,
    MovingAverageBaseline,
    inverse_target,
    EPOCHS,
    BATCH_SIZE,
    PATIENCE,
)
from tensorflow.keras import callbacks

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features", "geo")
MODELS_DIR   = os.path.join(BASE_DIR, "models", "geo")
os.makedirs(MODELS_DIR, exist_ok=True)

SEED = 42


# ── Callbacks dédiés (checkpoint par quartier) ───────────────────────────────
def get_callbacks_geo(slug: str):
    ckpt_path = os.path.join(MODELS_DIR, f"{slug}_best.keras")
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=0,
        ),
        callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=0,
        ),
    ]


# ── Évaluation ───────────────────────────────────────────────────────────────
def metrics(y_true_sc, y_pred_sc, scaler):
    y_true = inverse_target(y_true_sc, scaler)
    y_pred = inverse_target(y_pred_sc, scaler)
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    return mae, rmse, mape, y_true, y_pred


# ── Entraînement d'un quartier ───────────────────────────────────────────────
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

    # Reproductibilité
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    keras.utils.set_random_seed(SEED)

    input_shape = (X_train.shape[1], X_train.shape[2])

    # Baseline MovingAverage
    ma = MovingAverageBaseline(k=3)
    y_pred_ma = ma.predict(X_test)
    mae_ma, rmse_ma, mape_ma, _, _ = metrics(y_test, y_pred_ma, scaler)

    # LSTM
    model = build_lstm(input_shape)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks_geo(slug),
        verbose=0,
    )

    y_pred_lstm = model.predict(X_test, verbose=0).ravel()
    mae_l, rmse_l, mape_l, y_true, y_pred = metrics(y_test, y_pred_lstm, scaler)

    np.save(os.path.join(MODELS_DIR, f"{slug}_y_pred_lstm.npy"), y_pred_lstm)
    np.save(os.path.join(MODELS_DIR, f"{slug}_y_pred_ma.npy"),  y_pred_ma)
    np.save(os.path.join(MODELS_DIR, f"{slug}_y_test.npy"),      y_test)

    return {
        "slug": slug,
        "n_test": len(y_test),
        "lstm_mae": mae_l, "lstm_rmse": rmse_l, "lstm_mape": mape_l,
        "ma_mae": mae_ma,  "ma_rmse": rmse_ma,  "ma_mape": mape_ma,
        "epochs_trained": len(history.history["loss"]),
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Listes des quartiers à traiter (slugs créés par build_features_geo.py)
    quartiers = sorted([d for d in os.listdir(FEATURES_DIR)
                        if os.path.isdir(os.path.join(FEATURES_DIR, d))])
    print(f"🧠 Entraînement LSTM par quartier ({len(quartiers)} quartiers)")
    print(f"   Hyperparams : LSTM(64,32), dropout=0.2, epochs<={EPOCHS}, "
          f"patience={PATIENCE}, seed={SEED}\n")

    results = []
    for slug in quartiers:
        print(f"  → {slug}...", flush=True)
        info = train_quartier(slug)
        results.append(info)
        print(f"     LSTM   MAE={info['lstm_mae']:5.0f}  RMSE={info['lstm_rmse']:5.0f}  "
              f"MAPE={info['lstm_mape']:4.1f}%   ({info['epochs_trained']} epochs)")
        print(f"     MovAvg MAE={info['ma_mae']:5.0f}  RMSE={info['ma_rmse']:5.0f}  "
              f"MAPE={info['ma_mape']:4.1f}%")

    # Tableau récap
    df_metrics = pd.DataFrame(results)
    df_metrics["lstm_beats_ma"] = df_metrics["lstm_mae"] < df_metrics["ma_mae"]
    metrics_path = os.path.join(MODELS_DIR, "metrics.csv")
    df_metrics.to_csv(metrics_path, index=False)

    print(f"\n📊 Récap (MAE en €/m²) :")
    print(df_metrics[["slug", "lstm_mae", "ma_mae", "lstm_mape", "lstm_beats_ma"]]
          .to_string(index=False))
    print(f"\n   LSTM bat la baseline sur {df_metrics['lstm_beats_ma'].sum()}/{len(df_metrics)} quartiers")

    print(f"\n💾 Sauvegarde :")
    print(f"   {os.path.relpath(MODELS_DIR, BASE_DIR)}/{{slug}}_best.keras")
    print(f"   {os.path.relpath(metrics_path, BASE_DIR)}")
    print("\n✅  lstm_geo.py terminé. Prochaine étape : src/models/forecast_geo.py")


if __name__ == "__main__":
    main()
