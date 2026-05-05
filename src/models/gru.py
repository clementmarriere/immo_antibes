"""
src/models/gru.py
==================
Architecture GRU pour la prédiction du prix/m² mensuel à Antibes.

Variante du LSTM avec moins de paramètres (~25%) — utile sur petit dataset.
Mêmes hyperparams que lstm.py pour comparaison directe.

Contenu :
  - build_gru()       → modèle Keras 2 couches GRU + Dense
  - main()            → entraîne + évalue + sauvegarde

Usage :
  python src/models/gru.py
"""

import os
import sys
import pickle
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lstm import (
    MovingAverageBaseline,
    inverse_target,
    evaluate,
    EPOCHS,
    BATCH_SIZE,
    PATIENCE,
    LEARNING_RATE,
    DROPOUT_RATE,
    LSTM_UNITS_1 as GRU_UNITS_1,
    LSTM_UNITS_2 as GRU_UNITS_2,
)

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Architecture GRU ─────────────────────────────────────────────────────────
def build_gru(input_shape: tuple) -> keras.Model:
    """
    Architecture symétrique au LSTM, avec couches GRU.

    GRU(64, return_sequences=True)
      → Dropout(0.2)
    GRU(32, return_sequences=False)
      → Dropout(0.2)
    Dense(16, relu)
      → Dense(1)
    """
    model = keras.Sequential([
        keras.Input(shape=input_shape),

        layers.GRU(GRU_UNITS_1, return_sequences=True),
        layers.Dropout(DROPOUT_RATE),

        layers.GRU(GRU_UNITS_2, return_sequences=False),
        layers.Dropout(DROPOUT_RATE),

        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ], name="GRU_ImmoAntibes")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model


def get_callbacks(model_name: str = "gru"):
    ckpt_path = os.path.join(MODELS_DIR, f"{model_name}_best.keras")
    return [
        callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(filepath=ckpt_path, monitor="val_loss",
                                  save_best_only=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=10, min_lr=1e-6, verbose=1),
    ]


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("📂 Chargement des features...")
    X_train = np.load(os.path.join(FEATURES_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(FEATURES_DIR, "y_train.npy"))
    X_val   = np.load(os.path.join(FEATURES_DIR, "X_val.npy"))
    y_val   = np.load(os.path.join(FEATURES_DIR, "y_val.npy"))
    X_test  = np.load(os.path.join(FEATURES_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(FEATURES_DIR, "y_test.npy"))

    with open(os.path.join(FEATURES_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    np.random.seed(42)
    tf.random.set_seed(42)
    keras.utils.set_random_seed(42)

    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"   input_shape = {input_shape}")
    print(f"   train={len(X_train)}  val={len(X_val)}  test={len(X_test)}\n")

    print("🧠 Entraînement GRU...")
    gru = build_gru(input_shape)
    gru.summary()

    history = gru.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks("gru"),
        verbose=1,
    )

    y_pred = gru.predict(X_test).ravel()
    print("\n📊 Résultats sur Test set :")
    evaluate(y_test, y_pred, scaler, "GRU")

    np.save(os.path.join(MODELS_DIR, "y_pred_gru.npy"), y_pred)
    np.save(os.path.join(MODELS_DIR, "history_gru_loss.npy"),
            np.array(history.history["loss"]))
    np.save(os.path.join(MODELS_DIR, "history_gru_val.npy"),
            np.array(history.history["val_loss"]))

    print(f"\n💾 Sauvegarde : models/gru_best.keras + prédictions")
    print("\n✅  gru.py terminé.")


if __name__ == "__main__":
    main()
