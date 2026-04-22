"""
src/models/lstm.py
==================
Architecture LSTM pour la prédiction du prix/m² mensuel à Antibes.

Contenu :
  - build_lstm()     → modèle Keras 2 couches LSTM + Dense
  - build_mlp()      → baseline MLP dense (comparaison)
  - MovingAverage    → baseline statistique (moyenne mobile)
  - evaluate()       → calcul MAE / RMSE sur prédictions dénormalisées

Usage (direct) :
  python src/models/lstm.py          # entraîne + évalue + sauvegarde
"""

import numpy as np
import pickle
import os
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Hyperparamètres ───────────────────────────────────────────────────────────
LSTM_UNITS_1  = 64
LSTM_UNITS_2  = 32
DROPOUT_RATE  = 0.2
LEARNING_RATE = 1e-3
EPOCHS        = 200
BATCH_SIZE    = 16
PATIENCE      = 20       # early stopping


# ── 1. Architecture LSTM ──────────────────────────────────────────────────────
def build_lstm(input_shape: tuple) -> keras.Model:
    """
    input_shape : (n_steps, n_features)  ex: (12, 6)

    Architecture :
      LSTM(64, return_sequences=True)
        → Dropout(0.2)
      LSTM(32, return_sequences=False)
        → Dropout(0.2)
      Dense(16, relu)
        → Dense(1)
    """
    model = keras.Sequential([
        keras.Input(shape=input_shape),

        layers.LSTM(LSTM_UNITS_1, return_sequences=True),
        layers.Dropout(DROPOUT_RATE),

        layers.LSTM(LSTM_UNITS_2, return_sequences=False),
        layers.Dropout(DROPOUT_RATE),

        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ], name="LSTM_ImmoAntibes")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"]
    )
    return model


# ── 2. Architecture MLP (baseline deep) ──────────────────────────────────────
def build_mlp(input_shape: tuple) -> keras.Model:
    """
    MLP baseline : aplatit la fenêtre temporelle et passe dans 3 couches denses.
    input_shape : (n_steps, n_features)
    """
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(32, activation="relu"),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ], name="MLP_ImmoAntibes")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"]
    )
    return model


# ── 3. Baseline Moyenne Mobile ────────────────────────────────────────────────
class MovingAverageBaseline:
    """
    Prédit le prochain mois = moyenne des k derniers mois de la fenêtre.
    Travaille dans l'espace normalisé (colonne 0 = prix_m2_median).
    """
    def __init__(self, k: int = 3):
        self.k = k

    def predict(self, X: np.ndarray) -> np.ndarray:
        # X : (N, n_steps, n_features) → moyenne des k derniers pas sur col 0
        return X[:, -self.k:, 0].mean(axis=1)

    def __repr__(self):
        return f"MovingAverageBaseline(k={self.k})"


# ── 4. Métriques ──────────────────────────────────────────────────────────────
def inverse_target(y_scaled: np.ndarray, scaler) -> np.ndarray:
    nb_features = scaler.n_features_in_
    dummy = np.zeros((len(y_scaled), nb_features), dtype=np.float32)
    dummy[:, 0] = y_scaled.ravel()
    return scaler.inverse_transform(dummy)[:, 0]

def evaluate(y_true_sc: np.ndarray, y_pred_sc: np.ndarray,
             scaler, label: str = ""):
    y_true = inverse_target(y_true_sc, scaler)
    y_pred = inverse_target(y_pred_sc, scaler)

    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    print(f"  [{label:20s}]  MAE={mae:.0f} €/m²   RMSE={rmse:.0f} €/m²   MAPE={mape:.1f}%")
    return {"mae": mae, "rmse": rmse, "mape": mape,
            "y_true": y_true, "y_pred": y_pred}


# ── 5. Training callbacks ─────────────────────────────────────────────────────
def get_callbacks(model_name: str):
    ckpt_path = os.path.join(MODELS_DIR, f"{model_name}_best.keras")
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
    ]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Chargement des arrays
    print("📂 Chargement des features...")
    X_train = np.load(os.path.join(FEATURES_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(FEATURES_DIR, "y_train.npy"))
    X_val   = np.load(os.path.join(FEATURES_DIR, "X_val.npy"))
    y_val   = np.load(os.path.join(FEATURES_DIR, "y_val.npy"))
    X_test  = np.load(os.path.join(FEATURES_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(FEATURES_DIR, "y_test.npy"))

    with open(os.path.join(FEATURES_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"   input_shape = {input_shape}")
    print(f"   train={len(X_train)}  val={len(X_val)}  test={len(X_test)}\n")

    # ── Baseline Moyenne Mobile ───────────────────────────────────────────────
    print("📏 Baseline : Moyenne Mobile (k=3)")
    ma = MovingAverageBaseline(k=3)
    y_pred_ma_test = ma.predict(X_test)
    results_ma = evaluate(y_test, y_pred_ma_test, scaler, "MovingAvg k=3")

    # ── LSTM ──────────────────────────────────────────────────────────────────
    print("\n🧠 Entraînement LSTM...")
    lstm = build_lstm(input_shape)
    lstm.summary()

    history_lstm = lstm.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks("lstm"),
        verbose=1
    )

    y_pred_lstm_test = lstm.predict(X_test).ravel()
    print("\n📊 Résultats sur Test set :")
    results_lstm = evaluate(y_test, y_pred_lstm_test, scaler, "LSTM")
    results_ma_ref = evaluate(y_test, y_pred_ma_test, scaler, "MovingAvg k=3")

    # ── MLP ───────────────────────────────────────────────────────────────────
    print("\n🧠 Entraînement MLP (baseline deep)...")
    mlp = build_mlp(input_shape)

    history_mlp = mlp.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks("mlp"),
        verbose=1
    )

    y_pred_mlp_test = mlp.predict(X_test).ravel()
    print("\n📊 Résultats MLP sur Test set :")
    results_mlp = evaluate(y_test, y_pred_mlp_test, scaler, "MLP")

    # ── Sauvegarde des résultats pour visualisation ───────────────────────────
    print("\n💾 Sauvegarde...")

    np.save(os.path.join(MODELS_DIR, "y_test.npy"),          y_test)
    np.save(os.path.join(MODELS_DIR, "y_pred_lstm.npy"),     y_pred_lstm_test)
    np.save(os.path.join(MODELS_DIR, "y_pred_mlp.npy"),      y_pred_mlp_test)
    np.save(os.path.join(MODELS_DIR, "y_pred_ma.npy"),       y_pred_ma_test)
    np.save(os.path.join(MODELS_DIR, "history_lstm_loss.npy"), np.array(history_lstm.history["loss"]))
    np.save(os.path.join(MODELS_DIR, "history_lstm_val.npy"),  np.array(history_lstm.history["val_loss"]))
    np.save(os.path.join(MODELS_DIR, "history_mlp_loss.npy"),  np.array(history_mlp.history["loss"]))
    np.save(os.path.join(MODELS_DIR, "history_mlp_val.npy"),   np.array(history_mlp.history["val_loss"]))

    print(f"   → models/ : poids + historiques + prédictions")
    print("\n✅  lstm.py terminé. Prochaine étape : src/models/train.py  (ou analyse des résultats)")


if __name__ == "__main__":
    main()