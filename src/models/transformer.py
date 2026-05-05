"""
src/models/transformer.py
==========================
Encoder Transformer pour la prédiction du prix/m² mensuel à Antibes.

Architecture inspirée de "Attention Is All You Need" (Vaswani 2017),
adaptée à des séries temporelles courtes (12 mois × 6 features) :

  Input(12, 6)
    → Dense(d_model=32)                    (projection linéaire)
    → + Positional Encoding (sinusoïdal)
    → MultiHeadAttention(heads=4, key_dim=8)  (residual + LayerNorm)
    → FFN: Dense(64, relu) → Dropout → Dense(d_model)  (residual + LayerNorm)
    → GlobalAveragePooling1D                (pooling temporel)
    → Dense(1)

Contrairement au LSTM/GRU qui traite la séquence pas à pas, le Transformer
voit toute la fenêtre simultanément via l'attention — utile pour capturer
des dépendances longue portée (ex: corrélation prix juin-année-N / juin-année-N+1).

Usage :
  python src/models/transformer.py
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
    inverse_target,
    evaluate,
    EPOCHS,
    BATCH_SIZE,
    PATIENCE,
    LEARNING_RATE,
    DROPOUT_RATE,
)

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Hyperparamètres Transformer ──────────────────────────────────────────────
D_MODEL     = 32     # dim de projection
NUM_HEADS   = 4
KEY_DIM     = 8      # dim par tête (NUM_HEADS × KEY_DIM = 32 = D_MODEL)
FFN_UNITS   = 64
NUM_BLOCKS  = 1      # 1 bloc d'attention suffit sur série courte


# ── Positional encoding sinusoïdal ───────────────────────────────────────────
def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Encodage positionnel sinusoïdal classique (Vaswani 2017).
    Permet au Transformer d'avoir une notion d'ordre temporel.
    """
    pos = np.arange(seq_len)[:, np.newaxis]
    i   = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1.0 / np.power(10000.0, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    # Sin sur indices pairs, cos sur impairs
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return angle_rads.astype(np.float32)


# ── Bloc encoder ─────────────────────────────────────────────────────────────
def transformer_block(x, num_heads: int, key_dim: int,
                      ffn_units: int, dropout: float):
    """Bloc encoder = attention multi-tête + FFN, chacun avec residual + LayerNorm."""
    # Multi-Head Self-Attention
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, dropout=dropout
    )(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)

    # Feed-Forward Network
    ffn = layers.Dense(ffn_units, activation="relu")(x)
    ffn = layers.Dropout(dropout)(ffn)
    ffn = layers.Dense(x.shape[-1])(ffn)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)
    return x


# ── Architecture complète ────────────────────────────────────────────────────
def build_transformer(input_shape: tuple) -> keras.Model:
    seq_len, n_features = input_shape

    inputs = keras.Input(shape=input_shape)

    # Projection linéaire vers d_model
    x = layers.Dense(D_MODEL)(inputs)

    # Ajout du positional encoding (constant, non entraîné)
    pos_enc = positional_encoding(seq_len, D_MODEL)
    pos_enc_layer = layers.Lambda(
        lambda t: t + tf.constant(pos_enc, dtype=tf.float32),
        output_shape=(seq_len, D_MODEL),
    )
    x = pos_enc_layer(x)

    # Blocs Transformer
    for _ in range(NUM_BLOCKS):
        x = transformer_block(x, NUM_HEADS, KEY_DIM, FFN_UNITS, DROPOUT_RATE)

    # Pooling temporel + tête de régression
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs, outputs, name="Transformer_ImmoAntibes")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model


def get_callbacks(model_name: str = "transformer"):
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

    print("🧠 Entraînement Transformer...")
    model = build_transformer(input_shape)
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks("transformer"),
        verbose=1,
    )

    y_pred = model.predict(X_test).ravel()
    print("\n📊 Résultats sur Test set :")
    evaluate(y_test, y_pred, scaler, "Transformer")

    np.save(os.path.join(MODELS_DIR, "y_pred_transformer.npy"), y_pred)
    np.save(os.path.join(MODELS_DIR, "history_transformer_loss.npy"),
            np.array(history.history["loss"]))
    np.save(os.path.join(MODELS_DIR, "history_transformer_val.npy"),
            np.array(history.history["val_loss"]))

    print(f"\n💾 Sauvegarde : models/transformer_best.keras + prédictions")
    print("\n✅  transformer.py terminé.")


if __name__ == "__main__":
    main()
