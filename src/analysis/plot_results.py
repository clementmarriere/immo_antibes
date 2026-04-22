"""
src/analysis/plot_results.py
=============================
Visualisations post-entraînement :
  1. Courbes de loss LSTM (train vs val)
  2. Prédictions vs Réalité (LSTM + MovingAvg sur test set)
  3. Erreurs résiduelles LSTM
  4. Tableau comparatif des métriques

Usage :
  python src/analysis/plot_results.py
"""

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
OUTPUT_DIR   = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
COLOR_LSTM   = "#2563EB"   # bleu
COLOR_MA     = "#16A34A"   # vert
COLOR_REAL   = "#111827"   # noir
COLOR_MLP    = "#DC2626"   # rouge
COLOR_ERROR  = "#F59E0B"   # orange
COLOR_FILL   = "#DBEAFE"   # bleu clair


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_all():
    with open(os.path.join(FEATURES_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    def inv(y):
        nb = scaler.n_features_in_
        dummy = np.zeros((len(y), nb), dtype=np.float32)
        dummy[:, 0] = y.ravel()
        return scaler.inverse_transform(dummy)[:, 0]

    data = {
        "y_test"      : inv(np.load(os.path.join(MODELS_DIR, "y_test.npy"))),
        "y_pred_lstm" : inv(np.load(os.path.join(MODELS_DIR, "y_pred_lstm.npy"))),
        "y_pred_ma"   : inv(np.load(os.path.join(MODELS_DIR, "y_pred_ma.npy"))),
        "y_pred_mlp"  : inv(np.load(os.path.join(MODELS_DIR, "y_pred_mlp.npy"))),
        "loss_lstm"   : np.load(os.path.join(MODELS_DIR, "history_lstm_loss.npy")),
        "val_lstm"    : np.load(os.path.join(MODELS_DIR, "history_lstm_val.npy")),
        "loss_mlp"    : np.load(os.path.join(MODELS_DIR, "history_mlp_loss.npy")),
        "val_mlp"     : np.load(os.path.join(MODELS_DIR, "history_mlp_val.npy")),
    }
    return data

def metrics(y_true, y_pred, label):
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    print(f"  [{label:20s}]  MAE={mae:.0f} €/m²   RMSE={rmse:.0f} €/m²   MAPE={mape:.1f}%")
    return mae, rmse, mape


# ── Figure 1 : Courbes de Loss ────────────────────────────────────────────────
def plot_loss_curves(data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Courbes d'apprentissage", fontsize=14, fontweight="bold", y=1.01)

    # LSTM
    ax = axes[0]
    epochs_lstm = range(1, len(data["loss_lstm"]) + 1)
    ax.plot(epochs_lstm, data["loss_lstm"], color=COLOR_LSTM,  label="Train loss", linewidth=2)
    ax.plot(epochs_lstm, data["val_lstm"],  color=COLOR_ERROR, label="Val loss",   linewidth=2, linestyle="--")
    best_epoch = int(np.argmin(data["val_lstm"])) + 1
    ax.axvline(best_epoch, color="gray", linestyle=":", alpha=0.7, label=f"Best epoch ({best_epoch})")
    ax.set_title("LSTM", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MLP
    ax = axes[1]
    epochs_mlp = range(1, len(data["loss_mlp"]) + 1)
    ax.plot(epochs_mlp, data["loss_mlp"], color=COLOR_MLP,   label="Train loss", linewidth=2)
    ax.plot(epochs_mlp, data["val_mlp"],  color=COLOR_ERROR, label="Val loss",   linewidth=2, linestyle="--")
    best_epoch_mlp = int(np.argmin(data["val_mlp"])) + 1
    ax.axvline(best_epoch_mlp, color="gray", linestyle=":", alpha=0.7, label=f"Best epoch ({best_epoch_mlp})")
    ax.set_title("MLP (baseline deep)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_loss_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 {path}")


# ── Figure 2 : Prédictions vs Réalité ────────────────────────────────────────
def plot_predictions(data):
    y_true = data["y_test"]
    n      = len(y_true)
    x      = np.arange(n)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(x, y_true,            color=COLOR_REAL, linewidth=2.5, label="Réalité",          marker="o", markersize=5)
    ax.plot(x, data["y_pred_lstm"], color=COLOR_LSTM,  linewidth=2,   label="LSTM",             marker="s", markersize=4, linestyle="--")
    ax.plot(x, data["y_pred_ma"],   color=COLOR_MA,    linewidth=2,   label="Moyenne Mobile k=3", marker="^", markersize=4, linestyle="--")
    ax.plot(x, data["y_pred_mlp"],  color=COLOR_MLP,   linewidth=1.5, label="MLP",              marker="x", markersize=4, linestyle=":")

    # Zone d'erreur LSTM
    ax.fill_between(x, y_true, data["y_pred_lstm"],
                    alpha=0.08, color=COLOR_LSTM, label="Écart LSTM")

    ax.set_title("Prédictions vs Réalité — Test Set (11 mois)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Index temporel (mois du test set)")
    ax.set_ylabel("Prix/m² médian (€)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Annotation MAE
    mae_lstm = np.mean(np.abs(y_true - data["y_pred_lstm"]))
    mae_ma   = np.mean(np.abs(y_true - data["y_pred_ma"]))
    ax.text(0.98, 0.05,
            f"MAE LSTM : {mae_lstm:.0f} €/m²\nMAE MovAvg : {mae_ma:.0f} €/m²",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_predictions_vs_reality.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 {path}")


# ── Figure 3 : Résidus LSTM ───────────────────────────────────────────────────
def plot_residuals(data):
    y_true    = data["y_test"]
    res_lstm  = data["y_pred_lstm"] - y_true
    res_ma    = data["y_pred_ma"]   - y_true
    x         = np.arange(len(y_true))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Analyse des résidus (erreur = prédiction − réalité)", fontsize=13, fontweight="bold")

    for ax, res, color, label in [
        (axes[0], res_lstm, COLOR_LSTM, "LSTM"),
        (axes[1], res_ma,   COLOR_MA,   "Moyenne Mobile k=3"),
    ]:
        ax.bar(x, res, color=color, alpha=0.75, label=label)
        ax.axhline(0,               color="black", linewidth=1)
        ax.axhline(np.mean(res),    color="red",   linewidth=1.5, linestyle="--", label=f"Biais moyen : {np.mean(res):.0f} €/m²")
        ax.fill_between(x, res, 0, where=(res > 0), color=color, alpha=0.15)
        ax.fill_between(x, res, 0, where=(res < 0), color="red", alpha=0.10)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Index temporel")
        ax.set_ylabel("Résidu (€/m²)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_residuals.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 {path}")


# ── Figure 4 : Tableau comparatif ─────────────────────────────────────────────
def plot_metrics_table(data):
    y_true = data["y_test"]

    rows = []
    for label, y_pred in [
        ("LSTM",              data["y_pred_lstm"]),
        ("Moyenne Mobile k=3", data["y_pred_ma"]),
        ("MLP",               data["y_pred_mlp"]),
    ]:
        mae  = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        rows.append([label, f"{mae:.0f} €/m²", f"{rmse:.0f} €/m²", f"{mape:.1f}%"])

    fig, ax = plt.subplots(figsize=(9, 2.5))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=["Modèle", "MAE", "RMSE", "MAPE"],
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)

    # Couleurs
    colors_header = ["#1E3A5F"] * 4
    for j, c in enumerate(colors_header):
        table[0, j].set_facecolor(c)
        table[0, j].set_text_props(color="white", fontweight="bold")

    row_colors = ["#DBEAFE", "#DCFCE7", "#FEE2E2"]
    for i, rc in enumerate(row_colors):
        for j in range(4):
            table[i+1, j].set_facecolor(rc)

    ax.set_title("Comparaison des modèles — Test Set", fontsize=13,
                 fontweight="bold", pad=20)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_metrics_table.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("📂 Chargement des résultats...")
    data = load_all()

    print("\n📊 Métriques :")
    metrics(data["y_test"], data["y_pred_lstm"], "LSTM")
    metrics(data["y_test"], data["y_pred_ma"],   "Moyenne Mobile k=3")
    metrics(data["y_test"], data["y_pred_mlp"],  "MLP")

    print("\n🎨 Génération des figures...")
    plot_loss_curves(data)
    plot_predictions(data)
    plot_residuals(data)
    plot_metrics_table(data)

    print(f"\n✅ 4 figures sauvegardées dans reports/figures/")
    print("   01_loss_curves.png")
    print("   02_predictions_vs_reality.png")
    print("   03_residuals.png")
    print("   04_metrics_table.png")
    print("\n➡️  Prochaine étape : src/scoring/score.py")


if __name__ == "__main__":
    main()