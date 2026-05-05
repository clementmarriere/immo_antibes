"""
src/analysis/heatmap_forecast_compare.py
==========================================
Cartes thermiques prédictives 2026 — LSTM vs Transformer côte à côte.

Génère :
  15_heatmap_forecast_compare.png    → 2 panneaux LSTM | Transformer
  16_growth_2026_compare.png         → barplot croissance moyenne 2026 par quartier × modèle

Lit :
  reports/forecast_2026_geo.csv             (LSTM, généré par forecast_geo.py)
  reports/forecast_2026_transformer_geo.csv (Transformer, généré par forecast_transformer_geo.py)

Usage :
  python src/analysis/heatmap_forecast_compare.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_LSTM   = os.path.join(BASE_DIR, "reports", "forecast_2026_geo.csv")
INPUT_TFR    = os.path.join(BASE_DIR, "reports", "forecast_2026_transformer_geo.csv")
OUTPUT_DIR   = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

QUARTIER_ORDER = [
    "Vieille Ville",
    "Cap d'Antibes",
    "Centre Ville",
    "Juan-les-Pins",
    "La Fontonne",
    "Antibes Nord Ouest",
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def pivot_growth(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot(index="quartier", columns="periode", values="growth_pct_vs_dec25")
    pivot = pivot.reindex([q for q in QUARTIER_ORDER if q in pivot.index])
    pivot = pivot[sorted(pivot.columns)]
    return pivot


def plot_heatmap_panel(ax, pivot, title, vabs):
    norm = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    im = ax.imshow(pivot.values, cmap="RdBu_r", norm=norm, aspect="auto")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            color = "white" if abs(val) > vabs * 0.5 else "black"
            ax.text(j, i, f"{val:+.0f}", ha="center", va="center",
                    color=color, fontsize=8, fontweight="bold")
    months_short = [pd.Timestamp(c).strftime("%b") for c in pivot.columns]
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(months_short, fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Mois 2026", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    return im


# ── Figure 15 : 2 heatmaps côte à côte ───────────────────────────────────────
def plot_compare_heatmaps(df_lstm, df_tfr):
    p_lstm = pivot_growth(df_lstm)
    p_tfr  = pivot_growth(df_tfr)

    # Échelle commune entre les deux pour comparaison directe
    vabs = max(np.nanmax(np.abs(p_lstm.values)), np.nanmax(np.abs(p_tfr.values)))

    fig, axes = plt.subplots(1, 2, figsize=(20, 5), sharey=True)
    im1 = plot_heatmap_panel(axes[0], p_lstm,
                             "LSTM — Croissance prix/m² 2026 (% vs déc 2025)", vabs)
    im2 = plot_heatmap_panel(axes[1], p_tfr,
                             "Transformer — Croissance prix/m² 2026 (% vs déc 2025)", vabs)

    cbar = fig.colorbar(im2, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Croissance vs déc 2025 (%)", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))

    fig.suptitle("Carte thermique prédictive 2026 — LSTM vs Transformer (par quartier)\n"
                 "6 modèles indépendants par architecture, mêmes splits, seed=42",
                 fontsize=13, fontweight="bold", y=1.02)

    path = os.path.join(OUTPUT_DIR, "15_heatmap_forecast_compare.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── Figure 16 : barplot croissance moyenne 2026 par quartier × modèle ────────
def plot_growth_summary(df_lstm, df_tfr):
    summary_lstm = df_lstm.groupby("quartier")["growth_pct_vs_dec25"].mean()
    summary_tfr  = df_tfr.groupby("quartier")["growth_pct_vs_dec25"].mean()

    quartiers = [q for q in QUARTIER_ORDER if q in summary_lstm.index]
    x = np.arange(len(quartiers))
    width = 0.4

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width/2, [summary_lstm[q] for q in quartiers],
           width, label="LSTM",        color="#2563EB", alpha=0.85)
    ax.bar(x + width/2, [summary_tfr[q]  for q in quartiers],
           width, label="Transformer", color="#DC2626", alpha=0.85)

    # Annotations
    for xi, q in zip(x, quartiers):
        v_l = summary_lstm[q]
        v_t = summary_tfr[q]
        ax.annotate(f"{v_l:+.1f}%", xy=(xi - width/2, v_l),
                    xytext=(0, 3 if v_l >= 0 else -12),
                    textcoords="offset points", ha="center", fontsize=8,
                    color="#2563EB", fontweight="bold")
        ax.annotate(f"{v_t:+.1f}%", xy=(xi + width/2, v_t),
                    xytext=(0, 3 if v_t >= 0 else -12),
                    textcoords="offset points", ha="center", fontsize=8,
                    color="#DC2626", fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(quartiers, rotation=15, ha="right")
    ax.set_ylabel("Croissance moyenne 2026 vs déc 2025 (%)")
    ax.set_title("Forecast 2026 — croissance moyenne annuelle par quartier (LSTM vs Transformer)\n"
                 "Les écarts entre architectures révèlent la sensibilité du forecast au modèle choisi",
                 fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "16_growth_2026_compare.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(INPUT_LSTM):
        raise FileNotFoundError(f"Manquant : {INPUT_LSTM}\n"
                                "  → python src/models/forecast_geo.py")
    if not os.path.exists(INPUT_TFR):
        raise FileNotFoundError(f"Manquant : {INPUT_TFR}\n"
                                "  → python src/models/forecast_transformer_geo.py")

    df_lstm = pd.read_csv(INPUT_LSTM)
    df_tfr  = pd.read_csv(INPUT_TFR)
    print(f"📂 LSTM        : {len(df_lstm)} lignes, "
          f"{df_lstm['quartier'].nunique()} quartiers")
    print(f"📂 Transformer : {len(df_tfr)} lignes, "
          f"{df_tfr['quartier'].nunique()} quartiers\n")

    print("🎨 Génération des figures :")
    p1 = plot_compare_heatmaps(df_lstm, df_tfr)
    p2 = plot_growth_summary(df_lstm, df_tfr)
    print(f"  ✓ {os.path.relpath(p1, BASE_DIR)}")
    print(f"  ✓ {os.path.relpath(p2, BASE_DIR)}")

    print("\n✅  heatmap_forecast_compare.py terminé.")


if __name__ == "__main__":
    main()
