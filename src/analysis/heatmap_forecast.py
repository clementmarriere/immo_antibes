"""
src/analysis/heatmap_forecast.py
=================================
Carte thermique prédictive 2026 : projection LSTM par quartier × mois.

Génère deux figures :
  11_heatmap_forecast_2026.png  → heatmap (quartier × mois 2026), couleur = croissance % vs déc 2025
  12_trajectoires_forecast_2026.png → courbes de prix prédits par quartier sur 2026 (avec IC MC Dropout)

Lit :
  reports/forecast_2026_geo.csv  (généré par forecast_geo.py)

Usage :
  python src/analysis/heatmap_forecast.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_LONG = os.path.join(BASE_DIR, "reports", "forecast_2026_geo.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

QUARTIER_ORDER = [
    "Vieille Ville",
    "Cap d'Antibes",
    "Centre Ville",
    "Juan-les-Pins",
    "La Fontonne",
    "Antibes Nord Ouest",
]

QUARTIER_COLORS = {
    "Vieille Ville"     : "#DC2626",
    "Cap d'Antibes"     : "#2563EB",
    "Centre Ville"      : "#F59E0B",
    "Juan-les-Pins"     : "#16A34A",
    "La Fontonne"       : "#7C3AED",
    "Antibes Nord Ouest": "#0891B2",
}


# ── Heatmap prédictive ───────────────────────────────────────────────────────
def plot_heatmap(df: pd.DataFrame):
    """Heatmap quartier × mois 2026, couleur = croissance % vs déc 2025."""
    pivot = df.pivot(index="quartier", columns="periode",
                     values="growth_pct_vs_dec25")
    pivot = pivot.reindex([q for q in QUARTIER_ORDER if q in pivot.index])
    pivot = pivot[sorted(pivot.columns)]

    # Colormap divergente centrée sur 0
    vabs = np.nanmax(np.abs(pivot.values))
    norm = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(pivot.values, cmap="RdBu_r", norm=norm, aspect="auto")

    # Annotations dans chaque cellule
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            color = "white" if abs(val) > vabs * 0.5 else "black"
            ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                    color=color, fontsize=9, fontweight="bold")

    # Axes
    months_short = [pd.Timestamp(c).strftime("%b") for c in pivot.columns]
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(months_short, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_xlabel("Mois 2026", fontsize=11)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Croissance prédite vs déc 2025 (%)", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))

    ax.set_title("Carte thermique prédictive — Croissance prix/m² par quartier (jan-déc 2026)\n"
                 "LSTM(64,32) entraîné indépendamment par quartier sur 2014-2024",
                 fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "11_heatmap_forecast_2026.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {os.path.relpath(path, BASE_DIR)}")


# ── Trajectoires prédites avec IC ────────────────────────────────────────────
def plot_trajectories(df: pd.DataFrame):
    """Courbes prix prédits jan-déc 2026 par quartier + IC MC Dropout."""
    df = df.copy()
    df["periode"] = pd.to_datetime(df["periode"])

    fig, ax = plt.subplots(figsize=(14, 7))

    for q in QUARTIER_ORDER:
        sub = df[df["quartier"] == q].sort_values("periode")
        if sub.empty:
            continue
        color = QUARTIER_COLORS[q]
        ax.plot(sub["periode"], sub["prix_predit"],
                color=color, linewidth=2.2, marker="o", markersize=4, label=q)
        ax.fill_between(sub["periode"], sub["ci_low_10pct"], sub["ci_high_90pct"],
                        color=color, alpha=0.12)

    ax.set_title("Forecast LSTM 2026 par quartier — prix/m² médian\n"
                 "Bandes : intervalle 10–90% (Monte Carlo Dropout, 50 runs)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Période")
    ax.set_ylabel("Prix/m² médian (€)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f} €"))
    ax.legend(loc="best", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "12_trajectoires_forecast_2026.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {os.path.relpath(path, BASE_DIR)}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(INPUT_LONG):
        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_LONG}\n"
            "  → lance d'abord : python src/models/forecast_geo.py"
        )

    df = pd.read_csv(INPUT_LONG)
    print(f"📂 Chargement {os.path.relpath(INPUT_LONG, BASE_DIR)}  "
          f"({len(df)} lignes, {df['quartier'].nunique()} quartiers)")

    print("\n🎨 Génération des figures :")
    plot_heatmap(df)
    plot_trajectories(df)

    print("\n✅  heatmap_forecast.py terminé.")


if __name__ == "__main__":
    main()
