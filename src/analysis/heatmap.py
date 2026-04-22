"""
src/analysis/heatmap.py
========================
Visualisations spatio-temporelles du marche immobilier d'Antibes.

Figures generees :
  06_heatmap_prix_quartiers.png  -> heatmap annuelle prix/m2 par quartier
  07_trajectoires_quartiers.png  -> evolution des prix par quartier (2014-2025)
  08_croissance_quartiers.png    -> croissance cumulee par quartier (base 100)
  09_comparaison_annuelle.png    -> barplot prix median par quartier et par annee

Usage :
  python src/analysis/heatmap.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# -- Chemins --
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FILE  = os.path.join(BASE_DIR, "data", "processed", "antibes_geo_monthly.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

QUARTIER_COLORS = {
    "Vieille Ville"     : "#DC2626",
    "Cap d'Antibes"     : "#2563EB",
    "Centre Ville"      : "#F59E0B",
    "Juan-les-Pins"     : "#16A34A",
    "La Fontonne"       : "#7C3AED",
    "Antibes Nord Ouest": "#0891B2",
}

QUARTIER_ORDER = [
    "Vieille Ville",
    "Cap d'Antibes",
    "Centre Ville",
    "Juan-les-Pins",
    "La Fontonne",
    "Antibes Nord Ouest",
]


# -- Chargement --
def load_data():
    df = pd.read_csv(DATA_FILE, parse_dates=["periode"])
    df = df.sort_values(["quartier", "periode"]).reset_index(drop=True)
    print(f"Charge : {len(df)} lignes  |  "
          f"{df['quartier'].nunique()} quartiers  |  "
          f"{df['periode'].nunique()} mois")
    return df


# -- Aggregation annuelle --
def annual(df):
    agg = df.groupby(["annee", "quartier"]).agg(
        prix_m2_median=("prix_m2_median", "median"),
        volume=("volume", "sum"),
    ).reset_index()
    return agg


# -- Figure 1 : Heatmap prix/m2 par quartier et annee --
def plot_heatmap(agg):
    pivot = agg.pivot(index="quartier", columns="annee", values="prix_m2_median")

    # Reorder rows
    order = [q for q in QUARTIER_ORDER if q in pivot.index]
    pivot = pivot.reindex(order)

    fig, ax = plt.subplots(figsize=(16, 5))

    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest")

    # Axes
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.astype(int), fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=11)

    # Annotations valeurs
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = "white" if val > pivot.values.max() * 0.7 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Prix/m² median (€)", fontsize=10)

    ax.set_title("Evolution du prix/m² median par quartier — Antibes 2014-2025",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Annee", fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "06_heatmap_prix_quartiers.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sauvegarde : {path}")


# -- Figure 2 : Trajectoires mensuelles par quartier --
def plot_trajectoires(df):
    fig, ax = plt.subplots(figsize=(16, 7))

    QUARTIERS_PLOT = [q for q in QUARTIER_ORDER
                  if q in df["quartier"].unique()
                  and q != "Autre"]

    for q in QUARTIERS_PLOT:
        sub = df[df["quartier"] == q].sort_values("periode").copy()
        sub = sub[sub["volume"] >= 3]  # filtre mois sans volume suffisant
        color = QUARTIER_COLORS.get(q, "#888888")
        sub["smooth"] = sub["prix_m2_median"].rolling(6, center=True, min_periods=3).mean()
        ax.plot(sub["periode"], sub["smooth"], color=color,
                linewidth=2, label=q)
        ax.fill_between(sub["periode"], sub["smooth"],
                        alpha=0.05, color=color)

    # Annotations evenements cles
    events = {
        "2020-03": ("COVID", "#EF4444"),
        "2022-01": ("Hausse\ntaux BCE", "#F59E0B"),
    }
    for date_str, (label, color) in events.items():
        ax.axvline(pd.to_datetime(date_str), color=color,
                   linestyle="--", alpha=0.6, linewidth=1.2)
        ax.text(pd.to_datetime(date_str), ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 3000,
                label, color=color, fontsize=8, ha="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:,.0f} €"
    ))
    ax.set_title("Trajectoires des prix/m² par quartier — Antibes 2014-2025\n"
                 "(moyenne mobile 6 mois)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Periode")
    ax.set_ylabel("Prix/m² median (€)")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "07_trajectoires_quartiers.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sauvegarde : {path}")


# -- Figure 3 : Croissance cumulee (base 100 en 2014) --
def plot_croissance(agg):
    fig, ax = plt.subplots(figsize=(14, 6))

    quartiers = [q for q in QUARTIER_ORDER if q in agg["quartier"].unique()]

    for q in quartiers:
        sub = agg[agg["quartier"] == q].sort_values("annee").copy()
        base = sub[sub["annee"] == sub["annee"].min()]["prix_m2_median"].values
        if len(base) == 0 or base[0] == 0:
            continue
        sub["indice"] = sub["prix_m2_median"] / base[0] * 100
        color = QUARTIER_COLORS.get(q, "#888888")

        ax.plot(sub["annee"], sub["indice"], color=color,
                linewidth=2.5, marker="o", markersize=4, label=q)

        # Annotation valeur finale
        last = sub.iloc[-1]
        ax.annotate(f"+{last['indice']-100:.0f}%",
                    xy=(last["annee"], last["indice"]),
                    xytext=(5, 0), textcoords="offset points",
                    fontsize=8, color=color, fontweight="bold")

    ax.axhline(100, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_title("Croissance cumulee des prix/m² par quartier\n(base 100 = premiere annee disponible)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Annee")
    ax.set_ylabel("Indice (base 100)")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(agg["annee"].unique())
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "08_croissance_quartiers.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sauvegarde : {path}")


# -- Figure 4 : Barplot comparatif derniere annee disponible --
def plot_comparaison(agg):
    derniere_annee = agg["annee"].max()
    premiere_annee = agg["annee"].min()

    sub_last  = agg[agg["annee"] == derniere_annee].set_index("quartier")
    sub_first = agg[agg["annee"] == premiere_annee].set_index("quartier")

    quartiers = [q for q in QUARTIER_ORDER if q in sub_last.index]
    prix_last  = [sub_last.loc[q, "prix_m2_median"] for q in quartiers]
    prix_first = [sub_first.loc[q, "prix_m2_median"]
                  if q in sub_first.index else np.nan for q in quartiers]

    x = np.arange(len(quartiers))
    w = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(x - w/2, prix_first, w,
                   label=str(premiere_annee), color="#93C5FD", edgecolor="white")
    bars2 = ax.bar(x + w/2, prix_last,  w,
                   label=str(derniere_annee),
                   color=[QUARTIER_COLORS.get(q, "#888") for q in quartiers],
                   edgecolor="white")

    # Annotations delta
    for i, (q, pf, pl) in enumerate(zip(quartiers, prix_first, prix_last)):
        if not np.isnan(pf) and pf > 0:
            delta = (pl - pf) / pf * 100
            ax.text(x[i] + w/2, pl + 50, f"+{delta:.0f}%",
                    ha="center", fontsize=9, fontweight="bold", color="#1E3A5F")

    ax.set_xticks(x)
    ax.set_xticklabels(quartiers, rotation=15, ha="right", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:,.0f} €"
    ))
    ax.set_title(f"Prix/m² median par quartier : {premiere_annee} vs {derniere_annee}",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Prix/m² median (€)")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "09_comparaison_annuelle.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sauvegarde : {path}")


# -- Main --
def main():
    print("Chargement des donnees...")
    df  = load_data()
    agg = annual(df)

    print("\nGeneration des figures...")
    plot_heatmap(agg)
    plot_trajectoires(df)
    plot_croissance(agg)
    plot_comparaison(agg)

    print("\n4 figures sauvegardees dans reports/figures/")
    print("  06_heatmap_prix_quartiers.png")
    print("  07_trajectoires_quartiers.png")
    print("  08_croissance_quartiers.png")
    print("  09_comparaison_annuelle.png")
    print("\nProchaine etape : rapport 4 pages")


if __name__ == "__main__":
    main()