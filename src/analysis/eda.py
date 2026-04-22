import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parents[2]
INPUT_FILE = ROOT / "data/processed/antibes_appart_monthly.csv"
OUTPUT_DIR = ROOT / "reports/figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(INPUT_FILE, parse_dates=["periode"])
    df = df.sort_values("periode").reset_index(drop=True)

    print(f"✅ {len(df)} mois chargés ({df['periode'].min().date()} → {df['periode'].max().date()})")

    fig, axes = plt.subplots(4, 1, figsize=(14, 20))
    fig.suptitle("EDA — Marché Appartements Antibes 2014–2025", fontsize=16, fontweight="bold", y=0.98)

    # -----------------------------
    # 1. PRIX/M² MÉDIAN + TENDANCE
    # -----------------------------
    ax1 = axes[0]
    ax1.plot(df["periode"], df["prix_m2_median"], color="#2563eb", linewidth=1.5, alpha=0.8, label="Prix/m² médian")

    # Moyenne mobile 12 mois
    df["mm12"] = df["prix_m2_median"].rolling(12, center=True).mean()
    ax1.plot(df["periode"], df["mm12"], color="#dc2626", linewidth=2.5, label="Moyenne mobile 12 mois")

    # Zone covid
    ax1.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-01"),
                alpha=0.15, color="orange", label="Confinement")

    ax1.set_title("Évolution du Prix/m² Médian", fontweight="bold")
    ax1.set_ylabel("€/m²")
    ax1.legend()
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f} €"))
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())

    # -----------------------------
    # 2. VOLUME MENSUEL
    # -----------------------------
    ax2 = axes[1]
    ax2.bar(df["periode"], df["volume"], color="#2563eb", alpha=0.6, width=25, label="Transactions/mois")

    df["mm12_vol"] = df["volume"].rolling(12, center=True).mean()
    ax2.plot(df["periode"], df["mm12_vol"], color="#dc2626", linewidth=2, label="Moyenne mobile 12 mois")

    ax2.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-01"),
                alpha=0.15, color="orange", label="Confinement")

    ax2.set_title("Volume Mensuel de Transactions", fontweight="bold")
    ax2.set_ylabel("Nombre de transactions")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())

    # -----------------------------
    # 3. SAISONNALITÉ
    # -----------------------------
    ax3 = axes[2]

    df["mois_num"] = df["periode"].dt.month
    saisonnalite = df.groupby("mois_num")["prix_m2_median"].median()

    mois_labels = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
                   "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]

    bars = ax3.bar(range(1, 13), saisonnalite.values, color="#2563eb", alpha=0.7)

    # Colorier le min/max
    max_idx = saisonnalite.values.argmax()
    min_idx = saisonnalite.values.argmin()
    bars[max_idx].set_color("#16a34a")
    bars[min_idx].set_color("#dc2626")

    ax3.set_xticks(range(1, 13))
    ax3.set_xticklabels(mois_labels)
    ax3.set_title("Saisonnalité — Prix/m² Médian par Mois (2014–2025)", fontweight="bold")
    ax3.set_ylabel("€/m²")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f} €"))
    ax3.grid(True, alpha=0.3, axis="y")

    # Annotation min/max
    ax3.annotate(f"Max: {saisonnalite.values[max_idx]:,.0f}€",
                 xy=(max_idx + 1, saisonnalite.values[max_idx]),
                 xytext=(0, 8), textcoords="offset points",
                 ha="center", color="#16a34a", fontweight="bold", fontsize=9)
    ax3.annotate(f"Min: {saisonnalite.values[min_idx]:,.0f}€",
                 xy=(min_idx + 1, saisonnalite.values[min_idx]),
                 xytext=(0, 8), textcoords="offset points",
                 ha="center", color="#dc2626", fontweight="bold", fontsize=9)

    # -----------------------------
    # 4. CROISSANCE ANNUELLE YoY
    # -----------------------------
    ax4 = axes[3]

    df_annual = df.groupby("annee")["prix_m2_median"].median().reset_index()
    df_annual = df_annual[df_annual["annee"] < 2025]  # 2025 incomplet
    df_annual["yoy"] = df_annual["prix_m2_median"].pct_change() * 100

    colors = ["#16a34a" if v >= 0 else "#dc2626" for v in df_annual["yoy"].fillna(0)]
    ax4.bar(df_annual["annee"], df_annual["yoy"].fillna(0), color=colors, alpha=0.8)
    ax4.axhline(0, color="black", linewidth=0.8)

    ax4.set_title("Croissance Annuelle du Prix/m² (YoY %)", fontweight="bold")
    ax4.set_ylabel("Variation (%)")
    ax4.set_xticks(df_annual["annee"])
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.1f}%"))
    ax4.grid(True, alpha=0.3, axis="y")

    # Annotations valeurs
    for _, row in df_annual.iterrows():
        if pd.notna(row["yoy"]):
            ax4.annotate(f"{row['yoy']:+.1f}%",
                         xy=(row["annee"], row["yoy"]),
                         xytext=(0, 5 if row["yoy"] >= 0 else -15),
                         textcoords="offset points",
                         ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "eda_antibes_appartements.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n💾 Saved: {output_path}")

    # -----------------------------
    # STATS RÉSUMÉ
    # -----------------------------
    print("\n📊 Statistiques clés :")
    print(f"   Prix/m² médian 2014 : {df[df['annee']==2014]['prix_m2_median'].median():.0f} €")
    print(f"   Prix/m² médian 2024 : {df[df['annee']==2024]['prix_m2_median'].median():.0f} €")
    croissance_totale = (df[df['annee']==2024]['prix_m2_median'].median() /
                         df[df['annee']==2014]['prix_m2_median'].median() - 1) * 100
    print(f"   Croissance totale   : {croissance_totale:+.1f}%")
    print(f"   Mois le + actif     : {mois_labels[df.groupby('mois_num')['volume'].mean().idxmax()-1]}")
    print(f"   Mois le - actif     : {mois_labels[df.groupby('mois_num')['volume'].mean().idxmin()-1]}")


if __name__ == "__main__":
    main()