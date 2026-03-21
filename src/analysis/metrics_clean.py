# src/analysis/metrics_clean.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# CONFIGURATION
# -----------------------------
ROOT = Path(__file__).parents[2]
INPUT_FILE = ROOT / "data/processed/antibes_2024_clean.csv"

# -----------------------------
# CHARGER LES DONNÉES
# -----------------------------
print("Loading DVF...")
df = pd.read_csv(INPUT_FILE)

# -----------------------------
# NETTOYAGE ET CONVERSION
# -----------------------------
# enlever les espaces insécables et convertir en float
df["valeur_fonciere"] = (
    df["valeur_fonciere"]
    .astype(str)
    .str.replace("\xa0", "")
    .str.replace(",", ".")
    .astype(float)
)

df["surface_reelle_bati"] = pd.to_numeric(df["surface_reelle_bati"], errors="coerce")

# filtrer les surfaces nulles ou 0
df = df[df["surface_reelle_bati"] > 0]

# calculer prix au m²
df["prix_m2"] = df["valeur_fonciere"] / df["surface_reelle_bati"]

print(f"Done. {len(df)} transactions valides.")

# -----------------------------
# STATISTIQUES GLOBALES
# -----------------------------
print("\n----- STATISTIQUES GLOBALES -----")
print(f"Nombre total de transactions : {len(df)}")
print(f"Prix moyen €/m² : {df['prix_m2'].mean():.0f}")
print(f"Prix médian €/m² : {df['prix_m2'].median():.0f}")
print(f"Prix minimum €/m² : {df['prix_m2'].min():.0f}")
print(f"Prix maximum €/m² : {df['prix_m2'].max():.0f}")
print(f"Surface moyenne : {df['surface_reelle_bati'].mean():.0f} m²")
print(f"Surface médiane : {df['surface_reelle_bati'].median():.0f} m²\n")

# -----------------------------
# BONS PLANS (25e percentile)
# -----------------------------
q25 = df["prix_m2"].quantile(0.25)
df_bons_plans = df[df["prix_m2"] <= q25]

print("----- BONS PLANS (prix/m² <= 25e percentile) -----")
print(f"Nombre de biens : {len(df_bons_plans)}")
print(df_bons_plans[["type_local","valeur_fonciere","surface_reelle_bati","prix_m2"]].head(10))
print("\n")

# -----------------------------
# TOP 10 APPARTEMENTS LES PLUS INTÉRESSANTS
# -----------------------------
# critère simple : surface / prix_m² (plus grand = meilleur rapport)
df["score_investissement"] = df["surface_reelle_bati"] / df["prix_m2"]
top10 = df.sort_values("score_investissement", ascending=False).head(10)

print("----- TOP 10 APPARTEMENTS INTÉRESSANTS (surface/prix m²) -----")
print(top10[["type_local","valeur_fonciere","surface_reelle_bati","prix_m2","score_investissement"]])
print("\n")

# -----------------------------
# HISTOGRAMME ET BOXPLOT
# -----------------------------
plt.figure(figsize=(12,5))

# histogramme prix/m²
plt.subplot(1,2,1)
plt.hist(df["prix_m2"], bins=50, color='skyblue', edgecolor='black')
plt.title("Histogramme du prix au m² - Antibes")
plt.xlabel("Prix €/m²")
plt.ylabel("Nombre de transactions")

# boxplot prix/m²
plt.subplot(1,2,2)
plt.boxplot(df["prix_m2"], vert=True)
plt.title("Boxplot du prix au m² - Antibes")
plt.ylabel("Prix €/m²")

plt.tight_layout()
plt.show()