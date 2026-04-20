import pandas as pd
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
ROOT = Path(__file__).parents[2]
INPUT_FILE = ROOT / "data/processed/antibes_2024_clean.csv"
OUTPUT_FILE = ROOT / "data/processed/top20_antibes.csv"

BUDGET_MAX = 500000
SURFACE_MIN = 25

# -----------------------------
# LOAD
# -----------------------------
print("Loading clean dataset...")
df = pd.read_csv(INPUT_FILE)

print(f"{len(df)} biens chargés")

# -----------------------------
# FILTRES INVESTISSEUR
# -----------------------------
df = df[df["type_local"] == "Appartement"]
df = df[df["valeur_fonciere"] <= BUDGET_MAX]
df = df[df["surface_reelle_bati"] >= SURFACE_MIN]

# nettoyage supplémentaire
df = df[
    (df["prix_m2"] > 1000) &
    (df["prix_m2"] < 20000)
]

print(f"{len(df)} biens après filtres")

# -----------------------------
# NORMALISATION
# -----------------------------
df["prix_norm"] = (
    (df["prix_m2"] - df["prix_m2"].min()) /
    (df["prix_m2"].max() - df["prix_m2"].min())
)

df["surface_norm"] = (
    (df["surface_reelle_bati"] - df["surface_reelle_bati"].min()) /
    (df["surface_reelle_bati"].max() - df["surface_reelle_bati"].min())
)

# réduire l’effet des très grandes surfaces
df["surface_norm"] = df["surface_norm"] ** 0.5

# -----------------------------
# SCORE INVESTISSEMENT
# -----------------------------
df["score"] = (
    (1 - df["prix_norm"]) * 0.7 +
    df["surface_norm"] * 0.3
)

# -----------------------------
# TOP 20
# -----------------------------
top20 = df.sort_values("score", ascending=False).head(20)

print("\n🔥 ----- TOP 20 INVESTISSEMENTS RÉALISTES ----- 🔥\n")

for i, row in top20.iterrows():
    print(f"Bien #{i}")
    print(f"Prix : {row['valeur_fonciere']:.0f} €")
    print(f"Surface : {row['surface_reelle_bati']:.0f} m²")
    print(f"Prix/m² : {row['prix_m2']:.0f} €")
    print(f"Score : {row['score']:.3f}")
    print("-" * 40)

# -----------------------------
# SAVE
# -----------------------------
top20.to_csv(OUTPUT_FILE, index=False)

print(f"\nTop 20 sauvegardé dans {OUTPUT_FILE}")