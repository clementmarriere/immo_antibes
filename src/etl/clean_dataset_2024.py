# ⚠️ LEGACY SCRIPT
# Used for initial 2024 analysis (non time-series)
# Not used in current Deep Learning pipeline

import pandas as pd

from pathlib import Path

ROOT = Path(__file__).parents[2]
INPUT_FILE = ROOT / "data/processed/antibes_2024.csv"
OUTPUT_FILE = ROOT / "data/processed/antibes_2024_clean.csv"

df = pd.read_csv(INPUT_FILE)

# Filtrage intelligent
df_clean = df[
    (df["prix_m2"] > 2000) &
    (df["prix_m2"] < 15000)
]

df_clean.to_csv(OUTPUT_FILE, index=False)

print(f"Dataset clean créé : {len(df_clean)} lignes")
print("Avant nettoyage :", len(df))
print("Après nettoyage :", len(df_clean))
print("Supprimé :", len(df) - len(df_clean))