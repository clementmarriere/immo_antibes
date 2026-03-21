# src/etl/filter_antibes.py

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parents[2]
INPUT_FILE = ROOT / "data/raw/valeursfoncieres-2024.txt"
OUTPUT_FILE = ROOT / "data/processed/antibes_2024.csv"

def main():
    print("Loading DVF...")
    df = pd.read_csv(INPUT_FILE, sep="|", low_memory=False)

    # normaliser colonnes
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # filtrer Antibes
    df = df[df["commune"].str.lower() == "antibes"]

    # filtrer appartements
    df = df[df["type_local"].str.lower() == "appartement"]

    # Nettoyage valeur_fonciere
    df["valeur_fonciere"] = (
        df["valeur_fonciere"]
        .astype(str)
        .str.replace("\xa0", "")   # enlever les espaces insécables
        .str.replace(",", ".")     # virgule -> point
        .astype(float)
    )

    # Nettoyage surface
    df["surface_reelle_bati"] = pd.to_numeric(df["surface_reelle_bati"], errors="coerce")

    # enlever surfaces nulles
    df = df[df["surface_reelle_bati"] > 0]
    df = df[df["valeur_fonciere"] > 0]

    # calcul prix au m²
    df["prix_m2"] = df["valeur_fonciere"] / df["surface_reelle_bati"]

    # sauvegarder
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Done. {len(df)} transactions saved in {OUTPUT_FILE}")

if __name__ == "__main__":
    main()