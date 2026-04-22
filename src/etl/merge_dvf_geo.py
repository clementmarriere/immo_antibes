"""
src/etl/merge_dvf_geo.py
=========================
Pipeline ETL pour les DVF géolocalisées :
  1. Charge et fusionne tous les fichiers DVF géo (2014-2025)
  2. Filtre sur Antibes + Appartements
  3. Assigne chaque transaction à un quartier via polygones GPS
  4. Calcule prix_m2, filtre outliers
  5. Agrège en série temporelle mensuelle par quartier
  6. Sauvegarde antibes_geo_monthly.csv

Usage :
  python src/etl/merge_dvf_geo.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from shapely.geometry import Point, Polygon
import json
from shapely.geometry import shape

# -- Chemins --
ROOT       = Path(__file__).parents[2]
RAW_DIR    = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "antibes_geo_monthly.csv"

# -- Fichiers sources --
REGION_FILES = [
    RAW_DIR / "dvf_geoloc_2014_regionpaca.csv",
    RAW_DIR / "dvf_geoloc_2015_regionpaca.csv",
    RAW_DIR / "dvf_geoloc_2016_regionpaca.csv",
    RAW_DIR / "dvf_geoloc_2017_regionpaca.csv",
    RAW_DIR / "dvf_geoloc_2018_regionpaca.csv",
    RAW_DIR / "DVF_geoloc_2019_regionsud.csv",
    RAW_DIR / "DVF_geoloc_2020_regionsud.csv",
]

ANTIBES_FILES = [
    RAW_DIR / "dvf_geo_antibes_2021.csv",
    RAW_DIR / "dvf_geo_antibes_2022.csv",
    RAW_DIR / "dvf_geo_antibes_2023.csv",
    RAW_DIR / "dvf_geo_antibes_2024.csv",
    RAW_DIR / "dvf_geo_antibes_2025.csv",
]

# -- Definition des quartiers (polygones GPS) --
# IMPORTANT : ordre = priorite (zones specifiques avant generiques)

with open("data/raw/map.geojson", "r") as f:
    geojson = json.load(f)

NOMS = [
    "La Fontonne",
    "Vieille Ville",
    "Centre Ville",
    "Cap d'Antibes",
    "Juan-les-Pins",
    "Antibes Nord Ouest",
]

QUARTIERS = {
    NOMS[i]: shape(feature["geometry"])
    for i, feature in enumerate(geojson["features"])
}


def assign_quartier(lon, lat):
    """Assigne un quartier a partir des coordonnees GPS."""
    if pd.isna(lon) or pd.isna(lat):
        return "Autre"
    point = Point(lon, lat)
    for name, polygon in QUARTIERS.items():
        if polygon.contains(point):
            return name
    return "Autre"


# -- 1. Chargement --
def load_files():
    chunks = []

    # Fichiers region -> filtrer sur Antibes (detection auto du separateur)
    for path in REGION_FILES:
        if not path.exists():
            print(f"  [!] Fichier manquant : {path.name}")
            continue
        print(f"  {path.name} ...", end=" ")
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline()
        sep = ";" if first_line.count(";") > first_line.count(",") else ","
        df = pd.read_csv(path, sep=sep, dtype=str, low_memory=False)
        df.columns = df.columns.str.lower().str.strip()
        mask = df["nom_commune"].str.upper().str.contains("ANTIBES", na=False)
        df = df[mask]
        print(f"{len(df)} lignes Antibes")
        chunks.append(df)

    # Fichiers Antibes directs
    for path in ANTIBES_FILES:
        if not path.exists():
            print(f"  [!] Fichier manquant : {path.name}")
            continue
        print(f"  {path.name} ...", end=" ")
        df = pd.read_csv(path, dtype=str, low_memory=False)
        df.columns = df.columns.str.lower().str.strip()
        print(f"{len(df)} lignes")
        chunks.append(df)

    df = pd.concat(chunks, ignore_index=True)
    print(f"\nTotal charge : {len(df):,} lignes")
    return df


# -- 2. Nettoyage --
def clean(df):
    for col in ["valeur_fonciere", "surface_reelle_bati",
                "nombre_pieces_principales", "longitude", "latitude"]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", ".", regex=False),
            errors="coerce"
        )

    df["date_mutation"] = pd.to_datetime(df["date_mutation"], errors="coerce")

    # Propagation valeur fonciere (bug DVF)
    df["valeur_fonciere"] = df.groupby(
        ["date_mutation", "adresse_numero", "adresse_nom_voie", "code_postal"]
    )["valeur_fonciere"].transform(lambda x: x.ffill().bfill())

    df = df[df["type_local"] == "Appartement"].copy()
    print(f"Apres filtre Appartement      : {len(df):,}")

    df = df[df["valeur_fonciere"] > 1000]
    df = df[df["surface_reelle_bati"] > 1]
    df = df.dropna(subset=["date_mutation", "longitude", "latitude"])

    df["prix_m2"] = df["valeur_fonciere"] / df["surface_reelle_bati"]
    df["prix_m2"] = df["prix_m2"].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["prix_m2"])

    q_low  = df["prix_m2"].quantile(0.01)
    q_high = df["prix_m2"].quantile(0.99)
    df = df[df["prix_m2"].between(q_low, q_high)]

    print(f"Apres nettoyage complet       : {len(df):,}")
    return df.sort_values("date_mutation").reset_index(drop=True)


# -- 3. Assignation des quartiers --
def add_quartier(df):
    print("Assignation des quartiers...", end=" ")
    df["quartier"] = df.apply(
        lambda row: assign_quartier(row["longitude"], row["latitude"]),
        axis=1
    )
    dist = df["quartier"].value_counts()
    print("OK")
    print("\nDistribution par quartier :")
    print(dist.to_string())
    print(f"\nSans quartier (Autre) : {dist.get('Autre', 0)} "
          f"({dist.get('Autre', 0)/len(df)*100:.1f}%)")
    return df


# -- 4. Agregation mensuelle par quartier --
def aggregate(df):
    df["periode"] = df["date_mutation"].dt.to_period("M")

    monthly = df.groupby(["periode", "quartier"]).agg(
        prix_m2_median=("prix_m2", "median"),
        prix_m2_mean=("prix_m2", "mean"),
        prix_m2_std=("prix_m2", "std"),
        volume=("prix_m2", "count"),
        surface_median=("surface_reelle_bati", "median"),
        nb_pieces_median=("nombre_pieces_principales", "median"),
    ).reset_index()

    monthly["periode"] = monthly["periode"].dt.to_timestamp()
    monthly = monthly.sort_values(["quartier", "periode"]).reset_index(drop=True)

    # Completer les mois manquants par quartier
    full_range = pd.date_range(
        start=monthly["periode"].min(),
        end=monthly["periode"].max(),
        freq="MS"
    )
    quartiers = monthly["quartier"].unique()
    idx = pd.MultiIndex.from_product(
        [full_range, quartiers], names=["periode", "quartier"]
    )
    monthly = monthly.set_index(["periode", "quartier"]).reindex(idx).reset_index()

    cols_interp = ["prix_m2_median", "prix_m2_mean", "prix_m2_std",
                   "surface_median", "nb_pieces_median"]
    monthly[cols_interp] = (
        monthly.groupby("quartier")[cols_interp]
        .transform(lambda x: x.interpolate(method="linear"))
    )
    monthly["volume"] = monthly["volume"].fillna(0).astype(int)

    monthly["annee"]     = monthly["periode"].dt.year
    monthly["mois"]      = monthly["periode"].dt.month
    monthly["trimestre"] = monthly["periode"].dt.quarter

    print(f"\nSerie temporelle : {monthly['periode'].nunique()} mois "
          f"x {monthly['quartier'].nunique()} quartiers "
          f"= {len(monthly)} lignes")
    return monthly


# -- Main --
def main():
    print("Chargement des fichiers DVF geolocatises...\n")
    df = load_files()

    print("\nNettoyage...")
    df = clean(df)

    print("\nAssignation geographique...")
    df = add_quartier(df)

    print("\nAggregation mensuelle par quartier...")
    monthly = aggregate(df)

    monthly.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved : {OUTPUT_FILE}")

    print("\nApercu par quartier (prix median moyen) :")
    summary = (monthly.groupby("quartier")["prix_m2_median"]
               .mean().sort_values(ascending=False))
    print(summary.round(0).to_string())

    print("\nmerge_dvf_geo.py termine.")
    print("Prochaine etape : src/analysis/heatmap.py")


if __name__ == "__main__":
    main()