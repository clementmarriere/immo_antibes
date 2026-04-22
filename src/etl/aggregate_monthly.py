# aggregate_monthly.py

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parents[2]
INPUT_FILE = ROOT / "data/processed/antibes_2014_2025.csv"
OUTPUT_FILE = ROOT / "data/processed/antibes_appart_monthly.csv"


def main():
    print("🔄 Loading Antibes dataset...")

    df = pd.read_csv(INPUT_FILE, sep="|", encoding="utf-8", dtype=str)
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    df["date_mutation"] = pd.to_datetime(df["date_mutation"], errors="coerce")
    df["prix_m2"] = pd.to_numeric(df["prix_m2"], errors="coerce")
    df["valeur_fonciere"] = pd.to_numeric(df["valeur_fonciere"], errors="coerce")
    df["surface_reelle_bati"] = pd.to_numeric(df["surface_reelle_bati"], errors="coerce")
    df["nombre_pieces_principales"] = pd.to_numeric(df["nombre_pieces_principales"], errors="coerce")

    print(f"{len(df)} lignes chargées")

    # -----------------------------
    # FILTRE APPARTEMENTS
    # -----------------------------
    df = df[df["type_local"] == "Appartement"].copy()
    print(f"Après filtre Appartement: {len(df)}")

    # -----------------------------
    # COLONNE PERIODE MENSUELLE
    # -----------------------------
    df["periode"] = df["date_mutation"].dt.to_period("M")

    # -----------------------------
    # AGGREGATION MENSUELLE
    # -----------------------------
    monthly = df.groupby("periode").agg(
        prix_m2_median=("prix_m2", "median"),
        prix_m2_mean=("prix_m2", "mean"),
        prix_m2_std=("prix_m2", "std"),
        volume=("prix_m2", "count"),
        valeur_fonciere_median=("valeur_fonciere", "median"),
        surface_median=("surface_reelle_bati", "median"),
        nb_pieces_median=("nombre_pieces_principales", "median"),
    ).reset_index()

    monthly["periode"] = monthly["periode"].dt.to_timestamp()
    monthly = monthly.sort_values("periode").reset_index(drop=True)

    # -----------------------------
    # GESTION MOIS MANQUANTS
    # -----------------------------
    full_range = pd.date_range(
        start=monthly["periode"].min(),
        end=monthly["periode"].max(),
        freq="MS"
    )

    monthly = monthly.set_index("periode").reindex(full_range).reset_index()
    monthly = monthly.rename(columns={"index": "periode"})

    # Interpolation linéaire pour les mois sans transaction
    cols_to_interpolate = ["prix_m2_median", "prix_m2_mean", "prix_m2_std",
                           "valeur_fonciere_median", "surface_median", "nb_pieces_median"]
    monthly[cols_to_interpolate] = monthly[cols_to_interpolate].interpolate(method="linear")
    monthly["volume"] = monthly["volume"].fillna(0).astype(int)

    # -----------------------------
    # FEATURES TEMPORELLES
    # -----------------------------
    monthly["annee"] = monthly["periode"].dt.year
    monthly["mois"] = monthly["periode"].dt.month
    monthly["trimestre"] = monthly["periode"].dt.quarter

    # -----------------------------
    # STATS
    # -----------------------------
    n_mois = len(monthly)
    n_manquants = (monthly["volume"] == 0).sum()

    print(f"\n📊 Série temporelle : {n_mois} mois")
    print(f"   Mois sans transaction (interpolés) : {n_manquants}")
    print(f"   Range : {monthly['periode'].min().date()} → {monthly['periode'].max().date()}")
    print(f"   Prix/m² médian global : {monthly['prix_m2_median'].mean():.0f} €")
    print(f"   Volume moyen mensuel : {monthly['volume'].mean():.1f} transactions")

    print("\n📅 Aperçu :")
    print(monthly[["periode", "prix_m2_median", "volume"]].head(10).to_string(index=False))

    # -----------------------------
    # SAVE
    # -----------------------------
    monthly.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()