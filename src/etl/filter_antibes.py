import pandas as pd
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parents[2]
INPUT_FILE = ROOT / "data/processed/full_dvf_2021_2025.csv"
OUTPUT_FILE = ROOT / "data/processed/antibes_2021_2025.csv"

TYPE_LOCAL_MAP = {
    "Local industriel. commercial ou assimilÃ©": "Local industriel. commercial ou assimilé",
    "DÃ©pendance": "Dépendance",
    "AppartementÃ": "Appartement",
}


def to_float_fr(series):
    return pd.to_numeric(
        series.str.replace(",", ".", regex=False).str.replace(" ", "", regex=False),
        errors="coerce"
    )


def main():
    print("🔄 Streaming full DVF 2021-2025...")

    chunks = []
    total = 0

    for chunk in pd.read_csv(
        INPUT_FILE,
        sep="|",
        encoding="utf-8",
        dtype=str,
        chunksize=100_000
    ):
        total += len(chunk)
        chunk.columns = chunk.columns.str.lower().str.strip().str.replace(" ", "_")

        mask = chunk["commune"].astype(str).str.upper().str.contains("ANTIBES", na=False)
        filtered = chunk[mask]

        if len(filtered) > 0:
            chunks.append(filtered)

        print(f"\r  {total:,} lignes lues — {sum(len(c) for c in chunks)} Antibes trouvées", end="")

    print(f"\n✅ Streaming terminé")

    df = pd.concat(chunks, ignore_index=True)
    n_antibes = len(df)
    print(f"Après filtre Antibes: {n_antibes}")

    # -----------------------------
    # FIX ENCODAGE
    # -----------------------------
    if "type_local" in df.columns:
        df["type_local"] = df["type_local"].replace(TYPE_LOCAL_MAP)

    # -----------------------------
    # CONVERSION NUMÉRIQUE
    # -----------------------------
    df["valeur_fonciere"] = to_float_fr(df["valeur_fonciere"])
    df["surface_reelle_bati"] = to_float_fr(df["surface_reelle_bati"])
    df["date_mutation"] = pd.to_datetime(df["date_mutation"], errors="coerce")

    # -----------------------------
    # PROPAGATION VALEUR FONCIERE
    # -----------------------------
    df["valeur_fonciere"] = df.groupby(
        ["date_mutation", "no_voie", "voie", "code_postal"]
    )["valeur_fonciere"].transform(lambda x: x.ffill().bfill())

    # -----------------------------
    # CLEAN
    # -----------------------------
    n = len(df)

    df = df[df["valeur_fonciere"].notna()]
    print(f"Après dropna valeur_fonciere  : {len(df)} ({n - len(df)} supprimées)")
    n = len(df)

    df = df[df["valeur_fonciere"] > 1000]
    print(f"Après filtre valeur > 1000    : {len(df)} ({n - len(df)} supprimées)")
    n = len(df)

    df = df[df["surface_reelle_bati"].notna()]
    print(f"Après dropna surface          : {len(df)} ({n - len(df)} supprimées)")
    n = len(df)

    df = df[df["surface_reelle_bati"] > 1]
    print(f"Après filtre surface > 1      : {len(df)} ({n - len(df)} supprimées)")
    n = len(df)

    df = df.dropna(subset=["date_mutation"])
    print(f"Après dropna date_mutation    : {len(df)} ({n - len(df)} supprimées)")
    n = len(df)

    # -----------------------------
    # FEATURE ENGINEERING
    # -----------------------------
    df["prix_m2"] = df["valeur_fonciere"] / df["surface_reelle_bati"]
    df["prix_m2"] = df["prix_m2"].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["prix_m2"])
    print(f"Après calcul prix_m2          : {len(df)} ({n - len(df)} supprimées)")
    n = len(df)

    df["annee"] = df["date_mutation"].dt.year
    df["mois"] = df["date_mutation"].dt.month

    # -----------------------------
    # OUTLIERS PAR TYPE_LOCAL
    # -----------------------------
    if "type_local" in df.columns:
        def filter_outliers(g):
            if len(g) > 10:
                return g[g["prix_m2"].between(
                    g["prix_m2"].quantile(0.01),
                    g["prix_m2"].quantile(0.99)
                )]
            return g

        df = (
            df.groupby("type_local", group_keys=False)[df.columns]
            .apply(filter_outliers)
            .reset_index(drop=True)
        )
    else:
        q_low = df["prix_m2"].quantile(0.01)
        q_high = df["prix_m2"].quantile(0.99)
        df = df[df["prix_m2"].between(q_low, q_high)]

    print(f"Après filtre outliers         : {len(df)} ({n - len(df)} supprimées)")

    df = df.sort_values("date_mutation").reset_index(drop=True)

    print(f"\n✅ Final dataset: {len(df)} lignes")
    print(f"📊 Taux de rétention: {len(df)/n_antibes:.1%}")

    print("\n🏠 Distribution type_local:")
    if "type_local" in df.columns:
        print(df["type_local"].value_counts(dropna=False))
    else:
        print("⚠️  colonne type_local absente")

    # -----------------------------
    # SAVE
    # -----------------------------
    df.to_csv(OUTPUT_FILE, index=False, sep="|", encoding="utf-8")
    print(f"\n💾 Saved: {OUTPUT_FILE}")
    print("📅 Range:", df["date_mutation"].min(), "→", df["date_mutation"].max())
    print("\n📊 Preview:")
    print(df[["date_mutation", "type_local", "valeur_fonciere", "surface_reelle_bati", "prix_m2"]].head(10))


if __name__ == "__main__":
    main()