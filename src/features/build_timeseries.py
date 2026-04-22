import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parents[2]
INPUT_FILE = ROOT / "data/processed/antibes_full_clean.csv"
OUTPUT_FILE = ROOT / "data/processed/antibes_timeseries.csv"

def main():
    print("🔄 Loading Antibes dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"{len(df)} transactions")

    # -----------------------------
    # DATE
    # -----------------------------
    df["date_mutation"] = pd.to_datetime(df["date_mutation"], errors="coerce")

    df = df.dropna(subset=["date_mutation"])

    # -----------------------------
    # MONTHLY AGGREGATION
    # -----------------------------
    print("\n📊 Aggregating monthly price evolution...")

    df["month"] = df["date_mutation"].dt.to_period("M")

    monthly = df.groupby("month")["prix_m2"].mean().reset_index()

    monthly["month"] = monthly["month"].dt.to_timestamp()

    monthly = monthly.sort_values("month")

    print(f"{len(monthly)} mois obtenus")

    # -----------------------------
    # NORMALISATION
    # -----------------------------
    monthly["prix_m2_scaled"] = (
        (monthly["prix_m2"] - monthly["prix_m2"].min()) /
        (monthly["prix_m2"].max() - monthly["prix_m2"].min())
    )

    # -----------------------------
    # SAVE
    # -----------------------------
    monthly.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Sauvegardé : {OUTPUT_FILE}")

    print("\nAperçu :")
    print(monthly.head())
    print(df["date_mutation"].min(), df["date_mutation"].max())
    print(df["date_mutation"].dt.year.value_counts().sort_index())

if __name__ == "__main__":
    main()