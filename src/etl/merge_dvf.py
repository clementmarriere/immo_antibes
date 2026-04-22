import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parents[2]
RAW_DIR = ROOT / "data/raw"
OUTPUT_FILE = ROOT / "data/processed/full_dvf_2021_2025.csv"

TARGET_FILES = [
    "ValeursFoncieres-2021.txt",
    "ValeursFoncieres-2022.txt",
    "ValeursFoncieres-2023.txt",
    "ValeursFoncieres-2024.txt",
    "ValeursFoncieres-2025.txt",
]

def load_txt(file):
    print(f"📄 Loading {file.name}")
    df = pd.read_csv(
        file,
        sep="|",
        low_memory=False,
        encoding="latin-1",
        dtype=str          # ✅ tout en string, on ne touche à rien
    )
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    print(f"   → {len(df)} lignes")
    return df


def main():
    dfs = []

    for fname in TARGET_FILES:
        file = RAW_DIR / fname
        if not file.exists():
            print(f"❌ Missing: {fname}")
            continue
        try:
            dfs.append(load_txt(file))
        except Exception as e:
            print(f"❌ Error {fname}: {e}")

    print("\n🔄 Concatenating...")
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Total: {len(full_df)} lignes")

    # ✅ Sauvegarde en | pour préserver les virgules décimales françaises
    full_df.to_csv(OUTPUT_FILE, index=False, sep="|")
    print(f"\n✅ Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()