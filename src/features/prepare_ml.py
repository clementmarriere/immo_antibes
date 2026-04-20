# src/features/prepare_ml.py

import pandas as pd

def main():
    print("🔄 Loading clean dataset...")
    # Charger le dataset clean
    df = pd.read_csv("data/processed/antibes_2024_clean.csv")
    
    # Vérifier les colonnes nécessaires
    required_cols = ['type_local', 'surface_reelle_bati', 'valeur_fonciere', 'prix_m2']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Colonne manquante : {c}")
    
    # Filtrer les colonnes utiles pour le ML
    df_ml = df[required_cols].copy()
    
    # Supprimer les lignes invalides (surface nulle, prix_m2 nul)
    df_ml = df_ml[(df_ml['surface_reelle_bati'] > 0) & (df_ml['prix_m2'] > 0)]
    
    # Optionnel : réinitialiser l'index
    df_ml.reset_index(drop=True, inplace=True)
    
    # Sauvegarder le dataset ML
    df_ml.to_csv("data/processed/antibes_2024_ml.csv", index=False)
    print(f"✅ Dataset ML créé avec {len(df_ml)} biens : data/processed/antibes_2024_ml.csv")

    print(df.columns)

if __name__ == "__main__":
    main()