# src/models/ml_top20.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

def main():
    print("🔄 Loading ML dataset...")
    df = pd.read_csv("data/processed/antibes_2024_ml.csv")
    print(f"{len(df)} biens chargés")

    print(df.columns)

    # Vérifier les colonnes nécessaires
    required_cols = ['type_local', 'surface_reelle_bati', 'valeur_fonciere', 'prix_m2']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Colonne manquante : {c}")

    # Nettoyage simple
    df = df.dropna(subset=required_cols)
    print(f"{len(df)} biens après nettoyage")

    # Features et target
    X = df[['type_local', 'surface_reelle_bati']]
    y = df['prix_m2']

    # Encode les variables catégorielles
    categorical_features = ['type_local']
    numeric_features = ['surface_reelle_bati']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_features),
            ('num', 'passthrough', numeric_features)
        ]
    )

    # Pipeline avec RandomForest
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=200, random_state=42))
    ])

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraînement
    model.fit(X_train, y_train)

    # Prédiction sur tout le dataset
    df['prix_m2_pred'] = model.predict(X)

    # Score d'investissement : prix réel / prix prédit (plus <1 = sous-évalué)
    df['score_investissement'] = df['prix_m2'] / df['prix_m2_pred']

    # Top 20 meilleurs investissements (score le plus faible = sous-évalué)
    top20 = df.sort_values('score_investissement').head(20)
    top20.to_csv("data/processed/top20_ml.csv", index=False)
    print("\n Top 20 ML sauvegardé : data/processed/top20_ml.csv")

    print("\n🔥 ----- TOP 20 INVESTISSEMENTS ML ----- 🔥\n")
    for i, row in top20.iterrows():
        print(f"Bien #{i}")
        print(f"Prix : {row['valeur_fonciere']:.0f} €")
        print(f"Surface : {row['surface_reelle_bati']} m²")
        print(f"Prix/m² : {row['prix_m2']:.0f} €")
        print(f"Prix/m² prédit : {row['prix_m2_pred']:.0f} €")
        print(f"Score : {row['score_investissement']:.3f}")
        print("----------------------------------------")

if __name__ == "__main__":
    main()