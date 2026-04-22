# 🏡 Real Estate Forecasting — Antibes

> Prédiction de la dynamique du marché immobilier à Antibes par Deep Learning (LSTM) à partir des données DVF 2014-2025.

---

## 📌 Objectif

Ce projet explore l'utilisation des réseaux de neurones récurrents pour modéliser l'évolution temporelle des prix immobiliers à Antibes. À partir de la base DVF (Demandes de Valeurs Foncières), nous construisons une série temporelle mensuelle et entraînons un modèle LSTM pour anticiper les cycles du marché à un horizon de 12 mois.

Un **Score de Dynamique** est calculé en comparant les prix prédits aux prix actuels, permettant d'identifier mathématiquement les quartiers en phase d'accélération ou de ralentissement.

---

## 🗂 Structure du projet

```
immo_antibes/
├── data/
│   ├── raw/                        # Fichiers DVF bruts + DVF géolocalisés + GeoJSON quartiers
│   └── processed/                  # Données nettoyées et agrégées (ignoré par git)
├── src/
│   ├── etl/
│   │   ├── merge_dvf.py            # Fusion des fichiers DVF bruts 2014-2025
│   │   ├── filter_antibes.py       # Filtrage commune Antibes + nettoyage + outliers
│   │   ├── aggregate_monthly.py    # Agrégation mensuelle avec interpolation
│   │   └── merge_dvf_geo.py        # DVF géolocalisées + assignation quartiers GeoJSON
│   ├── features/
│   │   └── build_features.py       # Sliding windows, encodage cyclique, MinMaxScaler, split
│   ├── models/
│   │   └── lstm.py                 # LSTM + MLP + baseline Moyenne Mobile + évaluation
│   ├── analysis/
│   │   ├── eda.py                  # Analyse exploratoire (tendances, saisonnalité, YoY)
│   │   ├── plot_results.py         # Courbes de loss, prédictions vs réalité, résidus
│   │   └── heatmap.py              # Carte thermique et trajectoires par quartier
│   └── scoring/
│       └── score.py                # Score de Dynamique mensuel par seuils
├── models/                         # Poids sauvegardés (.keras) — ignoré par git
├── reports/
│   ├── figures/                    # Figures générées — ignoré par git
│   └── score_dynamique.csv         # Export score mensuel — ignoré par git
├── .gitignore
└── README.md
```

---

## 🔄 Pipeline principal

```
DVF bruts (2014-2025)
      ↓  merge_dvf.py
      ↓  filter_antibes.py          → 10 731 transactions propres
      ↓  aggregate_monthly.py       → 144 mois continus (0 mois manquants)
      ↓  build_features.py          → X_train(88,12,6) / X_val(9,12,6) / X_test(11,12,6)
      ↓  lstm.py                    → LSTM entraîné + évaluation comparative
      ↓  plot_results.py            → Courbes de loss + analyse des erreurs
      ↓  score.py                   → Score de Dynamique mensuel

DVF géolocalisées (2014-2025)
      ↓  merge_dvf_geo.py           → 19 689 transactions + assignation quartiers GPS
      ↓  heatmap.py                 → Carte thermique + trajectoires 2014-2025
```

---

## 📊 Résultats

### Métriques sur le test set (11 mois)

| Modèle | MAE | RMSE | MAPE |
|---|---|---|---|
| LSTM | 237 €/m² | 274 €/m² | 4.5% |
| Moyenne Mobile k=3 | 164 €/m² | 205 €/m² | 3.1% |
| MLP | 860 €/m² | 900 €/m² | 16.3% |

Le LSTM ne bat pas la moyenne mobile sur ce jeu de test — résultat attendu avec 88 fenêtres d'entraînement et un signal très auto-corrélé. Le MLP confirme l'importance de la structure temporelle.

### Évolution des prix par quartier (2014 → 2025)

| Quartier | Prix médian 2014 | Prix médian 2025 | Croissance |
|---|---|---|---|
| Vieille Ville | 5 089 €/m² | 7 634 €/m² | **+50%** |
| Centre Ville | 3 779 €/m² | 5 195 €/m² | **+37%** |
| Juan-les-Pins | 3 927 €/m² | 5 163 €/m² | **+31%** |
| Antibes Nord Ouest | 4 056 €/m² | 4 911 €/m² | +21% |
| Cap d'Antibes | 5 692 €/m² | 6 636 €/m² | +17% |
| La Fontonne | 4 446 €/m² | 5 155 €/m² | +16% |

---

## 🧠 Architecture LSTM

```
Input (12, 6)
  └─ LSTM(64, return_sequences=True)
  └─ Dropout(0.2)
  └─ LSTM(32)
  └─ Dropout(0.2)
  └─ Dense(16, relu)
  └─ Dense(1)          ← prix_m2_median normalisé
```

**Features** : `prix_m2_median`, `volume`, `surface_median`, `nb_pieces_median`, `mois_sin`, `mois_cos`

**Split** : 70% train / 15% val / 15% test (chronologique, sans shuffle)

**Callbacks** : EarlyStopping (patience=20) + ReduceLROnPlateau (patience=10) + ModelCheckpoint

---

## 🗺 Quartiers

Les quartiers sont définis par des polygones GeoJSON tracés manuellement sur [geojson.io](https://geojson.io), disponibles dans `data/raw/map.geojson`.

| Quartier | Transactions | Prix médian moyen |
|---|---|---|
| Antibes Nord Ouest | 6 330 | 4 296 €/m² |
| Juan-les-Pins | 4 935 | 4 339 €/m² |
| Centre Ville | 2 415 | 4 268 €/m² |
| La Fontonne | 2 335 | 4 504 €/m² |
| Cap d'Antibes | 2 235 | 5 966 €/m² |
| Vieille Ville | 1 427 | 6 161 €/m² |

---

## ⚡ Installation

```bash
git clone https://github.com/clementmarriere/immo_antibes.git
cd immo_antibes
pip install pandas numpy scikit-learn tensorflow matplotlib geopandas shapely
```

### Reproduire le pipeline complet

```bash
# ETL
python src/etl/merge_dvf.py
python src/etl/filter_antibes.py
python src/etl/aggregate_monthly.py
python src/etl/merge_dvf_geo.py

# Features
python src/features/build_features.py

# Modèle
python src/models/lstm.py

# Visualisations
python src/analysis/eda.py
python src/analysis/plot_results.py
python src/analysis/heatmap.py

# Score de Dynamique
python src/scoring/score.py
```

---

## 🗃 Scripts legacy (pré-Deep Learning)

Scripts de la phase d'exploration initiale sur les données 2024 uniquement. Non utilisés dans le pipeline actuel.

| Fichier | Description |
|---|---|
| `src/etl/clean_dataset_2024.py` | Nettoyage basique dataset 2024 |
| `src/features/build_timeseries.py` | Agrégation mensuelle version initiale |
| `src/features/prepare_ml.py` | Préparation features ML classique |
| `src/models/ml_top20.py` | Modèle ML classique sur top 20 investissements |
| `src/analysis/top20_investments.py` | Identification des 20 meilleurs investissements 2024 |
| `src/analysis/compare_top20.py` | Comparaison top 20 |
| `src/analysis/metrics_clean.py` | Métriques phase exploratoire |
| `src/analysis/metrics_full.py` | Métriques complètes phase exploratoire |

---

## 📉 Limites

- **Volume** : 88 fenêtres d'entraînement — trop peu pour un LSTM profond
- **Val set** : 9 fenêtres → signal de validation bruité, early stopping instable
- **Variables exogènes absentes** : taux BCE, inflation, IPC — le modèle ne peut pas anticiper les chocs macro
- **Type de bien** : appartements uniquement, pas de maisons ni locaux commerciaux
- **Biais LSTM** : sous-estimation systématique de -174 €/m² sur le test set — le Score de Dynamique doit être interprété avec cette limite en tête

---

## 📁 Données sources

- DVF brutes : [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/)
- DVF géolocalisées : [files.data.gouv.fr/geo-dvf](https://files.data.gouv.fr/geo-dvf/latest/csv/)

> Les fichiers de données bruts ne sont pas inclus dans ce dépôt (taille > 1 Go).

---

## ✍️ Auteur

Clément Marrière — Projet Deep Learning, 2026