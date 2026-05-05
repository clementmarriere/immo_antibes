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
│   │   ├── build_features.py       # Sliding windows, encodage cyclique, MinMaxScaler, split
│   │   └── build_features_geo.py   # Idem mais par quartier (6 datasets indépendants)
│   ├── models/
│   │   ├── lstm.py                 # LSTM + MLP + baseline Moyenne Mobile + évaluation
│   │   ├── gru.py                  # GRU(64,32) — variante allégée du LSTM
│   │   ├── transformer.py          # Encoder Transformer avec attention multi-tête
│   │   ├── cv_lstm.py              # TimeSeriesSplit 5-fold + IC sur métriques (LSTM seul)
│   │   ├── cv_compare.py           # CV comparative LSTM / GRU / Transformer / MovAvg
│   │   ├── cv_ensemble.py          # Ensemble (mean / median / NNLS) vs modèles de base
│   │   ├── lstm_geo.py             # 6 LSTM indépendants (un par quartier)
│   │   ├── forecast_geo.py         # Forecast récursif 12 mois 2026 par quartier + MC Dropout
│   │   ├── transformer_geo.py      # 6 Transformers indépendants par quartier
│   │   └── forecast_transformer_geo.py  # Forecast Transformer 2026 par quartier
│   ├── analysis/
│   │   ├── eda.py                  # Analyse exploratoire (tendances, saisonnalité, YoY)
│   │   ├── plot_results.py         # Courbes de loss, prédictions vs réalité, résidus
│   │   ├── heatmap.py              # Carte thermique et trajectoires par quartier (historique)
│   │   ├── heatmap_forecast.py     # Carte thermique prédictive 2026 (LSTM)
│   │   └── heatmap_forecast_compare.py  # Heatmap + barplot LSTM vs Transformer 2026
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
      ↓  build_features_geo.py      → 6 datasets indépendants par quartier
      ↓  lstm_geo.py                → 6 LSTM entraînés (un par zone)
      ↓  forecast_geo.py            → Forecast LSTM 2026 par quartier + IC MC Dropout
      ↓  transformer_geo.py         → 6 Transformers entraînés (un par zone)
      ↓  forecast_transformer_geo   → Forecast Transformer 2026 par quartier
      ↓  heatmap_forecast.py        → Carte thermique prédictive 2026 (LSTM)
      ↓  heatmap_forecast_compare   → Heatmap + barplot LSTM vs Transformer
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

### Cross-validation temporelle — comparaison de 4 architectures

Évaluation sur 5 folds glissants TimeSeriesSplit (test = 12 mois : 2021, 2022, 2023, 2024, 2025), hyperparams identiques pour les 3 modèles deep (epochs ≤ 200, patience 20, seed 42).

| Modèle | Paramètres | MAE (mean ± std) | RMSE (mean ± std) | MAPE (mean ± std) |
|---|---|---|---|---|
| **MovAvg(k=3)** | — | **243 ± 60 €/m²** | **289 ± 68 €/m²** | **4.87 ± 1.27%** |
| **Transformer** | **8 801** | **323 ± 175 €/m²** | **379 ± 187 €/m²** | **6.4 ± 3.6%** |
| GRU | 23 777 | 340 ± 139 €/m² | 396 ± 154 €/m² | 6.6 ± 2.5% |
| LSTM | 31 137 | 360 ± 146 €/m² | 427 ± 153 €/m² | 6.9 ± 2.7% |

**MAE par fold (lecture détaillée)** :

| Fold | Test | LSTM | GRU | Transformer | MovAvg |
|---|---|---|---|---|---|
| F1 | 2021 | 234 | 234 | 300 | 243 |
| F2 | 2022 | 347 | 346 | 610 | **293** |
| F3 | 2023 (choc BCE) | 603 | 566 | **313** | 303 |
| F4 | 2024 | 353 | 332 | 258 | **220** |
| F5 | 2025 | 260 | 220 | **135** | 156 |

### Ensembles : peut-on combiner les architectures ?

Trois stratégies d'ensemble testées sur les mêmes 5 folds (chaque modèle prédit sur le test, on combine) :

| Stratégie | MAE (mean ± std) | Position |
|---|---|---|
| **MovAvg(k=3)** | **243 ± 60 €/m²** | 🥇 1er global |
| **Ensemble_median** | **265 ± 82 €/m²** | 🥈 2e — bat tous les modèles deep individuels |
| Ensemble_mean | 267 ± 81 €/m² | 🥉 3e |
| Transformer | 323 ± 175 €/m² | 4e |
| Ensemble_NNLS | 338 ± 161 €/m² | 5e — overfit sur val (n=12 trop petit) |

**Conclusion clé** : sur le fold 2025 (le plus récent, donc avec le plus d'historique), **Ensemble_median MAE=140 < MovAvg=156** — l'ensemble bat enfin le baseline. Sur la moyenne des 5 folds, MovAvg garde une avance étroite (243 vs 265, +9%). Les ensembles **réduisent significativement la variance** (std 82 vs 138-175 pour les modèles individuels) — propriété attendue d'un mélange de prédictions partiellement décorrélées.

**Quatre enseignements clés** :
- **L'architecture compte plus que la profondeur** : le Transformer bat LSTM et GRU avec **3.5× moins de paramètres**
- **Transformer = premier modèle deep à battre MovAvg sur un fold** : sur 2025 (fold 5), MAE=135 €/m² < MovAvg=156 €/m²
- **Transformer survit au choc BCE 2023** (MAE=313) là où LSTM (603) et GRU (566) s'effondrent → l'attention multi-tête généralise mieux à un régime de marché différent
- **MovAvg reste compétitif** par sa stabilité (std=60 contre 139-175 pour les modèles deep) → confirme la difficulté du Deep Learning sur séries courtes (Makridakis 2018)

### Modélisation par quartier (LSTM zone-spécifique)

Un LSTM dédié est entraîné indépendamment pour chacun des 6 quartiers à partir des DVF géolocalisées. Même architecture que le modèle global, même seed (42), même split chronologique 70/15/15.

| Quartier | LSTM MAE | MovAvg MAE | LSTM MAPE | LSTM bat baseline |
|---|---|---|---|---|
| Centre Ville | 320 €/m² | 411 €/m² | 6.6% | ✅ |
| Juan-les-Pins | 333 €/m² | 418 €/m² | 6.3% | ✅ |
| Vieille Ville | 822 €/m² | 842 €/m² | 10.4% | ✅ |
| Antibes Nord Ouest | 188 €/m² | 161 €/m² | 3.9% | ❌ |
| La Fontonne | 631 €/m² | 329 €/m² | 11.8% | ❌ |
| Cap d'Antibes | 702 €/m² | 510 €/m² | 9.7% | ❌ |

Le LSTM bat la moyenne mobile sur 3/6 quartiers. Les écarts importants sur Cap d'Antibes et La Fontonne s'expliquent par la forte variance des transactions et le faible volume mensuel (~17 ventes/mois en médian).

### Forecast 2026 par quartier (vs déc 2025)

| Quartier | Croissance moyenne 2026 |
|---|---|
| Centre Ville | **+8.8%** |
| Antibes Nord Ouest | +2.1% |
| Vieille Ville | −0.8% |
| Juan-les-Pins | −6.3% |
| La Fontonne | −11.9% |
| Cap d'Antibes | −19.2% |

Les baisses prédites sur Cap d'Antibes et Vieille Ville reflètent davantage des **artefacts du modèle** (régression vers la moyenne pour Cap d'Antibes dont déc 2025 = pic local à 7286 €/m² ; drift du forecast récursif visible sur Vieille Ville à partir de septembre 2026) qu'un signal de marché. Détails dans le rapport § 4.6.

### Forecast 2026 par quartier — comparaison LSTM vs Transformer

Le Transformer (qui survit nettement mieux au choc BCE 2023 — voir CV § 4.2) a aussi été appliqué par quartier. Les prédictions divergent sensiblement :

| Quartier | LSTM | Transformer | Écart |
|---|---|---|---|
| Vieille Ville | −0.8% | **+37.1%** | **+38 pts** |
| Centre Ville | +8.8% | +7.1% | −2 pts |
| La Fontonne | −11.9% | **+4.6%** | **+17 pts** |
| Juan-les-Pins | −6.3% | −1.2% | +5 pts |
| Cap d'Antibes | −19.2% | −1.9% | **+17 pts** |
| Antibes Nord Ouest | +2.1% | −0.5% | −3 pts |

Le Transformer est **systématiquement plus optimiste** que le LSTM (sauf Centre Ville et Antibes Nord Ouest où l'écart est faible). Sur Vieille Ville, les deux modèles diffèrent de **38 points** — illustration spectaculaire de la **sensibilité du forecast au choix d'architecture**, et argument pour considérer un ensemble plutôt qu'un modèle unique en production.

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

# Modèles alternatifs (GRU, Transformer)
python src/models/gru.py
python src/models/transformer.py

# Cross-validation temporelle (LSTM seul, ou comparaison 4 modèles)
python src/models/cv_lstm.py
python src/models/cv_compare.py

# Modélisation par quartier
python src/features/build_features_geo.py
python src/models/lstm_geo.py
python src/models/forecast_geo.py
python src/analysis/heatmap_forecast.py
```

### Figures générées (`reports/figures/`)

| # | Fichier | Description |
|---|---|---|
| 01-04 | `01_loss.png` à `04_compare.png` | Loss, prédictions, résidus, comparatif modèles |
| 05 | `05_score_dynamique.png` | Score de Dynamique mensuel |
| 06-09 | `06_heatmap_*.png` à `09_comparaison_*.png` | Heatmap historique + trajectoires par quartier |
| 10 | `10_forecast_2026.png` | Forecast LSTM global jan-déc 2026 + IC MC Dropout |
| **11** | **`11_heatmap_forecast_2026.png`** | **Carte thermique prédictive 2026 par quartier** |
| **12** | **`12_trajectoires_forecast_2026.png`** | **Trajectoires forecast par quartier + IC** |
| **13** | **`13_timeseries_cv.png`** | **Cross-validation TimeSeriesSplit — MAE par fold + IC sur métriques** |
| **14** | **`14_cv_compare.png`** | **Comparaison LSTM / GRU / Transformer / MovAvg sur 5 folds** |
| **15** | **`15_heatmap_forecast_compare.png`** | **Heatmap forecast 2026 LSTM vs Transformer côte à côte** |
| **16** | **`16_growth_2026_compare.png`** | **Croissance moyenne 2026 par quartier — barplot LSTM vs Transformer** |
| **17** | **`17_cv_ensemble.png`** | **CV ensemble (mean/median/NNLS) vs modèles de base** |

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
- **Forecast récursif** : l'erreur s'accumule à chaque pas, visible sur Vieille Ville après le mois 8 du forecast 2026 — motive l'investigation d'une architecture Seq2Seq directe
- **Régression vers la moyenne** : le LSTM ramène les quartiers à pic local (Cap d'Antibes déc 2025 = 7286 €/m² vs moyenne historique 5500 €/m²) vers leur moyenne d'entraînement, produisant des baisses prédites qui ne reflètent pas le marché

---

## 📁 Données sources

- DVF brutes : [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/), [data.cquest.org](data.cquest.org/dgfip_dvf/)
- DVF géolocalisées : [files.data.gouv.fr/geo-dvf](https://files.data.gouv.fr/geo-dvf/latest/csv/)

> Les fichiers de données bruts ne sont pas inclus dans ce dépôt (taille > 1 Go).

---

## ✍️ Auteurs

Clément Marrière
Dimitri Gardarin

— Projet Deep Learning, 2026
