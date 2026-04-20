# Real Estate Investment Analysis in Antibes

## Overview

This project analyzes real estate transactions in Antibes using DVF (Demande de Valeurs Foncières) data.

The goal is to:
	•	Clean and process raw transaction data
	•	Compute descriptive statistics (price per m², distributions, etc.)
	•	Identify potentially good investments using:
	•	Rule-based methods
	•	Machine Learning (anomaly detection on price/m²)


## Project Structure

api/                    # (future) API to expose results

data/
│
├── raw/
│   └── ValeursFoncieres-2024.txt   # Raw DVF data
│
├── processed/
│   ├── antibes_2024.csv           # Filtered data (Antibes only)
│   ├── antibes_2024_clean.csv     # Cleaned dataset
│   ├── antibes_2024_ml.csv        # Dataset for ML
│   └── top20_antibes.csv          # Top 20 (rule-based)

notebooks/             # Exploration / experiments

src/
│
├── etl/
│   ├── filter_antibes.py          # Extract Antibes data from DVF
│   └── clean_dataset.py           # Clean and prepare dataset
│
├── analysis/
│   ├── metrics_full.py            # Stats on raw filtered data
│   ├── metrics_clean.py           # Stats on cleaned data
│   └── top20_investments.py       # Rule-based top 20 investments
│
├── features/
│   └── prepare_ml.py              # Prepare dataset for ML
│
├── models/
│   └── ml_top20.py                # ML model + investment scoring
│
└── scoring/                       # (future) scoring logic


## Data Pipeline

### 1. Raw Data

File:
data/raw/ValeursFoncieres-2024.txt

Contains all real estate transactions in France.


### 2. Filter Antibes

Script:
src/etl/filter_antibes.py

Output:
data/processed/antibes_2024.csv

Keeps only:
	•	Transactions in Antibes
	•	Relevant columns


### 3. Clean Dataset

Script:
src/etl/clean_dataset.py

Output:
data/processed/antibes_2024_clean.csv

Performs:
	•	Removal of invalid values
	•	Cleaning of surfaces and prices
	•	Creation of:
	•	prix_m2 = valeur_fonciere / surface_reelle_bati


### 4. Descriptive Analysis

Scripts:
	•	metrics_full.py → stats on raw filtered data
	•	metrics_clean.py → stats on cleaned data

Examples:
	•	Average price/m²
	•	Median
	•	Distribution


### 5. Rule-Based Investment Detection

Script:
src/analysis/top20_investments.py

Output:
data/processed/top20_antibes.csv

Logic:
	•	Score based on simple heuristics (e.g. price/m² vs surface)
	•	Returns top 20 “interesting” properties

Based only on past data (not predictive)


### 6. ML Dataset Preparation

Script:
src/features/prepare_ml.py

Output:
data/processed/antibes_2024_ml.csv

Contains:
	•	type_local (Appartement / Maison)
	•	surface_reelle_bati
	•	valeur_fonciere
	•	prix_m2

Cleaned and ready for modeling

### 7. Machine Learning Model

Script:
src/models/ml_top20.py

Steps:
	1.	Load ML dataset
	2.	Encode categorical variables (type_local)
	3.	Train a Random Forest model to predict prix_m2
	4.	Compute predicted price per m²:
prix_m2_pred
	5.	Compute investment score:
score = prix_m2 / prix_m2_pred


## Investment Score Interpretation

Score       Meaning
< 1         Underpriced (potential opportunity)
≈ 1         Fair price
> 1         Overpriced

## Comparison rule-based vs ml models

### 1. Rule-based model
- Uses handcrafted heuristics
- Filters properties based on:
  - budget constraints
  - minimum surface
  - price per m² normalization
- Produces a score based on:
  - low price per m²
  - larger surface preference

Interpretable and deterministic approach


### 2. Machine Learning model
- Uses Random Forest regression
- Predicts expected price per m²
- Computes investment score:

score = actual_price_m2 / predicted_price_m2

Detects statistical anomalies rather than explicit rules


### 3. Key insight

Both models operate on the same filtered dataset, but:
- Rule-based model relies on explicit human logic
- ML model learns patterns from historical data

They often converge on similar property subsets, showing that filtering dominates selection more than modeling differences.

## Limitations

Current model uses only:
	•	Property type
	•	Surface

It does NOT include:
	•	Location (very important)
	•	Property condition
	•	Floor, view, etc.

Therefore:
	•	Results highlight anomalies
	•	Not guaranteed “good deals”


## Future Improvements
	•	Add location features (postal code, coordinates)
	•	Estimate rental yield
	•	Connect with real-time listings (Leboncoin, etc.)
	•	Build API for querying properties


## Key Insight

This project evolves from:
	•	Descriptive analytics → “what happened”
	•	To predictive modeling → “what should this be worth”
