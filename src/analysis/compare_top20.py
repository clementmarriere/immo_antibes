import pandas as pd

# -----------------------------
# LOAD DATA
# -----------------------------
rule = pd.read_csv("data/processed/top20_antibes.csv")
ml = pd.read_csv("data/processed/top20_ml.csv")

print("\n📊 ----- COMPARAISON TOP 20 -----\n")

# -----------------------------
# 1. OVERLAP (biens communs)
# -----------------------------
common = set(rule.index).intersection(set(ml.index))

print(f"Biens communs (approx index) : {len(common)}")

# -----------------------------
# 2. COMPARAISON DES PRIX
# -----------------------------
print("\n--- RULE BASED ---")
print(rule["prix_m2"].describe())

print("\n--- ML ---")
print(ml["prix_m2"].describe())

# -----------------------------
# 3. TOP DIFFERENCES ML vs REALITY
# -----------------------------
ml["gap"] = ml["prix_m2_pred"] - ml["prix_m2"]

print("\n🔥 Top 5 anomalies ML (gap le plus négatif = sous-évalué)")
print(
    ml.sort_values("gap")[[
        "valeur_fonciere",
        "surface_reelle_bati",
        "prix_m2",
        "prix_m2_pred",
        "score_investissement"
    ]].head(5)
)

# -----------------------------
# 4. TOP RULE VS ML SIDE BY SIDE
# -----------------------------
print("\n📌 Rule-based TOP 5")
print(rule[["valeur_fonciere", "surface_reelle_bati", "prix_m2", "score"]].head(5))

print("\n📌 ML TOP 5")
print(ml[["valeur_fonciere", "surface_reelle_bati", "prix_m2", "score_investissement"]].head(5))