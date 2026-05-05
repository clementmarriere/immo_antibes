"""
src/models/forecast_transformer_geo.py
========================================
Forecast récursif 12 mois (jan-déc 2026) pour chaque quartier d'Antibes,
en utilisant les Transformers entraînés par transformer_geo.py.

Identique à forecast_geo.py (méthode récursive + MC Dropout) mais charge
les checkpoints Transformer plutôt que LSTM.

Sortie :
  - reports/forecast_2026_transformer_geo.csv
  - reports/forecast_2026_transformer_geo_summary.csv

Usage :
  python src/models/forecast_transformer_geo.py
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import re
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forecast_geo import (
    build_last_window,
    recursive_forecast,
    confidence_interval,
    EXCLUDED_QUARTIERS,
    N_FORECAST,
    N_STEPS,
)
from transformer import build_transformer

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features", "geo")
MODELS_DIR   = os.path.join(BASE_DIR, "models", "geo")
DATA_DIR     = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR  = os.path.join(BASE_DIR, "reports")
INPUT_FILE   = os.path.join(DATA_DIR, "antibes_geo_monthly.csv")


def slugify(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def main():
    print("📂 Chargement antibes_geo_monthly.csv...")
    df = pd.read_csv(INPUT_FILE, parse_dates=["periode"])
    quartiers = [q for q in sorted(df["quartier"].unique())
                 if q not in EXCLUDED_QUARTIERS]
    print(f"   {len(quartiers)} quartiers : {quartiers}\n")

    rows = []
    summary = []

    for q in quartiers:
        slug = slugify(q)
        ckpt = os.path.join(MODELS_DIR, f"{slug}_transformer_best.keras")
        sc_path = os.path.join(FEATURES_DIR, slug, "scaler.pkl")

        if not os.path.exists(ckpt):
            print(f"  ⚠ {q} : checkpoint Transformer manquant ({ckpt}) — skip")
            continue

        print(f"  → {q}...", flush=True)
        with open(sc_path, "rb") as f:
            scaler = pickle.load(f)

        # Rebuild architecture + load weights uniquement (la couche Lambda
        # du Transformer n'est pas désérialisable via load_model standard).
        n_features = scaler.n_features_in_
        model = build_transformer((N_STEPS, n_features))
        model.load_weights(ckpt)

        df_q = df[df["quartier"] == q].copy()
        last_window, last_date = build_last_window(df_q, scaler)

        dates, preds, preds_sc = recursive_forecast(
            model, scaler, last_window, last_date, training=False
        )
        ci_low, ci_high = confidence_interval(model, scaler, last_window, last_date)

        last_price = float(df_q.sort_values("periode")["prix_m2_median"].iloc[-1])

        for d, p, lo, hi in zip(dates, preds, ci_low, ci_high):
            rows.append({
                "quartier": q,
                "periode": d.strftime("%Y-%m"),
                "prix_predit": round(float(p), 0),
                "ci_low_10pct": round(float(lo), 0),
                "ci_high_90pct": round(float(hi), 0),
                "ref_dec_2025": round(last_price, 0),
                "growth_pct_vs_dec25": round((p - last_price) / last_price * 100, 2),
            })

        mean_pred = float(np.mean(preds))
        summary.append({
            "quartier": q,
            "ref_dec_2025": round(last_price, 0),
            "mean_2026": round(mean_pred, 0),
            "min_2026": round(float(np.min(preds)), 0),
            "max_2026": round(float(np.max(preds)), 0),
            "growth_pct_mean": round((mean_pred - last_price) / last_price * 100, 2),
            "growth_pct_dec26": round((preds[-1] - last_price) / last_price * 100, 2),
        })
        print(f"     déc 25 = {last_price:.0f} €/m²  →  moy 2026 = {mean_pred:.0f} €/m²  "
              f"({(mean_pred - last_price) / last_price * 100:+.1f}%)")

    df_long = pd.DataFrame(rows)
    df_summary = pd.DataFrame(summary)

    out_long = os.path.join(REPORTS_DIR, "forecast_2026_transformer_geo.csv")
    out_sum  = os.path.join(REPORTS_DIR, "forecast_2026_transformer_geo_summary.csv")
    df_long.to_csv(out_long, index=False)
    df_summary.to_csv(out_sum, index=False)

    print(f"\n📊 Résumé Transformer 2026 par quartier :")
    print(df_summary.to_string(index=False))

    print(f"\n💾 Sauvegarde :")
    print(f"   {os.path.relpath(out_long, BASE_DIR)}")
    print(f"   {os.path.relpath(out_sum, BASE_DIR)}")
    print("\n✅  forecast_transformer_geo.py terminé.")


if __name__ == "__main__":
    main()
