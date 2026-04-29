"""
src/models/forecast_geo.py
===========================
Forecast récursif 12 mois (jan-déc 2026) pour chaque quartier d'Antibes,
en utilisant les LSTM entraînés par lstm_geo.py.

Méthode (identique à forecast.py mais par quartier) :
  1. Reconstruit la dernière fenêtre connue (12 derniers mois) par quartier
  2. Forecast récursif 12 mois (déterministe)
  3. Monte Carlo Dropout n_runs=50 → IC 10%-90%

Sortie :
  - reports/forecast_2026_geo.csv  (long format : quartier, periode, prix, ci_low, ci_high)
  - reports/forecast_2026_geo_summary.csv (vs déc 2025, croissance %)

Usage :
  python src/models/forecast_geo.py
"""

import numpy as np
import pandas as pd
import pickle
import os
import re
from tensorflow import keras

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features", "geo")
MODELS_DIR   = os.path.join(BASE_DIR, "models", "geo")
DATA_DIR     = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR  = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

INPUT_FILE   = os.path.join(DATA_DIR, "antibes_geo_monthly.csv")

N_STEPS    = 12
N_FORECAST = 12
N_MC_RUNS  = 50

EXCLUDED_QUARTIERS = {"Autre"}

FEATURE_COLS = [
    "prix_m2_median",
    "volume",
    "surface_median",
    "nb_pieces_median",
    "mois_sin",
    "mois_cos",
]


# ── Helpers ──────────────────────────────────────────────────────────────────
def slugify(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def inverse_target(y_scaled, scaler):
    nb = scaler.n_features_in_
    dummy = np.zeros((len(y_scaled), nb), dtype=np.float32)
    dummy[:, 0] = y_scaled.ravel()
    return scaler.inverse_transform(dummy)[:, 0]


def build_last_window(df_q: pd.DataFrame, scaler) -> np.ndarray:
    """Reconstruit la fenêtre normalisée des 12 derniers mois pour un quartier."""
    df_q = df_q.sort_values("periode").reset_index(drop=True).copy()

    df_q["volume"] = df_q["volume"].fillna(0)
    for col in ["prix_m2_median", "surface_median", "nb_pieces_median"]:
        df_q[col] = df_q[col].ffill().bfill()

    df_q["mois_sin"] = np.sin(2 * np.pi * df_q["mois"] / 12)
    df_q["mois_cos"] = np.cos(2 * np.pi * df_q["mois"] / 12)

    data = df_q[FEATURE_COLS].values.astype(np.float32)
    data_sc = scaler.transform(data)
    return data_sc[-N_STEPS:].copy(), df_q["periode"].iloc[-1]


# ── Forecast récursif ────────────────────────────────────────────────────────
def recursive_forecast(model, scaler, last_window, last_date, training=False):
    """training=True active le dropout (utilisé pour MC Dropout)."""
    window = last_window.copy()
    preds_sc = []
    dates = []

    mean_volume  = window[:, 1].mean()
    mean_surface = window[:, 2].mean()
    mean_pieces  = window[:, 3].mean()
    current_date = pd.Timestamp(last_date)

    for _ in range(N_FORECAST):
        X = window[np.newaxis, :, :]
        if training:
            y = model(X, training=True).numpy()[0, 0]
        else:
            y = model.predict(X, verbose=0)[0, 0]
        preds_sc.append(y)

        next_date = current_date + pd.DateOffset(months=1)
        dates.append(next_date)
        next_sin = np.sin(2 * np.pi * next_date.month / 12)
        next_cos = np.cos(2 * np.pi * next_date.month / 12)

        new_row = np.array([y, mean_volume, mean_surface, mean_pieces,
                            next_sin, next_cos], dtype=np.float32)
        window = np.vstack([window[1:], new_row])
        current_date = next_date

    preds_sc = np.array(preds_sc, dtype=np.float32)
    return dates, inverse_target(preds_sc, scaler), preds_sc


def confidence_interval(model, scaler, last_window, last_date, n_runs=N_MC_RUNS):
    all_preds = []
    for _ in range(n_runs):
        _, _, preds_sc = recursive_forecast(
            model, scaler, last_window, last_date, training=True
        )
        all_preds.append(preds_sc)
    all_preds = np.array(all_preds)  # (n_runs, N_FORECAST)
    ci_low  = inverse_target(np.percentile(all_preds, 10, axis=0), scaler)
    ci_high = inverse_target(np.percentile(all_preds, 90, axis=0), scaler)
    return ci_low, ci_high


# ── Main ─────────────────────────────────────────────────────────────────────
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
        ckpt = os.path.join(MODELS_DIR, f"{slug}_best.keras")
        sc_path = os.path.join(FEATURES_DIR, slug, "scaler.pkl")

        if not os.path.exists(ckpt):
            print(f"  ⚠ {q} : modèle manquant ({ckpt}) — skip")
            continue

        print(f"  → {q}...", flush=True)
        model = keras.models.load_model(ckpt)
        with open(sc_path, "rb") as f:
            scaler = pickle.load(f)

        df_q = df[df["quartier"] == q].copy()
        last_window, last_date = build_last_window(df_q, scaler)

        # Forecast déterministe
        dates, preds, preds_sc = recursive_forecast(
            model, scaler, last_window, last_date, training=False
        )
        # IC Monte Carlo Dropout
        ci_low, ci_high = confidence_interval(model, scaler, last_window, last_date)

        # Prix de référence : dernier observé (déc 2025)
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

    out_long = os.path.join(REPORTS_DIR, "forecast_2026_geo.csv")
    out_sum  = os.path.join(REPORTS_DIR, "forecast_2026_geo_summary.csv")
    df_long.to_csv(out_long, index=False)
    df_summary.to_csv(out_sum, index=False)

    print(f"\n📊 Résumé 2026 par quartier :")
    print(df_summary.to_string(index=False))

    print(f"\n💾 Sauvegarde :")
    print(f"   {os.path.relpath(out_long, BASE_DIR)}")
    print(f"   {os.path.relpath(out_sum, BASE_DIR)}")
    print("\n✅  forecast_geo.py terminé. Prochaine étape : src/analysis/heatmap_forecast.py")


if __name__ == "__main__":
    main()
