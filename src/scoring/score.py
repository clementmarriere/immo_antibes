"""
src/scoring/score.py
=====================
Calcule le "Score de Dynamique" du marché immobilier d'Antibes.

Principe :
  score = (prix_prédit_LSTM - prix_actuel) / prix_actuel * 100

  > 0  → marché en phase d'accélération (sous-évalué par rapport à la tendance)
  < 0  → marché en phase de ralentissement ou surévaluation

Le script travaille sur le test set (11 derniers mois) et génère :
  - Un tableau des scores mois par mois
  - Une visualisation du Score de Dynamique
  - Une interprétation automatique par seuils

Usage :
  python src/scoring/score.py
"""

import numpy as np
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Chemins ───────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
DATA_DIR     = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR   = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Seuils du Score de Dynamique ──────────────────────────────────────────────
SEUIL_FORT    =  3.0   # % → accélération forte
SEUIL_MODERE  =  1.0   # % → accélération modérée
SEUIL_STABLE  = -1.0   # % → marché stable
# < SEUIL_STABLE → ralentissement

COLORS = {
    "accel_forte"   : "#16A34A",   # vert foncé
    "accel_moderee" : "#86EFAC",   # vert clair
    "stable"        : "#FCD34D",   # jaune
    "ralentissement": "#EF4444",   # rouge
}

LABELS = {
    "accel_forte"   : "Accélération forte  (> +3%)",
    "accel_moderee" : "Accélération modérée (+1% à +3%)",
    "stable"        : "Marché stable        (-1% à +1%)",
    "ralentissement": "Ralentissement       (< -1%)",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_data():
    with open(os.path.join(FEATURES_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    def inv(y):
        nb = scaler.n_features_in_
        dummy = np.zeros((len(y), nb), dtype=np.float32)
        dummy[:, 0] = y.ravel()
        return scaler.inverse_transform(dummy)[:, 0]

    y_test      = inv(np.load(os.path.join(MODELS_DIR, "y_test.npy")))
    y_pred_lstm = inv(np.load(os.path.join(MODELS_DIR, "y_pred_lstm.npy")))

    # Récupération des dates du test set depuis le CSV mensuel
    csv_path = os.path.join(DATA_DIR, "antibes_appart_monthly.csv")
    df = pd.read_csv(csv_path, parse_dates=["periode"])
    df = df.sort_values("periode").reset_index(drop=True)

    n_total = len(df)
    n_test  = len(y_test)
    # Les fenêtres de test correspondent aux derniers mois
    # (après le split 70/15/15 + n_steps=12)
    # Index de départ dans le CSV : n_total - n_test
    dates = df["periode"].iloc[n_total - n_test:].reset_index(drop=True)

    return y_test, y_pred_lstm, dates


def classify(score):
    if score >= SEUIL_FORT:
        return "accel_forte"
    elif score >= SEUIL_MODERE:
        return "accel_moderee"
    elif score >= SEUIL_STABLE:
        return "stable"
    else:
        return "ralentissement"


# ── Calcul du Score ───────────────────────────────────────────────────────────
def compute_scores(y_true, y_pred, dates):
    scores = (y_pred - y_true) / y_true * 100

    df = pd.DataFrame({
        "periode"       : dates,
        "prix_actuel"   : np.round(y_true, 0),
        "prix_predit"   : np.round(y_pred, 0),
        "score_dynamique": np.round(scores, 2),
    })
    df["signal"] = df["score_dynamique"].apply(classify)
    df["label"]  = df["signal"].map(LABELS)

    return df


# ── Visualisation ─────────────────────────────────────────────────────────────
def plot_score(df):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle("Score de Dynamique — Marché Immobilier Antibes",
                 fontsize=14, fontweight="bold")

    x      = np.arange(len(df))
    labels = df["periode"].dt.strftime("%b %Y")

    # ── Subplot 1 : barres du score colorées ──────────────────────────────────
    ax = axes[0]
    bar_colors = [COLORS[s] for s in df["signal"]]
    bars = ax.bar(x, df["score_dynamique"], color=bar_colors,
                  edgecolor="white", linewidth=0.8, alpha=0.9)

    ax.axhline(0,              color="black", linewidth=1.2)
    ax.axhline(SEUIL_FORT,     color=COLORS["accel_forte"],    linewidth=1,
               linestyle="--", alpha=0.6, label=f"+{SEUIL_FORT}% seuil fort")
    ax.axhline(SEUIL_MODERE,   color=COLORS["accel_moderee"],  linewidth=1,
               linestyle="--", alpha=0.6, label=f"+{SEUIL_MODERE}% seuil modéré")
    ax.axhline(SEUIL_STABLE,   color=COLORS["ralentissement"], linewidth=1,
               linestyle="--", alpha=0.6, label=f"{SEUIL_STABLE}% seuil stable")

    # Annotations valeurs
    for bar, val in zip(bars, df["score_dynamique"]):
        ypos = bar.get_height() + 0.1 if val >= 0 else bar.get_height() - 0.4
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:+.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Score de Dynamique (%)")
    ax.set_title("Score mensuel : (prix prédit − prix actuel) / prix actuel × 100",
                 fontsize=10, color="gray")
    ax.grid(True, axis="y", alpha=0.3)

    # Légende couleurs
    patches = [mpatches.Patch(color=COLORS[k], label=LABELS[k])
               for k in ["accel_forte", "accel_moderee", "stable", "ralentissement"]]
    ax.legend(handles=patches, loc="lower left", fontsize=8)

    # ── Subplot 2 : prix actuel vs prix prédit ────────────────────────────────
    ax2 = axes[1]
    ax2.plot(x, df["prix_actuel"], color="#111827", linewidth=2,
             marker="o", markersize=5, label="Prix actuel (€/m²)")
    ax2.plot(x, df["prix_predit"], color="#2563EB", linewidth=2,
             marker="s", markersize=4, linestyle="--", label="Prix prédit LSTM (€/m²)")
    ax2.fill_between(x, df["prix_actuel"], df["prix_predit"],
                     alpha=0.08, color="#2563EB")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Prix/m² (€)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "05_score_dynamique.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 {path}")


# ── Rapport texte ─────────────────────────────────────────────────────────────
def print_report(df):
    print("\n" + "═" * 60)
    print("  SCORE DE DYNAMIQUE — ANTIBES APPARTEMENTS")
    print("═" * 60)
    print(f"{'Période':<12} {'Actuel':>10} {'Prédit':>10} {'Score':>8}  Signal")
    print("─" * 60)
    for _, row in df.iterrows():
        periode = row["periode"].strftime("%b %Y")
        signal_icon = {
            "accel_forte"   : "🟢🟢",
            "accel_moderee" : "🟢",
            "stable"        : "🟡",
            "ralentissement": "🔴",
        }[row["signal"]]
        print(f"{periode:<12} {row['prix_actuel']:>8.0f} €  "
              f"{row['prix_predit']:>8.0f} €  "
              f"{row['score_dynamique']:>+6.1f}%  {signal_icon}")

    print("─" * 60)
    print(f"\n📊 Score moyen  : {df['score_dynamique'].mean():+.1f}%")
    print(f"   Score max    : {df['score_dynamique'].max():+.1f}% "
          f"({df.loc[df['score_dynamique'].idxmax(), 'periode'].strftime('%b %Y')})")
    print(f"   Score min    : {df['score_dynamique'].min():+.1f}% "
          f"({df.loc[df['score_dynamique'].idxmin(), 'periode'].strftime('%b %Y')})")

    dist = df["signal"].value_counts()
    print(f"\n📈 Distribution des signaux :")
    for signal, label in LABELS.items():
        count = dist.get(signal, 0)
        print(f"   {label:<45} : {count} mois")

    print("\n💡 Interprétation :")
    score_moyen = df["score_dynamique"].mean()
    if score_moyen >= SEUIL_FORT:
        print("   → Le marché est en forte accélération. "
              "Le modèle anticipe une hausse significative.")
    elif score_moyen >= SEUIL_MODERE:
        print("   → Le marché est en légère accélération. "
              "Tendance haussière modérée détectée.")
    elif score_moyen >= SEUIL_STABLE:
        print("   → Le marché est stable. "
              "Pas de signal fort d'accélération ou de ralentissement.")
    else:
        print("   → Le modèle détecte un ralentissement. "
              "Les prix prédits sont inférieurs aux prix actuels.")
    print("═" * 60)


# ── Sauvegarde CSV ────────────────────────────────────────────────────────────
def save_csv(df):
    path = os.path.join(BASE_DIR, "reports", "score_dynamique.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"💾 {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("📂 Chargement des données...")
    y_test, y_pred_lstm, dates = load_data()

    print("🧮 Calcul du Score de Dynamique...")
    df = compute_scores(y_test, y_pred_lstm, dates)

    print_report(df)
    plot_score(df)
    save_csv(df)

    print("\n✅ score.py terminé.")
    print("   → reports/figures/05_score_dynamique.png")
    print("   → reports/score_dynamique.csv")
    print("\n➡️  Prochaines étapes : rapport 4 pages + carte thermique")


if __name__ == "__main__":
    main()