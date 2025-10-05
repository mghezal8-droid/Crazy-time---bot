import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random

# -------------------------------
# Config Streamlit
# -------------------------------
st.set_page_config(page_title="Super Crazy Time Bot", layout="wide")

# -------------------------------
# Initialisation Session State
# -------------------------------
if "bankroll" not in st.session_state:
    st.session_state.bankroll = 100.0
if "initial_bankroll" not in st.session_state:
    st.session_state.initial_bankroll = st.session_state.bankroll
if "history" not in st.session_state:
    st.session_state.history = []
if "live_history" not in st.session_state:
    st.session_state.live_history = []
if "results_table" not in st.session_state:
    st.session_state.results_table = []
if "spin_counter" not in st.session_state:
    st.session_state.spin_counter = 0
if "martingale_1_loss_streak" not in st.session_state:
    st.session_state.martingale_1_loss_streak = 0
if "last_suggestion_name" not in st.session_state:
    st.session_state.last_suggestion_name = None
if "last_suggestion_mises" not in st.session_state:
    st.session_state.last_suggestion_mises = {}

# -------------------------------
# Données roue Crazy Time
# -------------------------------
segments_counts = {
    "1": 21,
    "2": 13,
    "5": 7,
    "10": 4,
    "Coin Flip": 4,
    "Cash Hunt": 2,
    "Pachinko": 2,
    "Crazy Time": 1
}

payouts = {
    "1": 1,
    "2": 2,
    "5": 5,
    "10": 10,
    "Coin Flip": 5,
    "Cash Hunt": 10,
    "Pachinko": 10,
    "Crazy Time": 20
}

# -------------------------------
# Stratégies fixes
# -------------------------------
def strategie_god_mode(scale=1):
    return {"1": 0, "2": 3*scale, "5": 2*scale, "10": 1*scale,
            "Coin Flip": 0, "Cash Hunt": 0, "Pachinko": 0, "Crazy Time": 0}

def strategie_god_mode_bonus(scale=1):
    return {"1": 0, "2": 3*scale, "5": 2*scale, "10": 1*scale,
            "Coin Flip": 1*scale, "Cash Hunt": 1*scale, "Pachinko": 1*scale, "Crazy Time": 1*scale}

def strategie_1_bonus(scale=1):
    return {"1": 4*scale, "2": 0, "5": 0, "10": 0,
            "Coin Flip": 1*scale, "Cash Hunt": 1*scale, "Pachinko": 1*scale, "Crazy Time": 1*scale}

# -------------------------------
# Gestion stratégie intelligente
# -------------------------------
def choose_strategy_intelligent(history, bankroll):
    total_segments = sum(segments_counts.values())
    probabilities = {k: v/total_segments for k, v in segments_counts.items()}

    # Bonus absents depuis longtemps → favorisés
    counts = {k: 0 for k in segments_counts}
    for spin in history[::-1]:
        counts[spin] += 1
        if spin in ["Coin Flip", "Cash Hunt", "Pachinko", "Crazy Time"]:
            break

    favor_bonus = max(counts, key=lambda k: counts[k] if k in ["Coin Flip", "Cash Hunt", "Pachinko", "Crazy Time"] else -1)

    strat = random.choice([strategie_god_mode, strategie_god_mode_bonus, strategie_1_bonus])
    mises = strat()

    # Ajustement bankroll dynamique
    scale = bankroll / st.session_state.initial_bankroll
    mises = {k: round(v*scale, 2) for k, v in mises.items()}

    return strat.__name__, mises

# -------------------------------
# Process Spin (calcul des gains)
# -------------------------------
def process_spin_real(spin_val, mises, bankroll, mult):
    mise_total = sum(mises.values())
    gain_brut = 0

    if spin_val in mises and mises[spin_val] > 0:
        gain_brut = mises[spin_val] * payouts[spin_val] * mult

    gain_net = gain_brut - mise_total
    new_bankroll = bankroll + gain_net

    return gain_net, gain_brut, mise_total, new_bankroll

# -------------------------------
# Interface Utilisateur
# -------------------------------
st.title("🎰 Super Crazy Time Bot")

# Multiplicateur manuel
mult_input = st.number_input("Multiplicateur (défaut = 1x)", min_value=1, step=1, value=1)

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Saisie Live Spin")
    spin_val = st.selectbox("Résultat du spin :", list(segments_counts.keys()))
    if st.button("Enregistrer live spin"):
        st.session_state.spin_counter += 1

        # Dernière suggestion
        mises_for_spin = st.session_state.last_suggestion_mises or {}

        gain_net, gain_brut, mise_total, new_bankroll = process_spin_real(spin_val, mises_for_spin, st.session_state.bankroll, mult_input)

        st.session_state.bankroll = new_bankroll
        st.session_state.history.append(spin_val)
        st.session_state.live_history.append((spin_val, mult_input))

        st.session_state.results_table.append({
            "Spin #": st.session_state.spin_counter,
            "Résultat": spin_val,
            "Multiplicateur": mult_input,
            "Mises $": {k: round(v, 2) for k, v in mises_for_spin.items()},
            "Mise Totale": round(mise_total, 2),
            "Gain Brut": round(gain_brut, 2),
            "Gain Net": round(gain_net, 2),
            "Bankroll": round(new_bankroll, 2)
        })

        # Martingale 1
        if gain_net > 0:
            st.session_state.martingale_1_loss_streak = 0
        else:
            st.session_state.martingale_1_loss_streak += 1

        # Nouvelle suggestion
        next_name, next_mises = choose_strategy_intelligent(st.session_state.history, st.session_state.bankroll)
        st.session_state.last_suggestion_name = next_name
        st.session_state.last_suggestion_mises = next_mises

        st.success(f"Spin {st.session_state.spin_counter} enregistré : {spin_val} x{mult_input} — Gain net {round(gain_net,2)} — Bankroll {round(new_bankroll,2)}")

with col2:
    st.subheader("📌 Stratégie suggérée")
    if st.session_state.last_suggestion_name:
        st.write(f"**Stratégie** : {st.session_state.last_suggestion_name}")
        st.json(st.session_state.last_suggestion_mises)
    else:
        st.write("Aucune stratégie encore générée.")

# -------------------------------
# Tableau et Graphique Bankroll
# -------------------------------
st.subheader("📊 Historique des Spins Live")
if st.session_state.results_table:
    df_results = pd.DataFrame(st.session_state.results_table)
    st.dataframe(df_results, use_container_width=True)

    st.subheader("📈 Évolution de la Bankroll")
    fig, ax = plt.subplots()
    ax.plot(df_results["Spin #"], df_results["Bankroll"], marker='o', label="Bankroll")
    ax.axhline(y=st.session_state.initial_bankroll, color="gray", linestyle="--", label="Bankroll initiale")
    ax.set_xlabel("Spin #")
    ax.set_ylabel("Bankroll ($)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
else:
    st.write("Aucun spin enregistré.")
