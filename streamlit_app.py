import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt

# -------------------------------
# INITIALISATION
# -------------------------------
if "bankroll" not in st.session_state:
    st.session_state.bankroll = 100.0
if "initial_bankroll" not in st.session_state:
    st.session_state.initial_bankroll = 100.0
if "live_history" not in st.session_state:
    st.session_state.live_history = []
if "results_table" not in st.session_state:
    st.session_state.results_table = []
if "history" not in st.session_state:
    st.session_state.history = []
if "martingale_1_loss_streak" not in st.session_state:
    st.session_state.martingale_1_loss_streak = 0
if "last_suggestion_name" not in st.session_state:
    st.session_state.last_suggestion_name = None
if "last_suggestion_mises" not in st.session_state:
    st.session_state.last_suggestion_mises = None

# -------------------------------
# SEGMENTS DE LA ROUE
# -------------------------------
segments = {
    '1': 1,
    '2': 2,
    '5': 5,
    '10': 10,
    'Coin Flip': 10,
    'Cash Hunt': 10,
    'Pachinko': 10,
    'Crazy Time': 10
}

# -------------------------------
# STRATÉGIES
# -------------------------------
def strategie_god_mode(bankroll):
    base = bankroll * 0.02
    return {
        '1': base,
        '2': base,
        '5': base,
        '10': base,
        'Coin Flip': base * 0.8,
        'Cash Hunt': base * 0.8,
        'Pachinko': base * 0.8,
        'Crazy Time': base * 0.8
    }

def strategie_god_mode_bonus(bankroll):
    base = bankroll * 0.015
    return {
        '1': base,
        '2': base,
        '5': base,
        '10': base,
        'Coin Flip': base * 1.5,
        'Cash Hunt': base * 1.5,
        'Pachinko': base * 1.5,
        'Crazy Time': base * 1.5
    }

def strategie_1_bonus(bankroll):
    base = bankroll * 0.025
    return {
        '1': base,
        '2': 0,
        '5': 0,
        '10': 0,
        'Coin Flip': base * 0.8,
        'Cash Hunt': base * 0.8,
        'Pachinko': base * 0.8,
        'Crazy Time': base * 0.8
    }

# -------------------------------
# CHOIX DE STRATÉGIE INTELLIGENT
# -------------------------------
def choose_strategy_intelligent(history, bankroll):
    strat_list = {
        "God Mode": strategie_god_mode,
        "God Mode + Bonus": strategie_god_mode_bonus,
        "1 + Bonus": strategie_1_bonus
    }
    strat_name = st.session_state.last_suggestion_name or random.choice(list(strat_list.keys()))
    strat_func = strat_list[strat_name]
    return strat_name, strat_func(bankroll)

# -------------------------------
# MARTINGALE 1 : double après perte
# -------------------------------
def appliquer_martingale_1(mises, perte_streak):
    facteur = 2 ** perte_streak
    return {k: round(v * facteur, 2) for k, v in mises.items()}

# -------------------------------
# CALCUL DU SPIN
# -------------------------------
def process_spin_real(spin_val, mult, mises, bankroll):
    spin_value = segments.get(spin_val, 0)
    mult_applique = mult if mult > 0 else 1

    mise_total = sum(mises.values())
    if spin_val in mises:
        gain_brut = (mises[spin_val] * spin_value * mult_applique) + mises[spin_val]
    else:
        gain_brut = 0.0

    gain_net = gain_brut - mise_total
    new_bankroll = bankroll + gain_net

    return gain_net, gain_brut, mise_total, new_bankroll, mult_applique

# -------------------------------
# INTERFACE PRINCIPALE
# -------------------------------
st.title("🎯 Crazy Time — Bot Stratégique (Version Complète)")

st.metric("💰 Bankroll actuelle", f"{st.session_state.bankroll:.2f} $")

mult_input = st.number_input("🎲 Multiplicateur manuel (x)", min_value=1.0, value=1.0, step=0.5)

# SUGGESTION DE STRATÉGIE
if st.button("📈 Obtenir la prochaine suggestion"):
    next_name, next_mises = choose_strategy_intelligent(st.session_state.history, st.session_state.bankroll)
    next_mises = appliquer_martingale_1(next_mises, st.session_state.martingale_1_loss_streak)
    st.session_state.last_suggestion_name = next_name
    st.session_state.last_suggestion_mises = next_mises

    st.success(f"Stratégie active : {next_name}")
    st.write(pd.DataFrame.from_dict(next_mises, orient='index', columns=['Mise $']))

# ENREGISTRER UN SPIN
st.subheader("🎡 Enregistrer un Spin Live")
col1, col2 = st.columns(2)
with col1:
    spin_val = st.selectbox("Résultat du spin :", list(segments.keys()))
with col2:
    mult_val = st.number_input("Multiplicateur Top Slot (x)", min_value=1.0, value=mult_input)

if st.button("✅ Enregistrer le spin live"):
    if st.session_state.last_suggestion_mises:
        gain_net, gain_brut, mise_total, new_bankroll, mult_applique = process_spin_real(
            spin_val, mult_val, st.session_state.last_suggestion_mises, st.session_state.bankroll
        )

        st.session_state.bankroll = new_bankroll
        st.session_state.history.append((spin_val, mult_applique, gain_net))
        st.session_state.results_table.append({
            "Spin #": len(st.session_state.results_table) + 1,
            "Résultat": spin_val,
            "Multiplicateur": mult_applique,
            "Mises $": {k: round(v, 2) for k, v in st.session_state.last_suggestion_mises.items()},
            "Mise Totale": round(mise_total, 2),
            "Gain Brut": round(gain_brut, 2),
            "Gain Net": round(gain_net, 2),
            "Bankroll": round(new_bankroll, 2)
        })

        if gain_net > 0:
            st.session_state.martingale_1_loss_streak = 0
        else:
            st.session_state.martingale_1_loss_streak += 1

        st.success(f"Spin enregistré ✅ — {spin_val} x{mult_applique} — Gain net: {gain_net:.2f} — Bankroll: {new_bankroll:.2f}")
    else:
        st.warning("⚠️ Clique sur 'Obtenir la prochaine suggestion' avant d’enregistrer un spin.")

# SUPPRIMER DERNIER SPIN
if st.button("❌ Supprimer dernier spin"):
    if st.session_state.results_table:
        st.session_state.results_table.pop()
        if st.session_state.history:
            st.session_state.history.pop()
        if st.session_state.martingale_1_loss_streak > 0:
            st.session_state.martingale_1_loss_streak -= 1
        st.warning("Dernier spin supprimé.")

# -------------------------------
# SECTION HISTORIQUE DES SPINS
# -------------------------------
st.subheader("📜 Historique complet des spins enregistrés")

if st.session_state.results_table:
    df_results = pd.DataFrame(st.session_state.results_table)
    st.dataframe(df_results, use_container_width=True)

    # Graphique bankroll
    st.subheader("📊 Évolution de la Bankroll")
    fig, ax = plt.subplots()
    ax.plot(df_results["Spin #"], df_results["Bankroll"], marker='o', label='Bankroll')
    ax.axhline(y=st.session_state.initial_bankroll, color='gray', linestyle='--', label='Bankroll initiale')
    ax.set_xlabel("Spin #")
    ax.set_ylabel("Bankroll ($)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # Boutons pour gérer l’historique
    if st.button("💾 Exporter l’historique en CSV"):
        df_results.to_csv("historique_spins.csv", index=False)
        st.success("✅ Historique exporté sous 'historique_spins.csv'")

    if st.button("🧹 Réinitialiser l’historique"):
        st.session_state.results_table = []
        st.session_state.history = []
        st.session_state.martingale_1_loss_streak = 0
        st.warning("Historique réinitialisé.")
else:
    st.info("Aucun spin enregistré pour le moment.")
