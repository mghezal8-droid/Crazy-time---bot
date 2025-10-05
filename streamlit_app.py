import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random

# -----------------------------------
# 🔧 CONFIG INITIALE
# -----------------------------------
st.set_page_config(page_title="🎰 Crazy Time Tracker", layout="wide")

# Valeurs des segments (sans multiplicateur)
VAL_SEG = {'1': 1, '2': 2, '5': 5, '10': 10}

# -----------------------------------
# ⚙️ INIT DES VARIABLES
# -----------------------------------
if "bankroll" not in st.session_state:
    st.session_state.bankroll = 150.0
if "initial_bankroll" not in st.session_state:
    st.session_state.initial_bankroll = 150.0
if "live_history" not in st.session_state:
    st.session_state.live_history = []
if "history" not in st.session_state:
    st.session_state.history = []  # historique manuel avant live
if "results_table" not in st.session_state:
    st.session_state.results_table = []
if "martingale_1_loss_streak" not in st.session_state:
    st.session_state.martingale_1_loss_streak = 0
if "last_suggestion_name" not in st.session_state:
    st.session_state.last_suggestion_name = None
if "last_suggestion_mises" not in st.session_state:
    st.session_state.last_suggestion_mises = {}

# -----------------------------------
# 🎯 DÉFINITION DES STRATÉGIES
# -----------------------------------
def strategy_martingale_1(bankroll, loss_streak):
    base_bet = 4.0
    mise_1 = base_bet * (2 ** loss_streak)
    return "Martingale 1", {'1': mise_1}

def strategy_god_mode(bankroll):
    return "God Mode", {'2': 3.0, '5': 2.0, '10': 1.0}

def strategy_god_mode_bonus(bankroll):
    return "God Mode + Bonus", {'2': 3.0, '5': 2.0, '10': 1.0,
                                'Coin Flip': 1.0, 'Cash Hunt': 1.0,
                                'Pachinko': 1.0, 'Crazy Time': 1.0}

def strategy_1_bonus(bankroll):
    return "1 + Bonus", {'1': 4.0, 'Coin Flip': 1.0,
                         'Cash Hunt': 1.0, 'Pachinko': 1.0,
                         'Crazy Time': 1.0}

STRATEGIES = [strategy_martingale_1, strategy_god_mode, strategy_god_mode_bonus, strategy_1_bonus]

# -----------------------------------
# 🧠 CHOIX INTELLIGENT
# -----------------------------------
def choose_strategy_intelligent(history, bankroll):
    if not history:
        return strategy_martingale_1(bankroll, 0)
    if st.session_state.martingale_1_loss_streak > 0:
        return strategy_martingale_1(bankroll, st.session_state.martingale_1_loss_streak)
    else:
        return random.choice([strategy_god_mode, strategy_god_mode_bonus, strategy_1_bonus])(bankroll)

# -----------------------------------
# 💰 CALCUL DU GAIN (formule exacte)
# -----------------------------------
def calcul_gain(mises, spin_result, multiplicateur):
    gain_brut = 0
    for segment, mise in mises.items():
        if segment == spin_result:
            if segment in VAL_SEG:
                gain_brut += mise * (VAL_SEG[segment] * multiplicateur) + mise
            else:
                gain_brut += mise * multiplicateur + mise
    mise_totale = sum(mises.values())
    gain_net = gain_brut - mise_totale
    return gain_brut, gain_net

# -----------------------------------
# 🧾 AFFICHAGE DE LA PROCHAINE STRATÉGIE
# -----------------------------------
def display_next_suggestion():
    st.subheader("🎯 Prochaine stratégie suggérée")
    st.write(f"**Stratégie :** {st.session_state.last_suggestion_name}")
    st.table(pd.DataFrame.from_dict(st.session_state.last_suggestion_mises, orient='index', columns=['Mise $']))

# -----------------------------------
# 📝 HISTORIQUE MANUEL AVANT SPINS LIVE
# -----------------------------------
st.header("📝 Historique Manuel (avant spins live)")
def segment_buttons_grid(segments, cols_per_row=4):
    rows = (len(segments)+cols_per_row-1)//cols_per_row
    idx = 0
    for r in range(rows):
        cols = st.columns(cols_per_row)
        for c in range(cols_per_row):
            if idx >= len(segments): break
            seg = segments[idx]
            if cols[c].button(seg, key=f"hist_{seg}_{idx}"):
                st.session_state.history.append(seg)
            idx += 1
segment_buttons_grid(['1','2','5','10','Coin Flip','Cash Hunt','Pachinko','Crazy Time'])

col_a, col_b = st.columns(2)
with col_a:
    if st.button("↩ Supprimer dernier spin historique"):
        if st.session_state.history:
            st.session_state.history.pop()
            st.success("Dernier spin historique supprimé.")
with col_b:
    if st.button("🏁 Terminer historique"):
        st.success(f"Historique terminé ({len(st.session_state.history)} spins).")
        # calcul de la première suggestion
        next_name, next_mises = choose_strategy_intelligent(st.session_state.history, st.session_state.bankroll)
        st.session_state.last_suggestion_name = next_name
        st.session_state.last_suggestion_mises = next_mises

# Affichage de l'historique
if st.session_state.history:
    st.subheader("📋 Historique manuel actuel")
    df_manual = pd.DataFrame({"#": range(1,len(st.session_state.history)+1),
                              "Résultat": st.session_state.history})
    st.dataframe(df_manual,use_container_width=True)

# -----------------------------------
# 🧮 ENTRÉES UTILISATEUR - SPINS LIVE
# -----------------------------------
st.title("🎡 Crazy Time Live Tracker")

col1, col2 = st.columns(2)
with col1:
    spin_val = st.selectbox("🎯 Résultat du spin :", ['1','2','5','10','Coin Flip','Cash Hunt','Pachinko','Crazy Time'])
    mult_input = st.text_input("💥 Multiplicateur (ex: x25 ou 25) :", "1")
    multiplicateur = float(mult_input.lower().replace('x','')) if mult_input else 1

with col2:
    if st.button("🎰 Enregistrer le spin live"):
        strategy_name = st.session_state.last_suggestion_name
        mises_for_spin = st.session_state.last_suggestion_mises

        gain_brut, gain_net = calcul_gain(mises_for_spin, spin_val, multiplicateur)
        mise_total = sum(mises_for_spin.values())
        new_bankroll = st.session_state.bankroll + gain_net

        st.session_state.bankroll = new_bankroll
        st.session_state.live_history.append(spin_val)

        st.session_state.results_table.append({
            "Spin #": len(st.session_state.results_table) + 1,
            "Stratégie": strategy_name,
            "Résultat": spin_val,
            "Multiplicateur": multiplicateur,
            "Mises $": {k: round(v,2) for k,v in mises_for_spin.items()},
            "Mise Totale": round(mise_total,2),
            "Gain Brut": round(gain_brut,2),
            "Gain Net": round(gain_net,2),
            "Bankroll": round(new_bankroll,2)
        })

        # Martingale
        if strategy_name == "Martingale 1":
            if spin_val == '1' and gain_net > 0:
                st.session_state.martingale_1_loss_streak = 0
            else:
                st.session_state.martingale_1_loss_streak += 1

        # Prochaine suggestion
        next_name, next_mises = choose_strategy_intelligent(st.session_state.live_history, new_bankroll)
        st.session_state.last_suggestion_name = next_name
        st.session_state.last_suggestion_mises = next_mises
        display_next_suggestion()

        st.success(f"Spin enregistré : {spin_val} x{multiplicateur} — Gain net: {round(gain_net,2)} — Bankroll: {round(new_bankroll,2)}")

# -----------------------------------
# 🔁 SUPPRESSION DERNIER SPIN
# -----------------------------------
if st.button("🗑️ Supprimer dernier spin"):
    if st.session_state.results_table:
        st.session_state.results_table.pop()
        if st.session_state.live_history:
            st.session_state.live_history.pop()
        if st.session_state.martingale_1_loss_streak > 0:
            st.session_state.martingale_1_loss_streak -= 1
        st.warning("Dernier spin supprimé.")
        next_name, next_mises = choose_strategy_intelligent(st.session_state.live_history, st.session_state.bankroll)
        st.session_state.last_suggestion_name = next_name
        st.session_state.last_suggestion_mises = next_mises
        display_next_suggestion()

# -----------------------------------
# 📊 HISTORIQUE ET GRAPHIQUE
# -----------------------------------
st.subheader("📈 Historique des Spins Live")
if st.session_state.results_table:
    df_results = pd.DataFrame(st.session_state.results_table)
    st.dataframe(df_results, use_container_width=True)

    st.subheader("💹 Évolution de la Bankroll")
    fig, ax = plt.subplots()
    ax.plot(df_results["Spin #"], df_results["Bankroll"], marker='o', label='Bankroll')
    ax.axhline(y=st.session_state.initial_bankroll, color='gray', linestyle='--', label='Bankroll initiale')
    ax.set_xlabel("Spin #")
    ax.set_ylabel("Bankroll ($)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
else:
    st.write("Aucun spin encore enregistré.")
