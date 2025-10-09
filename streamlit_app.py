# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import Counter

# -----------------------------------
# 🔧 CONFIG INITIALE
# -----------------------------------
st.set_page_config(page_title="🎰 Crazy Time Tracker", layout="wide")

VAL_SEG = {'1': 1, '2': 2, '5': 5, '10': 10}
THEO_COUNTS = {
    '1': 21, '2': 13, '5': 7, '10': 4,
    'Coin Flip': 4, 'Cash Hunt': 2, 'Pachinko': 2, 'Crazy Time': 1
}
THEO_TOTAL = sum(THEO_COUNTS.values())

# -----------------------------------
# ⚙️ INIT session_state
# -----------------------------------
if "bankroll" not in st.session_state:
    st.session_state.bankroll = 150.0
if "initial_bankroll" not in st.session_state:
    st.session_state.initial_bankroll = 150.0
if "live_history" not in st.session_state:
    st.session_state.live_history = []
if "history" not in st.session_state:
    st.session_state.history = []
if "results_table" not in st.session_state:
    st.session_state.results_table = []
if "martingale_1_loss_streak" not in st.session_state:
    st.session_state.martingale_1_loss_streak = 0
if "miss_streak" not in st.session_state:
    st.session_state.miss_streak = 0
if "last_suggestion_name" not in st.session_state:
    st.session_state.last_suggestion_name = None
if "last_suggestion_mises" not in st.session_state:
    st.session_state.last_suggestion_mises = {}
if "bonus_multiplier_assumption" not in st.session_state:
    st.session_state.bonus_multiplier_assumption = 10

SEGMENTS = ['1','2','5','10','Coin Flip','Cash Hunt','Pachinko','Crazy Time']

# -----------------------------------
# 🎯 STRATÉGIES
# -----------------------------------
def strategy_martingale_1(bankroll, loss_streak):
    base_bet = 4.0
    mise_1 = base_bet * (2 ** loss_streak)
    return "Martingale 1", {'1': mise_1}

def strategy_god_mode(bankroll):
    return "God Mode", {'2': 3.0, '5': 2.0, '10': 1.0}

def strategy_god_mode_bonus(bankroll):
    return "God Mode + Bonus", {
        '2': 3.0, '5': 2.0, '10': 1.0,
        'Coin Flip': 1.0, 'Cash Hunt': 1.0,
        'Pachinko': 1.0, 'Crazy Time': 1.0
    }

def strategy_1_bonus(bankroll):
    return "1 + Bonus", {
        '1': 4.0,
        'Coin Flip': 1.0, 'Cash Hunt': 1.0,
        'Pachinko': 1.0, 'Crazy Time': 1.0
    }

def strategy_only_numbers(bankroll):
    return "Only Numbers", {'1': 3.0, '2': 2.0, '5': 1.0, '10': 1.0}

def strategy_no_bets():
    return "No Bets", {}

# -----------------------------------
# 🧠 UTILITAIRES PROBABILITÉS / EV
# -----------------------------------
def theo_prob(segment):
    return THEO_COUNTS.get(segment, 0) / THEO_TOTAL

def hist_prob(full_history, segment, window=300):  # augmenté à 300
    if not full_history:
        return 0.0
    hist = full_history[-window:]
    return hist.count(segment) / len(hist)

def combined_prob(full_history, segment, window=300):  # aussi 300 ici
    return 0.5 * (theo_prob(segment) + hist_prob(full_history, segment, window=window))

def expected_value_for_strategy(mises, full_history, multiplicateur, bankroll):
    mise_totale = sum(mises.values()) if mises else 0.0
    ev = 0.0
    for seg in SEGMENTS:
        p = combined_prob(full_history, seg)
        if seg in mises and mises[seg] > 0:
            seg_val = VAL_SEG.get(seg, st.session_state.bonus_multiplier_assumption)
            payout = mises[seg] * (seg_val * multiplicateur) + mises[seg]
            net_if_hit = payout - mise_totale
        else:
            net_if_hit = -mise_totale
        ev += p * net_if_hit
    return ev

# -----------------------------------
# 🧠 CHOIX INTELLIGENT + MARTINGALE
# -----------------------------------
def choose_strategy_intelligent(full_history, bankroll, multiplicateur):
    if st.session_state.martingale_1_loss_streak > 0:
        return strategy_martingale_1(bankroll, st.session_state.martingale_1_loss_streak)
    if st.session_state.miss_streak >= 3:
        return strategy_martingale_1(bankroll, 0)
    candidates = []
    for builder in [strategy_only_numbers, strategy_god_mode, strategy_god_mode_bonus, strategy_1_bonus]:
        name, mises = builder(bankroll)
        ev = expected_value_for_strategy(mises, full_history, multiplicateur, bankroll)
        candidates.append((name, mises, ev))
    best = max(candidates, key=lambda x: x[2])
    if best[2] <= 0:
        return strategy_no_bets()
    return best[0], best[1]

# -----------------------------------
# 💰 CALCUL GAIN
# -----------------------------------
def calcul_gain(mises, spin_result, multiplicateur):
    if not mises:
        return 0.0, 0.0
    mise_totale = sum(mises.values())
    gain_brut = 0.0
    if spin_result in mises:
        seg_val = VAL_SEG.get(spin_result, st.session_state.bonus_multiplier_assumption)
        gain_brut = (mises[spin_result] * (seg_val * multiplicateur)) + mises[spin_result]
    gain_net = gain_brut - mise_totale
    return float(gain_brut), float(gain_net)

# -----------------------------------
# 🧾 AFFICHAGE PROCHAINE STRATÉGIE
# -----------------------------------
def display_next_suggestion():
    st.subheader("🎯 Prochaine stratégie suggérée")
    if st.session_state.last_suggestion_name:
        st.write(f"**Stratégie :** {st.session_state.last_suggestion_name}")
        if st.session_state.last_suggestion_mises:
            st.table(pd.DataFrame.from_dict(st.session_state.last_suggestion_mises, orient='index', columns=['Mise $']))
        else:
            st.info("🚫 No Bets")
    else:
        st.write("Aucune stratégie suggérée pour l’instant.")

# -----------------------------------
# 📝 HISTORIQUE MANUEL
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

segment_buttons_grid(SEGMENTS)

col_a, col_b, col_c = st.columns([1,1,1])
with col_a:
    if st.button("↩ Supprimer dernier spin historique"):
        if st.session_state.history:
            st.session_state.history.pop()
with col_b:
    if st.button("🔄 Réinitialiser historique manuel"):
        st.session_state.history = []
with col_c:
    if st.button("🏁 Terminer historique"):
        full_history = st.session_state.history + st.session_state.live_history
        next_name, next_mises = choose_strategy_intelligent(full_history, st.session_state.bankroll, st.session_state["mult_for_ev"])
        st.session_state.last_suggestion_name = next_name
        st.session_state.last_suggestion_mises = next_mises
        display_next_suggestion()

if st.session_state.history:
    st.subheader("📋 Historique manuel actuel")
    df_manual = pd.DataFrame({"#": range(1,len(st.session_state.history)+1),"Résultat": st.session_state.history})
    st.dataframe(df_manual,use_container_width=True)

# -----------------------------------
# 🧠 SIMULATION DU BOT SUR L’HISTORIQUE
# -----------------------------------
if st.button("🧠 Appliquer le bot sur l’historique"):
    if not st.session_state.history:
        st.warning("⚠️ Ajoute d’abord des spins manuels avant d’appliquer le bot.")
    else:
        bankroll_sim = st.session_state.initial_bankroll
        simulated_results = []
        miss_streak_sim = 0
        martingale_streak_sim = 0

        for i, spin_result in enumerate(st.session_state.history, start=1):
            full_history_temp = st.session_state.history[:i-1]
            strategy_name, mises = choose_strategy_intelligent(full_history_temp, bankroll_sim, st.session_state["mult_for_ev"])
            gain_brut, gain_net = calcul_gain(mises, spin_result, 1)
            bankroll_sim += gain_net
            bet_segments = list(mises.keys())
            if spin_result not in bet_segments:
                miss_streak_sim += 1
            else:
                miss_streak_sim = 0
            if strategy_name == "Martingale 1":
                if gain_net > 0:
                    martingale_streak_sim = 0
                    miss_streak_sim = 0
                else:
                    martingale_streak_sim += 1
            simulated_results.append({
                "Spin #": i, "Résultat": spin_result,
                "Stratégie": strategy_name,
                "Gain Net": round(gain_net, 2),
                "Bankroll": round(bankroll_sim, 2)
            })

        df_sim = pd.DataFrame(simulated_results)
        st.subheader("📊 Résultats du bot sur l’historique")
        st.dataframe(df_sim, use_container_width=True)

        st.subheader("💹 Évolution simulée de la bankroll (historique)")
        fig, ax = plt.subplots()
        ax.plot(df_sim["Spin #"], df_sim["Bankroll"], marker='o')
        ax.axhline(y=st.session_state.initial_bankroll, color='gray', linestyle='--', label='Bankroll initiale')
        ax.set_xlabel("Spin #")
        ax.set_ylabel("Bankroll ($)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        st.success(f"✅ Simulation terminée — Bankroll finale : {round(bankroll_sim, 2)}$")
