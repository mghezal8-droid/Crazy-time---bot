import streamlit as st
import pandas as pd

# --- Probabilités officielles Crazy Time ---
probabilities = {
    "1": 21, "2": 13, "5": 7, "10": 4,
    "CoinFlip": 4, "CashHunt": 2, "Pachinko": 2, "CrazyTime": 1
}
total_segments = sum(probabilities.values())

# --- Session State init ---
if "history" not in st.session_state:
    st.session_state.history = []
if "bankroll" not in st.session_state:
    st.session_state.bankroll = 150  # bankroll de départ
if "strategy" not in st.session_state:
    st.session_state.strategy = None
if "units" not in st.session_state:
    st.session_state.units = {}
if "last_bonus" not in st.session_state:
    st.session_state.last_bonus = None
if "live_mode" not in st.session_state:
    st.session_state.live_mode = False  # False = saisie historique, True = live

# --- Stratégies ---
def martingale_on_1(bankroll):
    sequence = [0.2, 0.4, 1, 2, 4, 10]
    step = min(len(st.session_state.history), len(sequence)-1)
    return {"1": sequence[step]}

def god_mode_2_5_10():
    return {"2": 2, "5": 1, "10": 1}

def god_mode_2_5_10_bonus(last_bonus=None):
    bets = god_mode_2_5_10()
    for b in ["CoinFlip", "CashHunt", "Pachinko", "CrazyTime"]:
        if b != last_bonus:
            bets[b] = 1
    return bets

def one_plus_bonus(last_bonus=None):
    bets = {"1": 1}
    for b in ["CoinFlip", "CashHunt", "Pachinko", "CrazyTime"]:
        if b != last_bonus:
            bets[b] = 1
    return bets

# --- Choix automatique stratégie ---
def choose_strategy(history):
    counts = {k: history.count(k) for k in probabilities}
    total_spins = len(history)
    if total_spins < 10:
        return "Attente", {}

    expected = {k: probabilities[k]/total_segments*total_spins for k in probabilities}
    diffs = {k: expected[k] - counts.get(k, 0) for k in probabilities}

    if diffs["1"] > 3:
        return "Martingale 1", martingale_on_1(st.session_state.bankroll)
    if diffs["2"] + diffs["5"] + diffs["10"] > 3:
        return "God Mode 2,5,10", god_mode_2_5_10()
    if diffs["2"] + diffs["5"] + diffs["10"] > 2 and (
        diffs["CoinFlip"] + diffs["CashHunt"] + diffs["Pachinko"] + diffs["CrazyTime"]) > 2:
        return "God Mode 2,5,10 + Bonus", god_mode_2_5_10_bonus(st.session_state.last_bonus)
    return "1 + Bonus Combo", one_plus_bonus(st.session_state.last_bonus)

# --- Interface ---
st.title("🎡 Crazy Time Bot")

st.write(f"💰 **Bankroll actuelle : {st.session_state.bankroll}$**")

col1, col2 = st.columns(2)
with col1:
    result = st.text_input("Résultat du spin (1,2,5,10,CoinFlip,CashHunt,Pachinko,CrazyTime)")
with col2:
    multiplier = st.number_input("Multiplicateur (ex: 1, 2, 10...)", min_value=1, step=1, value=1)

# --- Phase 1 : Entrée de l'historique ---
if not st.session_state.live_mode:
    if st.button("Ajouter à l'historique"):
        if result:
            st.session_state.history.append(result)
            if result in ["CoinFlip", "CashHunt", "Pachinko", "CrazyTime"]:
                st.session_state.last_bonus = result

    st.subheader("📜 Historique des spins saisis")
    if len(st.session_state.history) > 0:
        st.dataframe(pd.DataFrame(st.session_state.history, columns=["Spin"]))
    else:
        st.info("Ajoute des spins pour construire l'historique.")

    if st.button("✅ Historique terminé, commencer à miser"):
        st.session_state.live_mode = True
        st.success("Mode Live activé : le bot commence à suggérer et calculer.")

# --- Phase 2 : Mode Live ---
else:
    if st.button("Spin Live"):
        if result:
            st.session_state.history.append(result)
            if result in ["CoinFlip", "CashHunt", "Pachinko", "CrazyTime"]:
                st.session_state.last_bonus = result

            # Choix stratégie
            strat, units = choose_strategy(st.session_state.history)
            st.session_state.strategy = strat
            st.session_state.units = units

            # Calcul résultat
            gain = 0
            mise_totale = sum(units.values())
            if result in units:
                if result in ["1", "2", "5", "10"]:
                    payout = {"1": 2, "2": 3, "5": 6, "10": 11}[result]
                    gain = (multiplier * payout * units[result]) + units[result]
                else:
                    gain = (multiplier * units[result]) + units[result]
            net = gain - mise_totale
            st.session_state.bankroll += net

            # --- Affichage ---
            st.subheader("📊 Résultat du spin")
            st.write(f"🎯 Spin : **{result} (x{multiplier})**")
            st.write(f"🎯 Stratégie choisie : **{st.session_state.strategy}**")
            st.table(pd.DataFrame(list(units.items()), columns=["Segment", "Mise (unités)"]))
            st.write(f"💸 Total misé : {mise_totale}$")
            if gain > 0:
                st.success(f"✅ Gain : {gain}$  | Net : +{net}$")
            else:
                st.error(f"❌ Perte : {abs(net)}$")
            st.write(f"💰 Bankroll : {st.session_state.bankroll}$")

    st.subheader("📜 Historique complet")
    st.dataframe(pd.DataFrame(st.session_state.history, columns=["Spin"]))
