import streamlit as st
import pandas as pd

# --- Configuration ---
SPINS = ["1", "2", "5", "10", "CoinFlip", "CashHunt", "Pachinko", "CrazyTime"]
PAYOUTS = {"1": 2, "2": 3, "5": 6, "10": 11, "CoinFlip": 2, "CashHunt": 3, "Pachinko": 4, "CrazyTime": 5}
WHEEL = {"1":21, "2":13, "5":7, "10":4, "CoinFlip":4, "CashHunt":2, "Pachinko":2, "CrazyTime":1}
TOTAL_SEGMENTS = sum(WHEEL.values())

# --- Session State ---
if "history" not in st.session_state:
    st.session_state.history = []
if "bankroll" not in st.session_state:
    st.session_state.bankroll = 150
if "last_bonus" not in st.session_state:
    st.session_state.last_bonus = None
if "live_mode" not in st.session_state:
    st.session_state.live_mode = False
if "strategy" not in st.session_state:
    st.session_state.strategy = None
if "units" not in st.session_state:
    st.session_state.units = {}

# --- Stratégies ---
def martingale_on_1():
    sequence = [0.2, 0.4, 1, 2, 4, 10]
    step = min(len([s for s in st.session_state.history if s=="1"]), len(sequence)-1)
    return {"1": sequence[step]}

def god_mode_2_5_10():
    return {"2":2, "5":1, "10":1}

def god_mode_2_5_10_bonus():
    bets = god_mode_2_5_10()
    for b in ["CoinFlip","CashHunt","Pachinko","CrazyTime"]:
        if b != st.session_state.last_bonus:
            bets[b] = 1
    return bets

def one_plus_bonus():
    bets = {"1":1}
    for b in ["2","5","10","CoinFlip","CashHunt","Pachinko","CrazyTime"]:
        if b != st.session_state.last_bonus:
            bets[b] = 1
    return bets

# --- Choix stratégie ---
def choose_strategy():
    hist = st.session_state.history
    total = len(hist)
    if total < 10:
        return "Attente", {}

    counts = {s: hist.count(s) for s in SPINS}
    expected = {s: WHEEL[s]/TOTAL_SEGMENTS*total for s in SPINS}
    diffs = {s: expected[s]-counts.get(s,0) for s in SPINS}

    if diffs["1"] > 3:
        return "Martingale 1", martingale_on_1()
    if diffs["2"]+diffs["5"]+diffs["10"] > 3:
        return "God Mode 2,5,10", god_mode_2_5_10()
    if diffs["2"]+diffs["5"]+diffs["10"] > 2 and sum([diffs[b] for b in ["CoinFlip","CashHunt","Pachinko","CrazyTime"]]) > 2:
        return "God Mode 2,5,10 + Bonus", god_mode_2_5_10_bonus()
    return "1 + Bonus Combo", one_plus_bonus()

# --- Interface ---
st.title("🎡 Crazy Time Bot (version corrigée)")

col1, col2 = st.columns(2)
with col1:
    spin_input = st.text_input("Spin (1,2,5,10,CoinFlip,CashHunt,Pachinko,CrazyTime)")
with col2:
    multiplier = st.number_input("Multiplicateur", min_value=1, step=1, value=1)

# --- Phase 1 : Historique ---
if not st.session_state.live_mode:
    if st.button("Ajouter à l'historique"):
        if spin_input:
            st.session_state.history.append(spin_input)
            if spin_input in ["CoinFlip","CashHunt","Pachinko","CrazyTime"]:
                st.session_state.last_bonus = spin_input

    st.subheader("📜 Historique saisi")
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history, columns=["Spin"]))
    else:
        st.info("Ajoute les spins pour construire l'historique.")

    if st.button("✅ Historique terminé - Commencer live"):
        st.session_state.live_mode = True
        # Première stratégie directement après l’historique
        strat, units = choose_strategy()
        st.session_state.strategy = strat
        st.session_state.units = units
        st.success("Mode live activé. Première stratégie générée.")

# --- Phase 2 : Live ---
else:
    if st.button("Spin Live"):
        if spin_input:
            # Ajouter spin
            st.session_state.history.append(spin_input)
            if spin_input in ["CoinFlip","CashHunt","Pachinko","CrazyTime"]:
                st.session_state.last_bonus = spin_input

            # Gain calculé selon stratégie précédente
            units = st.session_state.units
            strat = st.session_state.strategy
            total_bet = sum(units.values())
            gain = 0
            if spin_input in units:
                if spin_input in ["1","2","5","10"]:
                    gain = (multiplier*PAYOUTS[spin_input]*units[spin_input]) + units[spin_input]
                else:
                    gain = (multiplier*units[spin_input]) + units[spin_input]
            net = gain - total_bet
            st.session_state.bankroll += net

            # Résultat du spin
            st.subheader("📊 Résultat Spin Live")
            st.write(f"🎯 Spin : **{spin_input} (x{multiplier})**")
            st.write(f"🎯 Stratégie utilisée : **{strat}**")
            st.table(pd.DataFrame(list(units.items()), columns=["Segment", "Mise (unités)"]))
            st.write(f"💸 Total misé : {total_bet}$")
            if gain > 0:
                st.success(f"✅ Gain : {gain}$ | Net : +{net}$")
            else:
                st.error(f"❌ Perte : {abs(net)}$")
            st.write(f"💰 Bankroll : {st.session_state.bankroll}$")

            # Nouvelle stratégie pour le prochain spin
            strat, units = choose_strategy()
            st.session_state.strategy = strat
            st.session_state.units = units

    # Toujours afficher la suggestion du prochain spin
    st.subheader("💡 Suggestion prochain spin")
    if st.session_state.strategy and st.session_state.units:
        st.write(f"Stratégie : **{st.session_state.strategy}**")
        st.table(pd.DataFrame(list(st.session_state.units.items()), columns=["Segment", "Mise (unités)"]))

    st.subheader("📜 Historique complet")
    st.dataframe(pd.DataFrame(st.session_state.history, columns=["Spin"]))
