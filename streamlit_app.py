import streamlit as st
import pandas as pd

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Crazy Time Bot", layout="centered")

# -----------------------------
# Variables de session
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "bankroll" not in st.session_state:
    st.session_state.bankroll = 100.0
if "last_strategy" not in st.session_state:
    st.session_state.last_strategy = None
if "skip_bonus" not in st.session_state:
    st.session_state.skip_bonus = None
if "martingale_loss" not in st.session_state:
    st.session_state.martingale_loss = 0

# -----------------------------
# Fonction unité minimale
# -----------------------------
def get_unit(bankroll):
    base = max(0.2, bankroll * 0.002)
    if base < 0.4: return 0.2
    elif base < 1: return 0.4
    elif base < 2: return 1
    elif base < 4: return 2
    elif base < 10: return 4
    else: return 10

# -----------------------------
# Probabilités
# -----------------------------
def compute_probabilities(history):
    counts = {}
    total = len(history)
    for spin in history:
        seg = spin["result"].split("x")[0].strip()
        counts[seg] = counts.get(seg, 0) + 1
    probs = {seg: counts.get(seg, 0)/total for seg in ["1","2","5","10","Coin Flip","Cash Hunt","Pachinko","Crazy Time"]}
    return probs

# -----------------------------
# Stratégies
# -----------------------------
def martingale_one(bankroll):
    unit = get_unit(bankroll)
    mise = unit * (2 ** st.session_state.martingale_loss)
    return {"1": round(mise,2)}

def god_mode(bankroll, probs=None):
    unit = get_unit(bankroll)
    return {"2": 2*unit, "5": 1*unit, "10": 1*unit}

def god_mode_bonus(bankroll, probs=None):
    unit = get_unit(bankroll)
    # Pondération relative : 2>5&10>bonus
    total_units = unit * 2 + unit*1 + unit*1 + 4*unit*0.2
    strat = {}
    strat["2"] = 0.8*unit
    strat["5"] = 0.4*unit
    strat["10"] = 0.4*unit
    for bonus in ["Coin Flip","Cash Hunt","Pachinko","Crazy Time"]:
        if st.session_state.skip_bonus == bonus: continue
        strat[bonus] = 0.2*unit
    return strat

def one_plus_bonus(bankroll, probs=None):
    unit = get_unit(bankroll)
    strat = {}
    # Calcul autres mises selon probabilité
    for seg in ["Coin Flip","Cash Hunt","Pachinko","Crazy Time"]:
        if st.session_state.skip_bonus == seg: continue
        strat[seg] = 0.5*unit
    # Mise sur 1 = 1/2 total des autres mises
    total_other = sum(strat.values())
    strat["1"] = round(total_other/2,2)
    return strat

# -----------------------------
# Choisir stratégie
# -----------------------------
def expected_profit(strategy, probs):
    profit = 0
    for seg, mise in strategy.items():
        mult = 1
        if seg == "1": mult = 2
        elif seg == "2": mult = 3
        elif seg == "5": mult = 6
        elif seg == "10": mult = 11
        # bonus par défaut = 1
        profit += mise*mult*probs.get(seg,0)
    return profit

def choose_strategy(history, bankroll):
    # Martingale prioritaire
    if st.session_state.martingale_loss > 0:
        return martingale_one(bankroll)
    
    probs = compute_probabilities(history)
    strategies = [
        god_mode(bankroll, probs),
        god_mode_bonus(bankroll, probs),
        one_plus_bonus(bankroll, probs)
    ]
    # Choisir stratégie avec profit attendu max
    best = None
    best_profit = -1
    for strat in strategies:
        profit = expected_profit(strat, probs)
        if profit > best_profit:
            best_profit = profit
            best = strat
    return best

# -----------------------------
# Calcul gain
# -----------------------------
def calculate_gain(spin, strategy):
    gain = 0
    parts = spin.split("x")
    seg = parts[0].strip()
    mult = int(parts[1]) if len(parts)>1 else 1
    for bet_seg, mise in strategy.items():
        if bet_seg == seg:
            if seg=="1": gain += mise*2
            elif seg=="2": gain += mise*3
            elif seg=="5": gain += mise*6
            elif seg=="10": gain += mise*11
            else: gain += mise*(mult+1)
    return gain

# -----------------------------
# Affichage
# -----------------------------
def display_suggestion(strategy):
    st.subheader("💡 Suggestion Prochain Spin")
    if not strategy:
        st.info("Aucune stratégie pour l'instant.")
        return
    for seg,mise in strategy.items():
        st.write(f"- {seg} → {round(mise,2)} $")

# -----------------------------
# Interface
# -----------------------------
st.title("🎡 Crazy Time Bot")
st.sidebar.header("⚙️ Paramètres")
st.session_state.bankroll = st.sidebar.number_input(
    "Bankroll initiale ($)", min_value=50.0, max_value=1000.0,
    value=float(st.session_state.bankroll), step=1.0
)

st.subheader("📜 Historique des Spins")
new_spin = st.text_input("Entrer un spin historique (ex: 1, 5, Coin Flip, Pachinko x20) :")
if st.button("Ajouter à l'historique"):
    if new_spin:
        st.session_state.history.append({"result": new_spin, "gain":0})
        st.success(f"Ajouté : {new_spin}")

if st.button("Historique terminé"):
    st.session_state.last_strategy = choose_strategy(st.session_state.history, st.session_state.bankroll)

st.write(pd.DataFrame(st.session_state.history))

st.subheader("🎯 Spin Live")
live_spin = st.text_input("Résultat Spin Live (ex: 5, Pachinko x20, Coin Flip x3) :")
if st.button("Résultat Spin Live"):
    if live_spin:
        gain = calculate_gain(live_spin, st.session_state.last_strategy)
        total_bet = sum(st.session_state.last_strategy.values())
        net = gain - total_bet
        st.session_state.bankroll += net

        # Gestion martingale
        if "1" in st.session_state.last_strategy:
            if "1" in live_spin: st.session_state.martingale_loss = 0
            else: st.session_state.martingale_loss += 1

        # Historique
        st.session_state.history.append({"result": live_spin, "gain":gain})
        st.success(f"Résultat {live_spin} → Gain {round(gain,2)} | Bankroll: {round(st.session_state.bankroll,2)}")

        # Nouvelle stratégie
        st.session_state.last_strategy = choose_strategy(st.session_state.history, st.session_state.bankroll)

display_suggestion(st.session_state.last_strategy)
