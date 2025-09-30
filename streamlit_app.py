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
if "strategy_repeat_count" not in st.session_state:
    st.session_state.strategy_repeat_count = 0
if "last_strategy_name" not in st.session_state:
    st.session_state.last_strategy_name = None

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
    all_segments = ["1","2","5","10","Coin Flip","Cash Hunt","Pachinko","Crazy Time"]
    probs = {seg: counts.get(seg,0)/total if total>0 else 0 for seg in all_segments}
    return probs

# -----------------------------
# Stratégies
# -----------------------------
def martingale_one(bankroll):
    unit = get_unit(bankroll)
    mise = unit * (2 ** st.session_state.martingale_loss)
    return {"1": round(mise,2)}, "Martingale 1"

def god_mode(bankroll):
    unit = get_unit(bankroll)
    return {"2": 2*unit, "5": 1*unit, "10": 1*unit}, "God Mode"

def god_mode_bonus(bankroll):
    unit = get_unit(bankroll)
    strat = {"2":0.8*unit,"5":0.4*unit,"10":0.4*unit}
    for bonus in ["Coin Flip","Cash Hunt","Pachinko","Crazy Time"]:
        if st.session_state.skip_bonus == bonus: continue
        strat[bonus] = 0.2*unit
    return strat, "God Mode + Bonus"

def one_plus_bonus(bankroll):
    unit = get_unit(bankroll)
    strat = {}
    for bonus in ["Coin Flip","Cash Hunt","Pachinko","Crazy Time"]:
        if st.session_state.skip_bonus == bonus: continue
        strat[bonus] = 0.5*unit
    total_other = sum(strat.values())
    strat["1"] = round(total_other/2,2)
    return strat, "1 + Bonus"

# -----------------------------
# Spin attendu et stratégie
# -----------------------------
def choose_strategy_expected_spin(history, bankroll):
    # Martingale prioritaire
    if st.session_state.martingale_loss > 0:
        strategy, name = martingale_one(bankroll)
        st.session_state.strategy_repeat_count = 0
        st.session_state.last_strategy_name = name
        return strategy, name

    probs = compute_probabilities(history)
    max_seg = max(probs, key=lambda k: probs[k])

    strategies = [
        god_mode(bankroll),
        god_mode_bonus(bankroll),
        one_plus_bonus(bankroll)
    ]
    best_strategy = None
    best_name = "No Bets"
    best_score = 0.0

    for strat_dict, name in strategies:
        score = strat_dict.get(max_seg,0)
        if name == st.session_state.last_strategy_name and st.session_state.strategy_repeat_count >=2:
            score *= 0.0
        if score > best_score:
            best_score = score
            best_strategy = strat_dict
            best_name = name

    if best_score < 0.05:
        best_strategy = {}
        best_name = "No Bets"

    if best_name == st.session_state.last_strategy_name:
        st.session_state.strategy_repeat_count +=1
    else:
        st.session_state.strategy_repeat_count =0
    st.session_state.last_strategy_name = best_name

    return best_strategy, best_name

# -----------------------------
# Calcul gain avec multiplicateur pour tous les segments
# -----------------------------
def calculate_gain(spin, strategy):
    gain = 0
    parts = spin.split("x")
    seg = parts[0].strip()
    mult = int(parts[1]) if len(parts)>1 else 1
    for bet_seg, mise in strategy.items():
        if bet_seg == seg:
            if seg=="1": gain += mise*2*mult
            elif seg=="2": gain += mise*3*mult
            elif seg=="5": gain += mise*6*mult
            elif seg=="10": gain += mise*11*mult
            else:  # bonus
                gain += mise*mult + mise
    return gain

# -----------------------------
# Affichage
# -----------------------------
def display_suggestion(strategy, name):
    st.subheader(f"💡 Suggestion Prochain Spin : {name}")
    if not strategy:
        st.info("No Bets")
        return
    for seg,mise in strategy.items():
        st.write(f"- {seg} → {round(mise,2)} $")

# -----------------------------
# Interface
# -----------------------------
st.title("🎡 Crazy Time Bot (Spin attendu avec multiplicateur)")
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
    st.session_state.last_strategy, _ = choose_strategy_expected_spin(st.session_state.history, st.session_state.bankroll)

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

        # Nouvelle stratégie basée sur spin attendu
        st.session_state.last_strategy, _ = choose_strategy_expected_spin(st.session_state.history, st.session_state.bankroll)

display_suggestion(st.session_state.last_strategy, st.session_state.last_strategy_name)
