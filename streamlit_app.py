import streamlit as st
import pandas as pd
import random
import math

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
    st.session_state.bankroll = 100
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
    if base < 0.4:
        return 0.2
    elif base < 1:
        return 0.4
    elif base < 2:
        return 1
    elif base < 4:
        return 2
    elif base < 10:
        return 4
    else:
        return 10

# -----------------------------
# Définition des stratégies
# -----------------------------
def martingale_one(bankroll):
    unit = get_unit(bankroll)
    mise = unit * (2 ** st.session_state.martingale_loss)
    return {"1": mise}

def god_mode(bankroll):
    unit = get_unit(bankroll)
    return {"2": 2 * unit, "5": 1 * unit, "10": 1 * unit}

def god_mode_bonus(bankroll):
    unit = get_unit(bankroll)
    strat = {"2": 2 * unit, "5": 1 * unit, "10": 1 * unit}
    for bonus in ["Coin Flip", "Cash Hunt", "Pachinko", "Crazy Time"]:
        if st.session_state.skip_bonus == bonus:
            continue
        strat[bonus] = 1 * unit
    return strat

def one_plus_bonus(bankroll):
    unit = get_unit(bankroll)
    strat = {"1": 1 * unit}
    for bonus in ["Coin Flip", "Cash Hunt", "Pachinko", "Crazy Time"]:
        if st.session_state.skip_bonus == bonus:
            continue
        strat[bonus] = 1 * unit
    return strat

# -----------------------------
# Choisir stratégie
# -----------------------------
def choose_strategy(history, bankroll):
    if not history:
        return one_plus_bonus(bankroll)

    last_spin = history[-1]["result"]

    if any(b in last_spin for b in ["Coin Flip", "Cash Hunt", "Pachinko", "Crazy Time"]):
        st.session_state.skip_bonus = last_spin.split(" ")[0]
    else:
        st.session_state.skip_bonus = None

    strategies = [
        martingale_one(bankroll),
        god_mode(bankroll),
        god_mode_bonus(bankroll),
        one_plus_bonus(bankroll),
    ]
    return random.choice(strategies)

# -----------------------------
# Calcul gain
# -----------------------------
def calculate_gain(spin, strategy):
    gain = 0
    spin_parts = spin.split("x")
    seg = spin_parts[0].strip()
    mult = int(spin_parts[1]) if len(spin_parts) > 1 else 1

    for bet_seg, mise in strategy.items():
        if bet_seg == seg:
            if seg == "1":
                gain += mise * 2
            elif seg == "2":
                gain += mise * 3
            elif seg == "5":
                gain += mise * 6
            elif seg == "10":
                gain += mise * 11
            else:  # Bonus
                gain += mise * (mult + 1)  # mise + (mise × multiplicateur)
    return gain

# -----------------------------
# Afficher suggestion
# -----------------------------
def display_suggestion(strategy):
    st.subheader("💡 Suggestion Prochain Spin")
    if strategy == None:
        st.info("Aucune stratégie pour l’instant. Ajoute de l’historique ou un spin live.")
        return
    for seg, mise in strategy.items():
        st.write(f"- {seg} → {round(mise,2)} $")

# -----------------------------
# Interface
# -----------------------------
st.title("🎡 Crazy Time Bot")

st.sidebar.header("⚙️ Paramètres")
st.session_state.bankroll = st.sidebar.number_input("Bankroll initiale ($)", 50, 1000, st.session_state.bankroll)

st.subheader("📜 Historique des Spins")
new_spin = st.text_input("Entre un résultat (ex: 1, 2, 5, 10, Coin Flip, Pachinko, Crazy Time):")

if st.button("Ajouter à l’historique"):
    if new_spin:
        st.session_state.history.append({"result": new_spin, "gain": 0})
        st.success(f"Ajouté: {new_spin}")

if st.button("Historique terminé"):
    st.session_state.last_strategy = choose_strategy(st.session_state.history, st.session_state.bankroll)

st.write(pd.DataFrame(st.session_state.history))

# -----------------------------
# Spin Live
# -----------------------------
st.subheader("🎯 Spin Live")
live_spin = st.text_input("Résultat Spin Live (ex: 5, Pachinko x20, Coin Flip x3):")

if st.button("Résultat Spin Live"):
    if live_spin:
        gain = calculate_gain(live_spin, st.session_state.last_strategy)

        total_bet = sum(st.session_state.last_strategy.values())
        net = gain - total_bet
        st.session_state.bankroll += net

        if "1" in st.session_state.last_strategy:
            if "1" in live_spin:
                st.session_state.martingale_loss = 0
            else:
                st.session_state.martingale_loss += 1

        st.session_state.history.append({"result": live_spin, "gain": gain})
        st.success(f"Résultat {live_spin} → Gain {round(gain,2)} | Bankroll: {round(st.session_state.bankroll,2)}")

        st.session_state.last_strategy = choose_strategy(st.session_state.history, st.session_state.bankroll)

# -----------------------------
# Suggestion affichée
# -----------------------------
display_suggestion(st.session_state.last_strategy)
