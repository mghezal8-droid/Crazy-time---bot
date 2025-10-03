import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt

# ============================
# CONFIGURATION INITIALE
# ============================

if "bankroll" not in st.session_state:
    st.session_state.bankroll = 100.0
if "start_bankroll" not in st.session_state:
    st.session_state.start_bankroll = 100.0
if "history" not in st.session_state:
    st.session_state.history = []
if "current_strategy" not in st.session_state:
    st.session_state.current_strategy = None
if "tested_results" not in st.session_state:
    st.session_state.tested_results = []

# ============================
# STRATEGIES
# ============================

strategies = {
    "Martingale": {"1": 1.0},
    "No Bet": {},
    "God Mode": {"2": 3.0, "5": 2.0, "10": 1.0},  # Total 6$
    "God Mode + Bonus": {"2": 3.0, "5": 2.0, "10": 1.0,
                         "Cash Hunt": 1.0, "Pachinko": 1.0,
                         "Coin Flip": 1.0, "Crazy Time": 1.0},  # Total 10$
    "1 + Bonus": {"1": 4.0, "Cash Hunt": 1.0,
                  "Pachinko": 1.0, "Coin Flip": 1.0,
                  "Crazy Time": 1.0}  # Total 8$
}

# Segments et probabilités réelles approximatives de Crazy Time
segment_probs = {
    "1": 0.382, "2": 0.255, "5": 0.074, "10": 0.046,
    "Coin Flip": 0.074, "Cash Hunt": 0.046,
    "Pachinko": 0.046, "Crazy Time": 0.011
}

# Paiements standards
payouts = {"1": 1, "2": 2, "5": 5, "10": 10,
           "Coin Flip": 2, "Cash Hunt": 5,
           "Pachinko": 10, "Crazy Time": 20}


# ============================
# FONCTIONS
# ============================

def suggest_strategy():
    """Propose une stratégie variée basée sur historique, EV et tendances"""
    history = st.session_state.history
    bankroll = st.session_state.bankroll

    # Si pas d'historique -> stratégie de base
    if len(history) < 5:
        return random.choice(list(strategies.keys()))

    # Compter fréquence de sortie récente
    recent = pd.Series(history[-20:]).value_counts(normalize=True)

    # Calcul EV pour chaque stratégie
    ev_scores = {}
    for strat, bets in strategies.items():
        ev = 0
        for seg, amount in bets.items():
            prob = segment_probs.get(seg, 0)
            ev += prob * (amount * payouts.get(seg, 0)) - amount * prob
        # bonus si couvre une tendance forte
        for seg in recent.index:
            if seg in bets:
                ev *= 1.1
        ev_scores[strat] = ev

    # Sélection pondérée pour varier
    strat_list = list(ev_scores.keys())
    weights = [max(0.01, ev) for ev in ev_scores.values()]
    chosen = random.choices(strat_list, weights=weights, k=1)[0]

    return chosen


def apply_strategy(strategy_name, spin_result):
    """Applique une stratégie et met à jour la bankroll"""
    bets = strategies[strategy_name]
    gain = 0
    total_bet = 0

    for seg, amount in bets.items():
        total_bet += amount
        if seg == spin_result:
            gain += amount * payouts[seg]

    net = gain - total_bet
    st.session_state.bankroll += net
    return net


def adjust_bets_by_bankroll(bets):
    """Ajuste la taille des mises selon bankroll"""
    factor = st.session_state.bankroll / st.session_state.start_bankroll
    adjusted = {k: round(v * factor, 2) for k, v in bets.items()}
    return adjusted


def plot_bankroll_evolution():
    """Trace l’évolution bankroll lors d’un test manuel"""
    if not st.session_state.tested_results:
        return

    results = st.session_state.tested_results
    x = list(range(len(results)))
    y = [st.session_state.start_bankroll]
    for r in results:
        y.append(y[-1] + r)

    fig, ax = plt.subplots()
    ax.plot(x, y[1:], marker="o")
    ax.set_title("Évolution de la bankroll (test manuel)")
    ax.set_xlabel("Spin")
    ax.set_ylabel("Bankroll")
    st.pyplot(fig)


# ============================
# INTERFACE STREAMLIT
# ============================

st.title("🎰 Crazy Time Bot")

# Historique manuel
st.subheader("📜 Entrée de l’historique")
col1, col2, col3, col4 = st.columns(4)
if col1.button("1"): st.session_state.history.append("1")
if col2.button("2"): st.session_state.history.append("2")
if col3.button("5"): st.session_state.history.append("5")
if col4.button("10"): st.session_state.history.append("10")

colb1, colb2, colb3, colb4 = st.columns(4)
if colb1.button("Coin Flip"): st.session_state.history.append("Coin Flip")
if colb2.button("Cash Hunt"): st.session_state.history.append("Cash Hunt")
if colb3.button("Pachinko"): st.session_state.history.append("Pachinko")
if colb4.button("Crazy Time"): st.session_state.history.append("Crazy Time")

st.write("Historique actuel :", st.session_state.history)

# Suggestion de stratégie
if st.button("🎯 Suggérer une stratégie"):
    chosen = suggest_strategy()
    st.session_state.current_strategy = chosen
    st.success(f"Stratégie suggérée : **{chosen}**")
    st.write("Mises :", adjust_bets_by_bankroll(strategies[chosen]))

# Appliquer un spin en live
if st.session_state.current_strategy:
    spin_result = st.selectbox("Résultat du spin :", list(segment_probs.keys()))
    if st.button("✅ Enregistrer spin"):
        net = apply_strategy(st.session_state.current_strategy, spin_result)
        st.write(f"Résultat net du spin : {net:.2f}")
        st.write(f"Bankroll actuelle : {st.session_state.bankroll:.2f}")

# ============================
# TEST MANUEL STRATEGIE
# ============================

st.subheader("🧪 Tester une stratégie manuellement")

selected_test = st.selectbox("Choisir une stratégie à tester :", list(strategies.keys()))

if st.button("▶️ Lancer le test (20 spins simulés)"):
    st.session_state.tested_results = []
    bankroll = st.session_state.start_bankroll

    for _ in range(20):
        spin = random.choices(list(segment_probs.keys()),
                              weights=segment_probs.values())[0]
        bets = strategies[selected_test]
        net = 0
        for seg, amount in bets.items():
            if seg == spin:
                net += amount * payouts[seg]
            net -= amount
        st.session_state.tested_results.append(net)

    st.success(f"Test terminé pour stratégie {selected_test}")
    plot_bankroll_evolution()
