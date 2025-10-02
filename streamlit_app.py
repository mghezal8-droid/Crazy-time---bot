import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

# --------------------------
# Répartition réelle Crazy Time
# --------------------------
segment_counts = {
    "1": 21,
    "2": 13,
    "5": 7,
    "10": 4,
    "Coin Flip": 4,
    "Cash Hunt": 2,
    "Pachinko": 2,
    "Crazy Time": 1
}
total_segments = sum(segment_counts.values())

# Paiements standards
payouts = {"1": 1, "2": 2, "5": 5, "10": 10,
           "Coin Flip": 2, "Cash Hunt": 2,
           "Pachinko": 2, "Crazy Time": 2}

# --------------------------
# Initialisation
# --------------------------
if "bankroll" not in st.session_state:
    st.session_state.bankroll = 100
if "history" not in st.session_state:
    st.session_state.history = []
if "base_unit" not in st.session_state:
    st.session_state.base_unit = 1

# --------------------------
# Fonctions
# --------------------------
def calc_probs(history):
    """Calcule les probabilités pondérées par l’historique"""
    probs = {}
    for seg, count in segment_counts.items():
        base_p = count / total_segments
        delay = len(history) - max([i for i, h in enumerate(history) if h == seg], default=-1)
        weight = 1 + (delay / 100)  # pondération légère
        probs[seg] = base_p * weight
    s = sum(probs.values())
    return {k: v / s for k, v in probs.items()}

def expected_value(segment, mise, probs):
    """EV du segment avec payout"""
    p = probs[segment]
    gain = payouts.get(segment, 0) * mise
    return p * gain - mise

def choose_strategy(probs, bankroll, unit):
    """Choix de la stratégie optimale"""
    strategies = {}

    # Exemples simples : miser sur 1 seul segment ou combo
    for seg in segment_counts.keys():
        strategies[f"Bet {seg}"] = {seg: unit}

    # Combos classiques
    strategies["2+5+10"] = {"2": unit, "5": unit, "10": unit}
    strategies["All bonuses"] = {"Coin Flip": unit, "Cash Hunt": unit,
                                 "Pachinko": unit, "Crazy Time": unit}

    # Évaluer EV
    best_name, best_ev, best_strat = "No Bet", -999, {}
    for name, strat in strategies.items():
        ev = sum(expected_value(seg, mise, probs) for seg, mise in strat.items())
        if ev > best_ev:
            best_name, best_ev, best_strat = name, ev, strat

    if best_ev <= 0 or bankroll < unit * len(best_strat):
        return "No Bet", {}

    return best_name, best_strat

def process_spin(result, mult, strat, bankroll):
    """Calcul du résultat du spin"""
    mise_totale = sum(strat.values())
    gain_brut = 0

    if result in strat:
        gain_brut = strat[result] * payouts[result] * mult + strat[result]

    gain_net = gain_brut - mise_totale
    bankroll += gain_net
    return gain_net, bankroll, mise_totale

# --------------------------
# Interface Streamlit
# --------------------------
st.title("🎡 Crazy Time Bot Optimisé")

# Bankroll initiale
init_bankroll = st.sidebar.number_input("Bankroll initiale ($)", 50, 1000, 100)
unit_choice = st.sidebar.radio("Unité de mise de base", [0.5, 1, 2], index=1)
st.session_state.base_unit = unit_choice

if st.sidebar.button("Réinitialiser"):
    st.session_state.bankroll = init_bankroll
    st.session_state.history = []

# --------------------------
# Historique manuel
# --------------------------
st.subheader("Entrer l’historique des spins")
cols = st.columns(4)
for i, seg in enumerate(segment_counts.keys()):
    if cols[i % 4].button(seg):
        st.session_state.history.append(seg)

mult = st.number_input("Multiplicateur Top Slot", 1, 100, 1)

if st.button("Fin historique et commencer"):
    st.session_state.bankroll = init_bankroll

# --------------------------
# Choix stratégie et Spin Live
# --------------------------
if st.session_state.history:
    probs = calc_probs(st.session_state.history)
    strat_name, strat = choose_strategy(probs, st.session_state.bankroll, st.session_state.base_unit)

    st.write(f"**Stratégie suggérée :** {strat_name}")
    if strat:
        df_strat = pd.DataFrame(
            [{"Segment": seg, "Unités": mises / st.session_state.base_unit,
              "Mise ($)": mises} for seg, mises in strat.items()]
        )
        st.table(df_strat)
        st.write(f"**Mise totale : {sum(strat.values())}$**")

    result = st.selectbox("Résultat du spin", list(segment_counts.keys()))
    if st.button("Spin Live"):
        gain_net, new_bankroll, mise_totale = process_spin(result, mult, strat, st.session_state.bankroll)
        st.session_state.bankroll = new_bankroll
        st.session_state.history.append(result)

        st.success(f"Résultat: {result} x{mult} | Gain net: {gain_net}$ | Nouveau bankroll: {new_bankroll}$")

# --------------------------
# Courbe bankroll
# --------------------------
if st.session_state.history:
    bankrolls = []
    br = init_bankroll
    for seg in st.session_state.history:
        _, br, _ = process_spin(seg, 1, {}, br)
        bankrolls.append(br)

    fig, ax = plt.subplots()
    ax.plot(bankrolls, marker="o")
    ax.set_title("Évolution du bankroll spin par spin")
    ax.set_xlabel("Spin")
    ax.set_ylabel("Bankroll ($)")
    st.pyplot(fig)
