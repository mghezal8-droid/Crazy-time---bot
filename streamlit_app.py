import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# -------------------------------
# Initialisation session
# -------------------------------
if 'history' not in st.session_state:
    st.session_state.history = []
if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 150.0   # bankroll du bot
if 'base_unit' not in st.session_state:
    st.session_state.base_unit = 1.0
if 'last_gain' not in st.session_state:
    st.session_state.last_gain = 0.0
if 'initial_bankroll' not in st.session_state:
    st.session_state.initial_bankroll = st.session_state.bankroll
if 'results_table' not in st.session_state:
    st.session_state.results_table = []

segments = ['1', '2', '5', '10', 'Coin Flip', 'Cash Hunt', 'Pachinko', 'Crazy Time']

# -------------------------------
# Barre latérale
# -------------------------------
st.sidebar.header("Paramètres Crazy Time Bot")

bankroll_input = st.sidebar.number_input(
    "Bankroll initial ($)",
    min_value=50.0,
    max_value=1000.0,
    value=float(st.session_state.bankroll),
    step=1.0
)
base_unit_input = st.sidebar.number_input(
    "Unité de base ($)",
    min_value=0.2,
    max_value=10.0,
    value=float(st.session_state.base_unit),
    step=0.1
)

bonus_multiplier_assumption = st.sidebar.number_input(
    "Hypothèse multiplicateur bonus",
    min_value=1,
    max_value=500,
    value=10
)

critical_threshold_pct = st.sidebar.slider(
    "Seuil critique bankroll (%)", min_value=1, max_value=100, value=25, step=1
)

# ⚡ Mise à jour des vraies variables du bot
st.session_state.bankroll = bankroll_input
st.session_state.base_unit = base_unit_input
critical_threshold_value = st.session_state.bankroll * (critical_threshold_pct / 100)

# -------------------------------
# Entrée historique
# -------------------------------
st.header("Historique Spins")

# Boutons compacts
cols = st.columns(8)
for i, seg in enumerate(segments):
    if cols[i].button(seg, use_container_width=True):
        st.session_state.history.append(seg)

if st.button("Supprimer dernier historique"):
    if st.session_state.history:
        st.session_state.history.pop()

if st.button("Fin historique et commencer"):
    st.success(f"Historique enregistré ({len(st.session_state.history)} spins). Le bot est prêt à suggérer.")

# -------------------------------
# Fonctions
# -------------------------------
def compute_segment_probabilities(history):
    # pondération approximative Crazy Time
    segment_count = {
        '1': 21, '2': 13, '5': 7, '10': 4,
        'Coin Flip': 4, 'Cash Hunt': 2, 'Pachinko': 2, 'Crazy Time': 1
    }
    total_segments = sum(segment_count.values())

    base_prob = {k: v / total_segments for k, v in segment_count.items()}
    hist_weight = {k: (history.count(k) / len(history) if history else 0) for k in segment_count.keys()}
    prob = {k: 0.5 * base_prob[k] + 0.5 * hist_weight[k] for k in segment_count.keys()}
    return prob

def adjust_unit(bankroll):
    if bankroll >= 2.5 * st.session_state.initial_bankroll:
        return st.session_state.base_unit * 2
    elif bankroll <= 0.5 * st.session_state.initial_bankroll:
        return st.session_state.base_unit * 0.5
    return st.session_state.base_unit

def choose_strategy_intelligent(history, bankroll):
    if bankroll <= critical_threshold_value:
        return "No-Bet", {k: 0 for k in segments}

    unit = adjust_unit(bankroll)
    probs = compute_segment_probabilities(history)

    # EV simple : proba * gain potentiel
    mult_table = {'1': 2, '2': 3, '5': 6, '10': 11,
                  'Coin Flip': bonus_multiplier_assumption,
                  'Cash Hunt': bonus_multiplier_assumption,
                  'Pachinko': bonus_multiplier_assumption,
                  'Crazy Time': bonus_multiplier_assumption}

    ev = {k: probs[k] * mult_table[k] for k in probs}

    best_seg = max(ev.items(), key=lambda x: x[1])[0]

    # Stratégies variées
    strategies = {}
    strategies[f"EV_{best_seg}"] = {k: unit if k == best_seg else 0 for k in segments}
    strategies["SafeMix"] = {'1': unit, '2': unit, '5': 0, '10': 0, 'Coin Flip': 0, 'Cash Hunt': 0, 'Pachinko': 0, 'Crazy Time': 0}
    strategies["BonusHunt"] = {k: (unit if k in ['Coin Flip', 'Cash Hunt', 'Pachinko', 'Crazy Time'] else 0) for k in segments}
    strategies["1+Bonus"] = {'1': unit * 2, '2': 0, '5': 0, '10': 0, 'Coin Flip': unit, 'Cash Hunt': unit, 'Pachinko': unit, 'Crazy Time': unit}

    best_strategy = max(strategies.items(), key=lambda x: sum(x[1].values()))[0]
    return best_strategy, strategies[best_strategy]

def process_spin(spin_result, multiplier, mises_utilisees, bankroll, last_gain, strategy_name):
    mise_total = sum(mises_utilisees.values())
    gain = 0
    if spin_result in mises_utilisees:
        if spin_result in ['Coin Flip', 'Cash Hunt', 'Pachinko', 'Crazy Time']:
            gain = mises_utilisees[spin_result] * multiplier
        else:
            mult_table = {'1': 2, '2': 3, '5': 6, '10': 11}
            gain = mises_utilisees[spin_result] * mult_table[spin_result]
    gain_net = gain - (mise_total - mises_utilisees.get(spin_result, 0))
    new_bankroll = bankroll + gain_net
    return gain_net, mise_total, new_bankroll, strategy_name, mises_utilisees

# -------------------------------
# Suggestion actuelle
# -------------------------------
if st.session_state.history:
    strategy_name, next_mises = choose_strategy_intelligent(st.session_state.history, st.session_state.bankroll)
    st.subheader("📊 Suggestion pour le prochain spin")
    st.write("Stratégie suggérée:", strategy_name)
    st.write("Mises proposées:", next_mises)

# -------------------------------
# Mode Live
# -------------------------------
st.header("Spin Live")
spin_val = st.selectbox("Spin Sorti", segments)
multiplier_val = st.number_input("Multiplicateur réel", 1, 500, value=1, step=1)

if st.button("Enregistrer Spin"):
    strategy_name, next_mises = choose_strategy_intelligent(st.session_state.history, st.session_state.bankroll)
    gain_net, mise_total, new_bankroll, strategy_name, next_mises = process_spin(
        spin_val, multiplier_val, next_mises, st.session_state.bankroll, st.session_state.last_gain, strategy_name
    )
    st.session_state.history.append(spin_val)
    st.session_state.last_gain = gain_net
    st.session_state.bankroll = new_bankroll

    st.session_state.results_table.append({
        "Spin #": len(st.session_state.results_table) + 1,
        "Résultat": spin_val,
        "Stratégie": strategy_name,
        "Mise Totale": mise_total,
        "Gain Net": gain_net,
        "Bankroll": new_bankroll
    })

    st.success(f"Spin: {spin_val}, Gain Net: {gain_net}, Bankroll: {st.session_state.bankroll}")
    st.write("Prochaine stratégie suggérée:", strategy_name)
    st.write("Mises proposées:", next_mises)

if st.button("Supprimer dernier live spin"):
    if st.session_state.history:
        st.session_state.history.pop()
    if st.session_state.results_table:
        st.session_state.results_table.pop()
    st.warning("Dernier spin supprimé.")

# -------------------------------
# Affichage tableau live
# -------------------------------
st.subheader("📈 Historique des Spins Live")
if st.session_state.results_table:
    df_results = pd.DataFrame(st.session_state.results_table)
    st.dataframe(df_results, use_container_width=True)

    # 📉 Graphique bankroll
    st.subheader("📊 Évolution de la Bankroll")
    fig, ax = plt.subplots()
    ax.plot(df_results["Spin #"], df_results["Bankroll"], marker='o')
    ax.set_xlabel("Spin #")
    ax.set_ylabel("Bankroll ($)")
    ax.set_title("Évolution de la Bankroll")
    st.pyplot(fig)

# -------------------------------
# Test manuel stratégie
# -------------------------------
st.subheader("⚡ Tester une stratégie manuellement")
strategy_choice = st.selectbox(
    "Choisir une stratégie",
    ["EV_1", "EV_2", "EV_5", "EV_10", "EV_Coin Flip", "EV_Cash Hunt", "EV_Pachinko", "EV_Crazy Time",
     "SafeMix", "BonusHunt", "1+Bonus", "No-Bet"]
)

if st.button("Tester Stratégie"):
    if strategy_choice == "No-Bet":
        mises = {k: 0 for k in segments}
    elif strategy_choice.startswith("EV_"):
        target = strategy_choice.replace("EV_", "")
        mises = {k: st.session_state.base_unit if k == target else 0 for k in segments}
    elif strategy_choice == "SafeMix":
        mises = {'1': 1, '2': 1, '5': 0, '10': 0, 'Coin Flip': 0, 'Cash Hunt': 0, 'Pachinko': 0, 'Crazy Time': 0}
    elif strategy_choice == "BonusHunt":
        mises = {k: 1 if k in ['Coin Flip', 'Cash Hunt', 'Pachinko', 'Crazy Time'] else 0 for k in segments}
    elif strategy_choice == "1+Bonus":
        mises = {'1': 2, '2': 0, '5': 0, '10': 0, 'Coin Flip': 1, 'Cash Hunt': 1, 'Pachinko': 1, 'Crazy Time': 1}
    else:
        mises = {k: 0 for k in segments}
    st.info(f"Stratégie {strategy_choice} → Mises: {mises}")
