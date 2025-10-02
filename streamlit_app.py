import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# -------------------------------
# Initialisation session
# -------------------------------
if 'history' not in st.session_state:
    st.session_state.history = []
if 'live_history' not in st.session_state:
    st.session_state.live_history = []
if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 150.0
if 'base_unit' not in st.session_state:
    st.session_state.base_unit = 1.0
if 'last_gain' not in st.session_state:
    st.session_state.last_gain = 0.0

segments = ['1','2','5','10','Bonus']

# -------------------------------
# Barre latérale
# -------------------------------
st.sidebar.header("Paramètres Crazy Time Bot")
st.sidebar.number_input(
    "Bankroll initial ($)",
    min_value=50.0,
    max_value=1000.0,
    value=float(st.session_state.bankroll),
    step=1.0,
    key='bankroll'
)
st.sidebar.number_input(
    "Unité de base ($)",
    min_value=0.2,
    max_value=10.0,
    value=float(st.session_state.base_unit),
    step=0.1,
    key='base_unit'
)
bonus_multiplier_assumption = st.sidebar.number_input(
    "Hypothèse multiplicateur bonus",
    min_value=1,
    max_value=50,
    value=10
)
critical_threshold_pct = st.sidebar.slider(
    "Seuil critique bankroll (%)", min_value=1, max_value=100, value=25, step=1
)
critical_threshold_value = st.session_state.bankroll * (critical_threshold_pct/100)

# -------------------------------
# Entrée historique manuel
# -------------------------------
st.header("Historique Spins (Manuel)")
cols = st.columns([1]*6)
for i, seg in enumerate(segments):
    if cols[i].button(seg):
        st.session_state.history.append(seg)

col_clear = st.columns([1]*2)
if col_clear[0].button("Supprimer dernier historique"):
    if st.session_state.history:
        st.session_state.history.pop()

if col_clear[1].button("Fin historique et commencer"):
    st.success(f"Historique enregistré ({len(st.session_state.history)} spins)")

# -------------------------------
# Fonctions stratégies & EV intelligent
# -------------------------------
def compute_segment_probabilities(history):
    segment_count = {'1':1,'2':2,'5':2,'10':1,'Bonus':1}
    total_segments = sum(segment_count.values())
    base_prob = {k:v/total_segments for k,v in segment_count.items()}
    hist_weight = {k:(history.count(k)/len(history) if history else 0) for k in segment_count.keys()}
    prob = {k: 0.5*base_prob[k] + 0.5*hist_weight[k] for k in segment_count.keys()}
    return prob

def choose_strategy_intelligent(history, bankroll):
    if bankroll <= critical_threshold_value:
        return "No-Bet", {k:0 for k in segments}
    probs = compute_segment_probabilities(history)
    sorted_probs = sorted(probs.items(), key=lambda x:x[1], reverse=True)
    target_segment = sorted_probs[0][0]
    strategy_name = f"Martingale_{target_segment}"
    mises = {k: st.session_state.base_unit if k==target_segment else 0 for k in segments}
    return strategy_name, mises

def process_spin(spin_result, multiplier, mises_utilisees, bankroll, last_gain, strategy_name):
    mise_total = sum(mises_utilisees.values())
    gain = 0
    if spin_result in mises_utilisees:
        if spin_result=='Bonus':
            gain = mises_utilisees[spin_result]*multiplier
        else:
            mult_table = {'1':2,'2':3,'5':6,'10':11}
            gain = mises_utilisees[spin_result]*mult_table[spin_result]
    gain_net = gain - (mise_total - mises_utilisees.get(spin_result,0))
    new_bankroll = bankroll + gain_net

    next_mises = mises_utilisees.copy()
    if 'Martingale' in strategy_name:
        if gain_net <= 0:
            next_mises = {k:v*2 for k,v in mises_utilisees.items()}
        else:
            next_mises = {k:(st.session_state.base_unit if v>0 else 0) for k,v in mises_utilisees.items()}

    return float(gain_net), float(mise_total), float(new_bankroll), strategy_name, next_mises

# -------------------------------
# Prochaine stratégie suggérée (avant live spin)
# -------------------------------
st.header("Prochaine stratégie suggérée")
suggestion_name, suggestion_mises = choose_strategy_intelligent(st.session_state.history, st.session_state.bankroll)
st.write("Stratégie:", suggestion_name)
st.write("Mises proposées:", suggestion_mises)

# -------------------------------
# Mode Live
# -------------------------------
st.header("Spin Live")
live_cols = st.columns([1]*6)
spin_val = st.selectbox("Spin Sorti", segments)
multiplier_val = st.number_input("Multiplicateur réel", 1, 50, value=1, step=1)

col_live = st.columns([1]*2)
if col_live[0].button("Enregistrer Spin"):
    gain_net, mise_total, new_bankroll, strategy_name, next_mises = process_spin(
        spin_val, multiplier_val, suggestion_mises, st.session_state.bankroll, st.session_state.last_gain, suggestion_name
    )
    st.session_state.bankroll = new_bankroll
    st.session_state.last_gain = gain_net
    st.session_state.live_history.append(spin_val)
    st.success(f"Spin enregistré: {spin_val}, Gain Net: {gain_net}, Bankroll: {st.session_state.bankroll}")
    st.write("Prochaine stratégie suggérée:", strategy_name)
    st.write("Mises proposées:", next_mises)

if col_live[1].button("Supprimer dernier live spin"):
    if st.session_state.live_history:
        st.session_state.live_history.pop()

# -------------------------------
# Tableau historique complet
# -------------------------------
if st.session_state.history or st.session_state.live_history:
    df_rows = []
    temp_bankroll = st.session_state.bankroll
    temp_history = st.session_state.history + st.session_state.live_history
    for idx, spin in enumerate(temp_history):
        strategy_name, mises = choose_strategy_intelligent(temp_history[:idx], temp_bankroll)
        multiplier = bonus_multiplier_assumption if spin=='Bonus' else {'1':2,'2':3,'5':6,'10':11}[spin]
        gain = mises[spin]*multiplier if spin in mises else 0
        mise_total = sum(mises.values())
        gain_net = gain - (mise_total - mises.get(spin,0))
        temp_bankroll += gain_net
        df_rows.append({
            'Spin n°': idx+1,
            'Segment': spin,
            'Stratégie': strategy_name,
            'Mises $': {k:round(v,2) for k,v in mises.items()},
            'Total Mise': round(mise_total,2),
            'Gain Net': round(gain_net,2),
            'Bankroll': round(temp_bankroll,2)
        })
    df_hist = pd.DataFrame(df_rows)
    st.subheader("Tableau Historique Spin by Spin")
    st.dataframe(df_hist)

    st.subheader("Graphique Bankroll Spin by Spin")
    plt.figure(figsize=(10,4))
    plt.plot(df_hist['Spin n°'], df_hist['Bankroll'], marker='o')
    plt.xlabel("Spin n°")
    plt.ylabel("Bankroll ($)")
    plt.grid(True)
    st.pyplot(plt)

# -------------------------------
# Simulation Batch
# -------------------------------
st.subheader("Simulation Batch")
batch_units = st.multiselect(
    "Unités à tester ($)", options=[0.5,1,2,5], default=[st.session_state.base_unit]
)

if st.button("Appliquer Batch sur toute l'historique") or st.button("Lancer Simulation Batch"):
    batch_results = []
    for unit in batch_units:
        bankroll_sim = st.session_state.bankroll
        history_sim = st.session_state.history + st.session_state.live_history
        spin_results_sim = []

        for spin_idx, spin_val in enumerate(history_sim):
            strategy_name, next_mises = choose_strategy_intelligent(history_sim[:spin_idx], bankroll_sim)
            next_mises_unit = {seg: (v/unit)*unit for seg,v in next_mises.items()}

            mise_total = sum(next_mises_unit.values())
            gain = 0
            if spin_val in next_mises_unit:
                multiplier = bonus_multiplier_assumption if spin_val=='Bonus' else {'1':2,'2':3
