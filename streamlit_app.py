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
    st.session_state.bankroll = 150
if 'base_unit' not in st.session_state:
    st.session_state.base_unit = 1
if 'last_gain' not in st.session_state:
    st.session_state.last_gain = 0
if 'bankroll_progress' not in st.session_state:
    st.session_state.bankroll_progress = [st.session_state.bankroll]

# -------------------------------
# Barre latérale
# -------------------------------
st.sidebar.header("Paramètres Crazy Time Bot")
st.sidebar.number_input("Bankroll initial ($)", 50, 1000, key='bankroll')
st.sidebar.number_input("Unité de base ($)", 0.2, 10, step=0.1, key='base_unit')
bonus_multiplier_assumption = st.sidebar.number_input("Hypothèse multiplicateur bonus", 1, 50, value=10)

# -------------------------------
# Entrée historique
# -------------------------------
st.header("Historique Spins")
cols = st.columns([1]*6)
segments = ['1','2','5','10','Bonus']
for i, seg in enumerate(segments):
    if cols[i].button(seg):
        st.session_state.history.append(seg)

if st.button("Fin historique et commencer"):
    st.success(f"Historique enregistré ({len(st.session_state.history)} spins)")

# -------------------------------
# Fonctions stratégies & EV
# -------------------------------
def choose_strategy(history, bankroll):
    freq = {s: history.count(s)/len(history) if history else 1/len(segments) for s in segments}
    strategies = ['Martingale1','GodMode','GodMode+Bonus','1+Bonus','NoBet']
    # Choisir stratégie max EV (simplifiée)
    best = max(strategies, key=lambda s: freq.get('1',0))
    if best=='Martingale1':
        mises = {'1':st.session_state.base_unit}
    elif best=='GodMode':
        mises = {'2':2*st.session_state.base_unit,'5':st.session_state.base_unit,'10':st.session_state.base_unit}
    elif best=='GodMode+Bonus':
        mises = {'2':0.8*st.session_state.base_unit,'5':0.4*st.session_state.base_unit,'10':0.4*st.session_state.base_unit,'Bonus':0.2*st.session_state.base_unit}
    elif best=='1+Bonus':
        mises = {'1':0.5*st.session_state.base_unit,'Bonus':0.5*st.session_state.base_unit}
    else:
        mises = {}
    return best, mises

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

    # Martingale simple : doubler après perte si stratégie Martingale1
    next_mises = mises_utilisees.copy()
    if strategy_name=='Martingale1':
        if gain_net <=0:
            next_mises = {k:v*2 for k,v in mises_utilisees.items()}
        else:
            next_mises = {k:st.session_state.base_unit for k in mises_utilisees.keys()}

    return gain_net, mise_total, new_bankroll, strategy_name, next_mises

# -------------------------------
# Mode Live
# -------------------------------
st.header("Spin Live")
live_cols = st.columns([1]*6)
spin_val = st.selectbox("Spin Sorti", segments)
multiplier_val = st.number_input("Multiplicateur réel", 1, 50, value=1, step=1)

if st.button("Enregistrer Spin"):
    strategy_name, next_mises = choose_strategy(st.session_state.history, st.session_state.bankroll)
    gain_net, mise_total, st.session_state.bankroll, strategy_name, next_mises = process_spin(
        spin_val, multiplier_val, next_mises, st.session_state.bankroll, st.session_state.last_gain, strategy_name
    )
    st.session_state.history.append(spin_val)
    st.session_state.last_gain = gain_net
    st.session_state.bankroll_progress.append(st.session_state.bankroll)
    st.success(f"Spin enregistré: {spin_val}, Gain Net: {gain_net}, Bankroll: {st.session_state.bankroll}")
    st.write("Prochaine stratégie suggérée:", strategy_name)
    st.write("Mises proposées:", next_mises)

# -------------------------------
# Tableau historique & graphique
# -------------------------------
if st.session_state.history:
    df_hist = pd.DataFrame({
        'Spin n°': list(range(1,len(st.session_state.history)+1)),
        'Segment': st.session_state.history,
        'Bankroll': st.session_state.bankroll_progress
    })
    st.subheader("Tableau Historique")
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

if st.button("Lancer Simulation Batch"):
    batch_results = []
    for unit in batch_units:
        bankroll_sim = st.session_state.bankroll
        history_sim = st.session_state.history.copy()
        spin_results_sim = []

        for spin_idx, spin_val in enumerate(history_sim):
            strategy_name, next_mises = choose_strategy(history_sim[:spin_idx], bankroll_sim)
            next_mises_unit = {seg: mise/unit*unit for seg, mise in next_mises.items()}

            mise_total = sum(next_mises_unit.values())
            gain = 0
            if spin_val in next_mises_unit:
                multiplier = bonus_multiplier_assumption if spin_val=='Bonus' else {'1':2,'2':3,'5':6,'10':11}[spin_val]
                gain = next_mises_unit[spin_val]*multiplier
            gain_net = gain - (mise_total - next_mises_unit.get(spin_val,0))
            bankroll_sim += gain_net

            spin_results_sim.append({
                'Spin': spin_idx+1,
                'Segment': spin_val,
                'Stratégie': strategy_name,
                'Unit': unit,
                'Mises': next_mises_unit,
                'Total Mise': mise_total,
                'Gain Net': gain_net,
                'Bankroll': bankroll_sim
            })

        batch_results.extend(spin_results_sim)

    df_batch = pd.DataFrame(batch_results)
    st.dataframe(df_batch)

    st.subheader("Graphique bankroll - Simulation Batch")
    plt.figure(figsize=(12,5))
    for unit in batch_units:
        df_unit = df_batch[df_batch['Unit']==unit]
        plt.plot(df_unit['Spin'], df_unit['Bankroll'], marker='o', label=f'Unit ${unit}')
    plt.xlabel("Spin n°")
    plt.ylabel("Bankroll ($)")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)
