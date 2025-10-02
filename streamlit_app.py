import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- INITIALISATION SESSION ---
if 'history' not in st.session_state:
    st.session_state.history = []

if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 150.0

if 'base_unit' not in st.session_state:
    st.session_state.base_unit = 1.0

if 'dynamic_unit' not in st.session_state:
    st.session_state.dynamic_unit = st.session_state.base_unit

if 'results' not in st.session_state:
    st.session_state.results = pd.DataFrame(columns=['Spin','Segment','Unit','Total Mise','Gain Net','Bankroll','Stratégie'])

if 'last_spin_result' not in st.session_state:
    st.session_state.last_spin_result = 0.0

# --- SIDEBAR ---
st.sidebar.header("Paramètres du Bot")
st.session_state.bankroll = st.sidebar.number_input("Bankroll initial ($)", 50, 1000, value=int(st.session_state.bankroll))
st.session_state.base_unit = st.sidebar.selectbox("Unité de base ($)", [0.5,1,2,5,10], index=1)
st.session_state.dynamic_unit = st.session_state.base_unit
no_bet_threshold = st.sidebar.slider("Seuil bankroll critique (%)", 10, 100, 50)

# --- FONCTIONS ---
def calculate_gain(segment, unit, multiplier):
    if segment.startswith('Spin Lettre'):
        gain = (unit * 25 * multiplier) + unit
    elif segment in ["Crazy Time","Coin Flip","Cash Hunt","Pachinko"]:
        gain = (unit * multiplier) + unit
    else:
        gain = 0
    return gain

def apply_martingale(unit, gain_net, base_unit):
    return unit*2 if gain_net <= 0 else base_unit

def update_bankroll(bankroll, total_mise, gain_net):
    return bankroll - total_mise + gain_net

def suggest_units(bankroll, base_unit):
    return max(base_unit, round(bankroll/150,2))

def process_spin(segment, multiplier, last_unit, base_unit, strategy_name):
    unit = suggest_units(st.session_state.bankroll, last_unit)
    total_mise = unit*13 if strategy_name.startswith("Martingale Lettres + Staying Alive") else unit*3  # ajustement simple pour God Mode
    gain_net = calculate_gain(segment, unit, multiplier)
    bankroll = update_bankroll(st.session_state.bankroll, total_mise, gain_net)
    next_unit = apply_martingale(unit, gain_net, base_unit)
    return gain_net, total_mise, bankroll, next_unit

# --- INTERFACE ---
st.title("Crazy Time Bot Optimisé")

# Historique manuel
st.subheader("Historique des spins")
segments = ['1','2','5','10','Crazy Time','Coin Flip','Cash Hunt','Pachinko']
cols = st.columns(4)
for i, seg in enumerate(segments):
    if cols[i%4].button(seg):
        st.session_state.history.append(seg)

# Live Spin
st.subheader("Live Spin")
selected_spin = st.selectbox("Sélectionner segment live", segments)
mult = st.number_input("Multiplier (1 si pas de multiplicateur)", 1.0, 100.0, 1.0, step=0.1)
strategy_options = ["Martingale Lettres + Staying Alive","God Mode 2,5,10","God Mode 2,5,10 + Bonus","1+Bonus Combo"]
strategy_name = st.selectbox("Stratégie live", strategy_options)

if st.button("Spin Live"):
    gain, total_mise, new_bankroll, next_unit = process_spin(selected_spin, mult, st.session_state.dynamic_unit, st.session_state.base_unit, strategy_name)
    st.session_state.dynamic_unit = next_unit
    st.session_state.bankroll = new_bankroll
    st.session_state.results = pd.concat([st.session_state.results, pd.DataFrame({
        'Spin':[len(st.session_state.results)+1],
        'Segment':[selected_spin],
        'Unit':[next_unit],
        'Total Mise':[total_mise],
        'Gain Net':[gain],
        'Bankroll':[new_bankroll],
        'Stratégie':[strategy_name]
    })], ignore_index=True)
    st.write(f"Gain net: {gain}, Nouveau Bankroll: {new_bankroll}")
    if new_bankroll < st.session_state.base_unit * (no_bet_threshold/100):
        st.warning("Bankroll critique! No-bet recommandé.")

# Fin historique et simulation
if st.button("Fin Historique et Commencer"):
    st.write("Simulation sur historique...")
    for seg in st.session_state.history:
        gain, total_mise, new_bankroll, next_unit = process_spin(seg, 1.0, st.session_state.dynamic_unit, st.session_state.base_unit, strategy_name)
        st.session_state.dynamic_unit = next_unit
        st.session_state.bankroll = new_bankroll
        st.session_state.results = pd.concat([st.session_state.results, pd.DataFrame({
            'Spin':[len(st.session_state.results)+1],
            'Segment':[seg],
            'Unit':[next_unit],
            'Total Mise':[total_mise],
            'Gain Net':[gain],
            'Bankroll':[new_bankroll],
            'Stratégie':[strategy_name]
        })], ignore_index=True)

# Tableau résultats
st.subheader("Résultats Spin par Spin")
st.dataframe(st.session_state.results)

# Graphique bankroll
st.subheader("Évolution du Bankroll")
fig, ax = plt.subplots()
ax.plot(st.session_state.results['Spin'], st.session_state.results['Bankroll'], marker='o')
ax.set_xlabel("Spin")
ax.set_ylabel("Bankroll ($)")
st.pyplot(fig)
