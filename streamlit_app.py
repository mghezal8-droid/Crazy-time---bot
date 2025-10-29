# streamlit_app_final.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------
# 🔧 CONFIG INITIALE
# -----------------------------------
st.set_page_config(page_title="🎰 Crazy Time Tracker + EV + ML", layout="wide")

VAL_SEG = {'1':1,'2':2,'5':5,'10':10}
SEGMENTS = ['1','2','5','10','Coin Flip','Cash Hunt','Pachinko','Crazy Time']
THEO_COUNTS = {'1':21,'2':13,'5':7,'10':4,'Coin Flip':4,'Cash Hunt':2,'Pachinko':2,'Crazy Time':1}
THEO_TOTAL = sum(THEO_COUNTS.values())

# -----------------------------------
# ⚙️ INIT session_state
# -----------------------------------
for key, default in [
    ("bankroll",150.0),
    ("initial_bankroll",150.0),
    ("history",[]),
    ("live_history",[]),
    ("results_table",[]),
    ("martingale_1_loss_streak",0),
    ("miss_streak",0),
    ("last_suggestion_name",None),
    ("last_suggestion_mises",{}),
    ("bonus_multiplier_assumption",10),
    ("mult_for_ev",1),
    ("show_history_table", True),
    ("rtp_weight", 50),
    ("ml_window", 10)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# -----------------------------------
# 🎯 STRATÉGIES
# -----------------------------------
def strategy_martingale_1(bankroll, loss_streak):
    base_bet = 4.0
    mise_1 = base_bet * (2**loss_streak)
    return "Martingale 1", {'1': mise_1}

def strategy_god_mode(bankroll):
    return "God Mode", {'2':3.0,'5':2.0,'10':1.0}

def strategy_god_mode_bonus(bankroll):
    return "God Mode + Bonus", {'2':3.0,'5':2.0,'10':1.0,
                                'Coin Flip':1.0,'Cash Hunt':1.0,'Pachinko':1.0,'Crazy Time':1.0}

def strategy_1_bonus(bankroll):
    return "1 + Bonus", {'1':4.0,'Coin Flip':1.0,'Cash Hunt':1.0,'Pachinko':1.0,'Crazy Time':1.0}

def strategy_only_numbers(bankroll):
    return "Only Numbers", {'1':3.0,'2':2.0,'5':1.0,'10':1.0}

def strategy_all_but_1(bankroll):
    return "All but 1", {'2':2.0,'5':2.0,'10':2.0,'Coin Flip':1.0,'Cash Hunt':1.0,'Pachinko':1.0,'Crazy Time':1.0}

# -----------------------------------
# 🧠 UTILITAIRES EV / PROBABILITÉS
# -----------------------------------
def theo_prob(segment):
    return THEO_COUNTS.get(segment,0)/THEO_TOTAL

def hist_prob(full_history, segment, window=300):
    if not full_history: return 0.0
    hist = full_history[-window:]
    return hist.count(segment)/len(hist)

def combined_prob(full_history, segment, window=300):
    return 0.5*(theo_prob(segment)+hist_prob(full_history, segment, window=window))

# -----------------------------------
# 🧮 CALCUL GAIN
# -----------------------------------
def calcul_gain(mises, spin_result, multiplicateur):
    if not mises: return 0.0, 0.0
    mise_totale = sum(mises.values())
    gain_brut = 0.0
    if spin_result in mises:
        n = mises[spin_result]
        if spin_result in ['1','2','5','10']:
            factor = {'1':2,'2':3,'5':6,'10':11}[spin_result]
            gain_brut = n*factor
        else:  # Bonus
            gain_brut = n * multiplicateur + n
    gain_net = gain_brut - mise_totale
    return float(gain_brut), float(gain_net)

# -----------------------------------
# 🧠 CHOIX STRATÉGIE
# -----------------------------------
def choose_strategy_intelligent(full_history, bankroll, multiplicateur):
    if st.session_state.martingale_1_loss_streak>0:
        return strategy_martingale_1(bankroll, st.session_state.martingale_1_loss_streak)
    if st.session_state.miss_streak>=3:
        return strategy_martingale_1(bankroll,0)
    candidates=[]
    for builder in [strategy_only_numbers,strategy_god_mode,strategy_god_mode_bonus,strategy_1_bonus,strategy_all_but_1]:
        name, mises = builder(bankroll)
        candidates.append((name,mises))
    # Pour simplification: on choisit la première stratégie comme exemple
    return candidates[0]

# -----------------------------------
# 🔘 HISTORIQUE MANUEL
# -----------------------------------
st.header("📝 Historique Manuel")
def segment_buttons_grid(segments, cols_per_row=4):
    rows = (len(segments)+cols_per_row-1)//cols_per_row
    idx = 0
    for r in range(rows):
        cols = st.columns(cols_per_row)
        for c in range(cols_per_row):
            if idx >= len(segments): break
            seg = segments[idx]
            if cols[c].button(seg,key=f"hist_{seg}_{idx}"):
                st.session_state.history.append(seg)
            idx += 1
segment_buttons_grid(SEGMENTS)

col_a, col_b, col_c = st.columns([1,1,1])
with col_a:
    if st.button("↩ Supprimer dernier spin historique") and st.session_state.history:
        st.session_state.history.pop()
with col_b:
    if st.button("🔄 Réinitialiser historique manuel"):
        st.session_state.history=[]
with col_c:
    if st.button("🏁 Terminer historique"):
        full_history = st.session_state.history + st.session_state.live_history
        next_name, next_mises = choose_strategy_intelligent(full_history, st.session_state.bankroll, st.session_state["mult_for_ev"])
        st.session_state.last_suggestion_name = next_name
        st.session_state.last_suggestion_mises = next_mises

# Affichage historique si coché
if st.session_state.show_history_table and st.session_state.history:
    st.subheader("📋 Historique Manuel Actuel")
    df_manual = pd.DataFrame({"#": range(1,len(st.session_state.history)+1),"Résultat": st.session_state.history})
    st.dataframe(df_manual,use_container_width=True)

# -----------------------------------
# Sidebar Paramètres
# -----------------------------------
st.sidebar.header("Paramètres")
mult_for_ev_input = st.sidebar.number_input("Multiplicateur manuel (pour EV)", min_value=1,max_value=200,value=st.session_state["mult_for_ev"],step=1)
st.session_state["mult_for_ev"]=mult_for_ev_input
bonus_ass = st.sidebar.number_input("Hypothèse multiplicateur bonus",min_value=1,max_value=1000,value=st.session_state.bonus_multiplier_assumption,step=1)
st.session_state.bonus_multiplier_assumption=int(bonus_ass)
st.sidebar.checkbox("Afficher tableau historique", value=st.session_state.show_history_table,key="show_history_table")

# -----------------------------------
# 🧮 SPINS LIVE
# -----------------------------------
st.title("🎡 Crazy Time Live Tracker")
col1,col2,col3 = st.columns([1,1,1])
with col1:
    spin_val = st.selectbox("🎯 Résultat du spin :",SEGMENTS)
with col2:
    mult_input = st.text_input("💥 Multiplicateur réel bonus (ex: x25 ou 25):","1")
    multiplicateur = float(mult_input.lower().replace('x','')) if mult_input else 1
with col3:
    if st.button("🎰 Enregistrer spin live"):
        mises_for_spin = st.session_state.last_suggestion_mises or {}
        strategy_name = st.session_state.last_suggestion_name or "Unknown"
        gain_brut,gain_net = calcul_gain(mises_for_spin,spin_val,multiplicateur)
        st.session_state.bankroll += gain_net
        st.session_state.live_history.append(spin_val)
        st.session_state.results_table.append({
            "Spin #": len(st.session_state.results_table)+1,
            "Stratégie": strategy_name,
            "Résultat": spin_val,
            "Multiplicateur": multiplicateur,
            "Mises $": mises_for_spin,
            "Gain Brut": round(gain_brut,2),
            "Gain Net": round(gain_net,2),
            "Bankroll": round(st.session_state.bankroll,2)
        })
        bet_segments = [s for s,v in mises_for_spin.items() if v>0]
        st.session_state.miss_streak += 0 if spin_val in bet_segments else 1
        if strategy_name=="Martingale 1":
            st.session_state.martingale_1_loss_streak = 0 if gain_net>0 else st.session_state.martingale_1_loss_streak+1
        # Prochaine stratégie
        full_history = st.session_state.history+st.session_state.live_history
        next_name,next_mises=choose_strategy_intelligent(full_history,st.session_state.bankroll,multiplicateur)
        st.session_state.last_suggestion_name=next_name
        st.session_state.last_suggestion_mises=next_mises

# -----------------------------------
# Simulation stratégie choisie
# -----------------------------------
st.subheader("🧠 Simulation sur historique + live spins")
strategy_option = st.selectbox("Choisir stratégie à simuler",["Martingale 1","God Mode","God Mode + Bonus","1 + Bonus","Only Numbers","All but 1"])
if st.button("🔁 Appliquer simulation"):
    sim_bankroll = st.session_state.initial_bankroll
    simulated_results = []
    full_hist = st.session_state.history + st.session_state.live_history
    for i, spin_result in enumerate(full_hist, start=1):
        if strategy_option=="Martingale 1":
            _,mises = strategy_martingale_1(sim_bankroll,0)
        elif strategy_option=="God Mode":
            _,mises = strategy_god_mode(sim_bankroll)
        elif strategy_option=="God Mode + Bonus":
            _,mises = strategy_god_mode_bonus(sim_bankroll)
        elif strategy_option=="1 + Bonus":
            _,mises = strategy_1_bonus(sim_bankroll)
        elif strategy_option=="Only Numbers":
            _,mises = strategy_only_numbers(sim_bankroll)
        elif strategy_option=="All but 1":
            _,mises = strategy_all_but_1(sim_bankroll)
        gain_brut,gain_net = calcul_gain(mises,spin_result,1)
        sim_bankroll += gain_net
        simulated_results.append({"Spin #":i,"Résultat":spin_result,"Mises":mises,"Gain Net":gain_net,"Bankroll":sim_bankroll})
    df_sim = pd.DataFrame(simulated_results)
    st.dataframe(df_sim,use_container_width=True)
    # Graphique bankroll simulation
    fig, ax = plt.subplots()
    ax.plot(df_sim["Spin #"],df_sim["Bankroll"],marker='o',label='Bankroll Simulation')
    ax.axhline(y=st.session_state.initial_bankroll,color='gray',linestyle='--',label='Bankroll initiale')
    ax.set_xlabel("Spin #")
    ax.set_ylabel("Bankroll ($)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
