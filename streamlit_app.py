import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Crazy Time Bot Auto-Strategy", layout="wide")

# --- Initialisation ---
if "history" not in st.session_state:
    st.session_state.history = []
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame(columns=["Spin","Résultat","Multiplicateur","Total Mise","Gain Net","Bankroll","Stratégie"])
if "bankroll" not in st.session_state:
    st.session_state.bankroll = 150
if "next_mises" not in st.session_state:
    st.session_state.next_mises = {}
if "mode_live" not in st.session_state:
    st.session_state.mode_live = False
if "last_spin_val" not in st.session_state:
    st.session_state.last_spin_val = None
if "last_gain" not in st.session_state:
    st.session_state.last_gain = 0
if "strategy_name" not in st.session_state:
    st.session_state.strategy_name = None

# --- Segments Crazy Time ---
segments_numbers = ["1","2","5","10"]
segments_bonus = ["CoinFlip","Pachinko","CashHunt","CrazyTime"]
segments = segments_numbers + segments_bonus

# --- Fonctions ---
def calc_gain_net(result,mises,mult):
    total_mise = sum(mises.values())
    if result in segments_numbers:
        n = mises[result]
        gain_net = (n * (int(result)) + n) - (total_mise - n)
    elif result in segments_bonus:
        n = mises[result]
        gain_net = (n * mult + n) - (total_mise - n)
    else:
        gain_net = -total_mise
    return gain_net,total_mise

def generate_mises_option(strategy_name,last_spin_val=None,bankroll=150):
    mises = {}
    if strategy_name=="Martingale1":
        mises = {"1":1,"2":0,"5":0,"10":0,"CoinFlip":0,"Pachinko":0,"CashHunt":0,"CrazyTime":0}
    elif strategy_name=="GodMode2_5_10":
        mises = {"1":0,"2":2,"5":1,"10":1,"CoinFlip":0,"Pachinko":0,"CashHunt":0,"CrazyTime":0}
    elif strategy_name=="GodMode2_5_10_Bonus":
        mises = {"1":0,"2":2,"5":1,"10":1,"CoinFlip":0.2,"Pachinko":0.2,"CashHunt":0.2,"CrazyTime":0.2}
    elif strategy_name=="1+Bonus":
        mises = {"1":0.5,"2":0.2,"5":0.2,"10":0.2,"CoinFlip":0.2,"Pachinko":0.2,"CashHunt":0.2,"CrazyTime":0.2}
    # Ne pas miser sur bonus sorti tour précédent
    if last_spin_val in segments_bonus:
        mises[last_spin_val]=0
    # Ajuster si total mise > bankroll
    total_mise = sum(mises.values())
    if total_mise>bankroll:
        scale=bankroll/total_mise
        mises={seg:m*scale for seg,m in mises.items()}
    return mises

def expected_profit(strat,probs):
    return sum(strat[seg]*probs.get(seg,0) for seg in strat)

def choose_strategy(history,bankroll):
    # Calcul des probabilités basiques à partir de l'historique
    counts = {seg:0 for seg in segments}
    for h in history:
        counts[h["Résultat"]] +=1
    total = max(len(history),1)
    probs = {seg: counts[seg]/total for seg in segments}
    # Calcul profit attendu pour chaque stratégie
    strategies = ["Martingale1","GodMode2_5_10","GodMode2_5_10_Bonus","1+Bonus"]
    best_profit = -np.inf
    best_strat = None
    for strat_name in strategies:
        mises = generate_mises_option(strat_name,bankroll=bankroll)
        profit = expected_profit(mises,probs)
        if profit>best_profit:
            best_profit = profit
            best_strat = strat_name
    return best_strat

def process_spin(result,mult,last_mises,last_bankroll,last_gain,strategy_name):
    mises_utilisees = last_mises.copy()
    gain_net,total_mise = calc_gain_net(result,mises_utilisees,mult)
    new_bankroll = last_bankroll + gain_net
    # Ajustement martingale
    if strategy_name=="Martingale1":
        if gain_net>=0:
            next_mises=generate_mises_option(strategy_name,bankroll=new_bankroll)
        else:
            next_mises={seg:m*2 for seg,m in last_mises.items()}
            total_next = sum(next_mises.values())
            if total_next>new_bankroll:
                scale=new_bankroll/total_next
                next_mises={seg:m*scale for seg,m in next_mises.items()}
    else:
        next_mises = generate_mises_option(strategy_name,result,bankroll=new_bankroll)
    return gain_net,total_mise,new_bankroll,strategy_name,next_mises

# --- Interface ---
st.title("🎰 Crazy Time Bot Auto-Strategy")

# Sidebar paramètres
st.sidebar.header("⚙️ Paramètres")
st.session_state.bankroll = st.sidebar.number_input("Bankroll initiale ($)",50,1000,st.session_state.bankroll)
st.sidebar.markdown("Le bot choisira automatiquement la meilleure stratégie après analyse de l'historique.")

# Sidebar historique
st.sidebar.header("📥 Ajouter spin à l'historique")
mult_input = st.sidebar.number_input("Multiplicateur Bonus / Live Spin",min_value=1,value=1,step=1)

st.sidebar.subheader("Historique Spins")
for seg in segments:
    if st.sidebar.button(f"{seg} ➕ Historique"):
        spin_num=len(st.session_state.history)+1
        st.session_state.history.append({"Spin":spin_num,"Résultat":seg,"Multiplicateur":mult_input})
        st.session_state.results_df=pd.DataFrame(st.session_state.history)

if st.sidebar.button("🗑 Supprimer dernier spin"):
    if st.session_state.history:
        st.session_state.history.pop()
        st.session_state.results_df=pd.DataFrame(st.session_state.history)

if st.sidebar.button("✅ Fin historique et commencer"):
    bankroll=st.session_state.bankroll
    last_spin_val=None
    last_gain=0
    # Choisir la meilleure stratégie
    st.session_state.strategy_name=choose_strategy(st.session_state.history,bankroll)
    strategy_name=st.session_state.strategy_name
    last_mises=generate_mises_option(strategy_name,bankroll=bankroll)
    results=[]
    for spin in st.session_state.history:
        result=spin["Résultat"]
        mult=spin["Multiplicateur"]
        mises_utilisees=last_mises.copy()
        gain_net,total_mise,new_bankroll,strategy,next_mises=process_spin(result,mult,mises_utilisees,bankroll,last_gain,strategy_name)
        results.append({
            "Spin":spin["Spin"],
            "Résultat":result,
            "Multiplicateur":mult,
            "Total Mise":total_mise,
            "Gain Net":gain_net,
            "Bankroll":new_bankroll,
            "Stratégie":strategy
        })
        last_spin_val=result
        last_gain=gain_net
        last_mises=next_mises.copy()
        bankroll=new_bankroll
        st.session_state.next_mises=next_mises
    st.session_state.results_df=pd.DataFrame(results)
    st.session_state.mode_live=True
    st.session_state.last_spin_val=last_spin_val
    st.session_state.last_gain=last_gain

# Tableau historique
st.subheader("📜 Historique des spins")
if not st.session_state.results_df.empty:
    st.dataframe(st.session_state.results_df,use_container_width=True)
    st.line_chart(st.session_state.results_df.set_index("Spin")["Bankroll"])

# Mode live
if st.session_state.mode_live:
    st.subheader("🎯 Mode Live - spin par spin")
    mult_live=st.number_input("Multiplicateur Live",min_value=1,value=1,step=1)
    
    st.subheader("Cliquer sur un segment pour live spin")
    for seg in segments:
        if st.button(f"{seg} ➡️ Live Spin"):
            last_bankroll=st.session_state.results_df["Bankroll"].iloc[-1] if not st.session_state.results_df.empty else st.session_state.bankroll
            last_spin_val=st.session_state.results_df["Résultat"].iloc[-1] if not st.session_state.results_df.empty else None
            last_gain=st.session_state.last_gain
            mises_utilisees=st.session_state.next_mises.copy()
            gain_net,total_mise,new_bankroll,strategy,next_mises=process_spin(seg,mult_live,mises_utilisees,last_bankroll,last_gain,st.session_state.strategy_name)
            new_row={
                "Spin":len(st.session_state.results_df)+1,
                "Résultat":seg,
                "Multiplicateur":mult_live,
                "Total Mise":total_mise,
                "Gain Net":gain_net,
                "Bankroll":new_bankroll,
                "Stratégie":strategy
            }
            st.session_state.results_df=pd.concat([st.session_state.results_df,pd.DataFrame([new_row])],ignore_index=True)
            st.session_state.next_mises=next_mises
            st.session_state.last_spin_val=seg
            st.session_state.last_gain=gain_net
            st.success(f"Spin ajouté : {seg} x{mult_live} | Stratégie : {strategy}")

    if st.session_state.next_mises:
        st.subheader("📌 Mise conseillée pour le prochain spin")
        mises_df=pd.DataFrame(list(st.session_state.next_mises.items()),columns=["Segment","Mise ($)"])
        st.dataframe(mises_df,use_container_width=True)
        st.info(f"Mise totale conseillée : {mises_df['Mise ($)'].sum():.2f} $")
