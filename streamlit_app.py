import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random

st.set_page_config(layout="wide")

# -------------------------------
# Initialisation session
# -------------------------------
if 'history' not in st.session_state:
    st.session_state.history = []               # historique manuel (entrées)
if 'live_history' not in st.session_state:
    st.session_state.live_history = []          # live spins (enregistrés)
if 'results_table' not in st.session_state:
    st.session_state.results_table = []         # tableau live (spin by spin)
if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 150.0
if 'initial_bankroll' not in st.session_state:
    st.session_state.initial_bankroll = 150.0
if 'base_unit' not in st.session_state:
    st.session_state.base_unit = 1.0
if 'last_gain' not in st.session_state:
    st.session_state.last_gain = 0.0

segments = ['1','2','5','10','Cash Hunt','Pachinko','Coin Flip','Crazy Time']

# -------------------------------
# Barre latérale
# -------------------------------
st.sidebar.header("Paramètres Crazy Time Bot")

initial_bankroll_input = st.sidebar.number_input(
    "Bankroll initial ($)",
    min_value=50.0,
    max_value=10000.0,
    value=float(st.session_state.initial_bankroll),
    step=1.0
)
if initial_bankroll_input != st.session_state.initial_bankroll:
    st.session_state.initial_bankroll = float(initial_bankroll_input)
    st.session_state.bankroll = float(initial_bankroll_input)

base_unit_input = st.sidebar.number_input(
    "Unité de base ($)",
    min_value=0.2,
    max_value=100.0,
    value=float(st.session_state.base_unit),
    step=0.1
)
if base_unit_input != st.session_state.base_unit:
    st.session_state.base_unit = float(base_unit_input)

bonus_multiplier_assumption = st.sidebar.number_input(
    "Hypothèse multiplicateur bonus",
    min_value=1,
    max_value=100,
    value=10,
    step=1
)

critical_threshold_pct = st.sidebar.slider(
    "Seuil critique bankroll (%)",
    min_value=1, max_value=100, value=25, step=1
)
critical_threshold_value = float(st.session_state.initial_bankroll) * (critical_threshold_pct/100)

# -------------------------------
# Entrée historique manuel
# -------------------------------
st.header("Historique Spins (manuel)")
cols = st.columns(len(segments))
for i, seg in enumerate(segments):
    if cols[i].button(seg):
        st.session_state.history.append(seg)

hist_col1, hist_col2 = st.columns([1,1])
with hist_col1:
    if st.button("Supprimer dernier historique"):
        if st.session_state.history:
            st.session_state.history.pop()
            st.success("Dernier historique supprimé.")
with hist_col2:
    if st.button("Fin historique et commencer"):
        st.success(f"Historique enregistré ({len(st.session_state.history)} spins). Le bot est prêt à suggérer.")

# Affichage tableau historique manuel
st.subheader("Tableau Historique Manuel (sans simulation)")
if st.session_state.history:
    df_manual = pd.DataFrame({
        "Spin n°": list(range(1, len(st.session_state.history)+1)),
        "Segment": st.session_state.history
    })
    st.dataframe(df_manual, use_container_width=True)
else:
    st.write("Aucun spin manuel enregistré.")

# -------------------------------
# Fonctions probabilités & stratégies
# -------------------------------
def compute_segment_probabilities(history):
    segment_count = {'1':1,'2':2,'5':2,'10':1,'Cash Hunt':1,'Pachinko':1,'Coin Flip':1,'Crazy Time':1}
    total_segments = sum(segment_count.values())
    base_prob = {k: v/total_segments for k,v in segment_count.items()}
    hist_weight = {k: (history.count(k)/len(history) if history else 0) for k in segment_count.keys()}
    prob = {k: 0.5*base_prob[k] + 0.5*hist_weight[k] for k in segment_count.keys()}
    return prob

def adjust_unit(bankroll):
    if bankroll >= 2.5 * st.session_state.initial_bankroll:
        return st.session_state.base_unit * 2.0
    elif bankroll <= 0.5 * st.session_state.initial_bankroll:
        return st.session_state.base_unit * 0.5
    return st.session_state.base_unit

def estimate_ev(mises, probs):
    mult_table = {'1':2,'2':3,'5':6,'10':11,'Cash Hunt':bonus_multiplier_assumption,
                  'Pachinko':bonus_multiplier_assumption,'Coin Flip':bonus_multiplier_assumption,
                  'Crazy Time':bonus_multiplier_assumption}
    mise_total = sum(mises.values())
    ev = 0.0
    for seg in segments:
        bet = mises.get(seg,0)
        if bet <= 0:
            continue
        payout = mult_table[seg]
        gain_net_if_hit = bet*payout - (mise_total - bet)
        ev += probs[seg]*gain_net_if_hit
    return ev

def choose_strategy_intelligent(history, bankroll):
    if float(bankroll) <= float(critical_threshold_value):
        return "No-Bet", {k:0.0 for k in segments}, 0.0
    unit = adjust_unit(bankroll)
    scale = unit / st.session_state.base_unit if st.session_state.base_unit>0 else 1.0
    probs = compute_segment_probabilities(history)

    strategies = {}
    sorted_probs = sorted(probs.items(), key=lambda x:x[1], reverse=True)
    target_segment = sorted_probs[0][0]
    strategies[f"Martingale_{target_segment}"] = {k:(unit if k==target_segment else 0.0) for k in segments}

    strategies["God Mode"] = {'1':0.0,'2':round(3*scale,2),'5':round(2*scale,2),'10':round(1*scale,2),
                              'Cash Hunt':0.0,'Pachinko':0.0,'Coin Flip':0.0,'Crazy Time':0.0}
    strategies["God Mode + Bonus"] = {'1':0.0,'2':round(3*scale,2),'5':round(2*scale,2),'10':round(1*scale,2),
                                     'Cash Hunt':round(1*scale,2),'Pachinko':round(1*scale,2),
                                     'Coin Flip':round(1*scale,2),'Crazy Time':round(1*scale,2)}
    strategies["1 + Bonus"] = {'1':round(4*scale,2),'2':0.0,'5':0.0,'10':0.0,
                               'Cash Hunt':round(1*scale,2),'Pachinko':round(1*scale,2),
                               'Coin Flip':round(1*scale,2),'Crazy Time':round(1*scale,2)}

    evs = {name: estimate_ev(mises,probs) for name,mises in strategies.items()}
    max_ev = max(evs.values())

    # ✅ VARIATION : garder plusieurs choix proches du max
    candidates = [name for name,ev in evs.items() if ev >= max_ev*0.95]

    # ✅ si aucun bonus sorti récemment → favoriser les stratégies avec bonus
    if len(history) >= 15 and not any(seg in history[-15:] for seg in ["Cash Hunt","Pachinko","Coin Flip","Crazy Time"]):
        bonus_strats = [name for name in candidates if "Bonus" in name]
        if bonus_strats:
            candidates = bonus_strats

    best_name = random.choice(candidates)
    best_mises = strategies[best_name]
    best_ev = evs[best_name]

    if best_ev <= 0:
        return "No-Bet",{k:0.0 for k in segments},best_ev
    return best_name,best_mises,best_ev

def process_spin(spin_result,mises_utilisees,bankroll):
    mult_table = {'1':2,'2':3,'5':6,'10':11,'Cash Hunt':bonus_multiplier_assumption,
                  'Pachinko':bonus_multiplier_assumption,'Coin Flip':bonus_multiplier_assumption,
                  'Crazy Time':bonus_multiplier_assumption}
    mise_total = sum(mises_utilisees.values())
    gain = 0.0
    if spin_result in mises_utilisees and mises_utilisees[spin_result]>0:
        gain = mises_utilisees[spin_result]*mult_table[spin_result]
    gain_net = gain - (mise_total - mises_utilisees.get(spin_result,0.0))
    new_bankroll = float(bankroll)+float(gain_net)
    return float(gain_net), float(mise_total), float(new_bankroll)

# -------------------------------
# Suggestion actuelle
# -------------------------------
st.subheader("📊 Suggestion stratégie (basée sur l'historique)")
if st.session_state.history:
    strat_name, strat_mises, strat_ev = choose_strategy_intelligent(st.session_state.history, st.session_state.bankroll)
    st.write("Stratégie suggérée:", strat_name)
    st.write("Mises $:", {k:round(v,2) for k,v in strat_mises.items()})
    st.write("EV estimé par spin:", round(strat_ev,3))
else:
    st.write("Entrez l'historique manuel et appuyez sur 'Fin historique et commencer' pour générer une suggestion.")

# -------------------------------
# Mode Live
# -------------------------------
st.header("Spin Live")
spin_val = st.selectbox("Spin Sorti", segments)
multiplier_val = st.number_input("Multiplicateur réel (pour bonus)",1,200,value=1,step=1)

if st.button("Enregistrer Spin"):
    strat_name, strat_mises, strat_ev = choose_strategy_intelligent(st.session_state.history, st.session_state.bankroll)
    if strat_name=="No-Bet":
        strat_mises = {k:0.0 for k in segments}
    gain_net,mise_total,new_bankroll = process_spin(spin_val,strat_mises,st.session_state.bankroll)
    st.session_state.history.append(spin_val)
    st.session_state.live_history.append(spin_val)
    st.session_state.last_gain = float(gain_net)
    st.session_state.bankroll = float(new_bankroll)
    st.session_state.results_table.append({
        "Spin #": len(st.session_state.results_table)+1,
        "Résultat": spin_val,
        "Stratégie": strat_name,
        "Mises $": {k:round(v,2) for k,v in strat_mises.items()},
        "Mise Totale": round(mise_total,2),
        "Gain Net": round(gain_net,2),
        "Bankroll": round(new_bankroll,2)
    })
    st.success(f"Spin: {spin_val} — Gain net: {round(gain_net,2)} — Bankroll: {round(new_bankroll,2)}")

# -------------------------------
# Tableau live
# -------------------------------
st.subheader("📈 Historique des Spins Live")
if st.session_state.results_table:
    df_results = pd.DataFrame(st.session_state.results_table)
    st.dataframe(df_results,use_container_width=True)
    st.subheader("📊 Évolution Bankroll")
    fig,ax = plt.subplots()
    ax.plot(df_results["Spin #"],df_results["Bankroll"],marker='o',label='Bankroll')
    ax.axhline(y=st.session_state.initial_bankroll,color='gray',linestyle='--',label='Bankroll initiale')
    ax.set_xlabel("Spin #")
    ax.set_ylabel("Bankroll ($)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
else:
    st.write("Aucun spin live enregistré.")

# -------------------------------
# Tester une stratégie
# -------------------------------
st.subheader("⚡ Tester une stratégie manuellement")
strategy_choice = st.selectbox(
    "Choisir une stratégie",
    ["Martingale_1","Martingale_2","Martingale_5","Martingale_10",
     "God Mode","God Mode + Bonus","1 + Bonus","No-Bet"]
)

if st.button("Tester Stratégie"):
    bankroll_test = st.session_state.initial_bankroll
    test_results = []
    history_test = st.session_state.history.copy()

    for i,spin in enumerate(history_test, start=1):
        if strategy_choice=="No-Bet":
            mises = {k:0.0 for k in segments}
        elif "Martingale" in strategy_choice:
            target = strategy_choice.split("_")[1]
            mises = {k:(st.session_state.base_unit if k==target else 0.0) for k in segments}
        elif strategy_choice=="God Mode":
            mises = {'1':0.0,'2':3,'5':2,'10':1,'Cash Hunt':0,'Pachinko':0,'Coin Flip':0,'Crazy Time':0}
        elif strategy_choice=="God Mode + Bonus":
            mises = {'1':0.0,'2':3,'5':2,'10':1,'Cash Hunt':1,'Pachinko':1,'Coin Flip':1,'Crazy Time':1}
        elif strategy_choice=="1 + Bonus":
            mises = {'1':4,'2':0,'5':0,'10':0,'Cash Hunt':1,'Pachinko':1,'Coin Flip':1,'Crazy Time':1}
        else:
            mises = {k:0.0 for k in segments}

        gain_net,mise_total,new_bankroll = process_spin(spin,mises,bankroll_test)
        bankroll_test = new_bankroll
        test_results.append({
            "Spin #": i,
            "Résultat": spin,
            "Mises": mises,
            "Mise Totale": mise_total,
            "Gain Net": gain_net,
            "Bankroll": bankroll_test
        })

    df_test = pd.DataFrame(test_results)
    st.dataframe(df_test, use_container_width=True)

    st.subheader("📊 Évolution bankroll (test stratégie)")
    fig, ax = plt.subplots()
    ax.plot(df_test["Spin #"], df_test["Bankroll"], marker='o', label='Bankroll (test)')
    ax.axhline(y=st.session_state.initial_bankroll, color='gray', linestyle='--', label='Bankroll initiale')
    ax.set_xlabel("Spin #")
    ax.set_ylabel("Bankroll ($)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
