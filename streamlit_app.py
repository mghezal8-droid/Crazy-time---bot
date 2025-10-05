# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random

st.set_page_config(layout="wide")

# -------------------------------
# Init session_state
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
if 'last_suggestion_name' not in st.session_state:
    st.session_state.last_suggestion_name = None
if 'last_suggestion_mises' not in st.session_state:
    st.session_state.last_suggestion_mises = {}
if 'martingale_1_loss_streak' not in st.session_state:
    st.session_state.martingale_1_loss_streak = 0
if 'mult_real_manual' not in st.session_state:
    st.session_state.mult_real_manual = 1

segments = ['1','2','5','10','Cash Hunt','Pachinko','Coin Flip','Crazy Time']

# -------------------------------
# Sidebar params
# -------------------------------
st.sidebar.header("Paramètres Super Crazy Time Bot")
initial_bankroll_input = st.sidebar.number_input(
    "Bankroll initial ($)", min_value=50.0, max_value=10000.0,
    value=float(st.session_state.initial_bankroll), step=1.0, format="%.2f"
)
if initial_bankroll_input != st.session_state.initial_bankroll:
    st.session_state.initial_bankroll = float(initial_bankroll_input)
    st.session_state.bankroll = float(initial_bankroll_input)

base_unit_input = st.sidebar.number_input(
    "Unité de base ($)", min_value=0.2, max_value=100.0,
    value=float(st.session_state.base_unit), step=0.1, format="%.2f"
)
if base_unit_input != st.session_state.base_unit:
    st.session_state.base_unit = float(base_unit_input)

bonus_multiplier_assumption = st.sidebar.number_input(
    "Hypothèse multiplicateur bonus (pour tests/EV)", min_value=1, max_value=1000,
    value=10, step=1
)

critical_threshold_pct = st.sidebar.slider(
    "Seuil critique bankroll (%) (relatif au bankroll initial)",
    min_value=1, max_value=100, value=25, step=1
)
critical_threshold_value = float(st.session_state.initial_bankroll) * (critical_threshold_pct/100)

# -------------------------------
# Utility functions
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

def martingale_1_mise(base_unit):
    streak = st.session_state.martingale_1_loss_streak
    cap = 10
    n = min(streak, cap)
    return base_unit * (2 ** n)

def process_spin_real(spin_result, mises_utilisees, bankroll, mult_manual):
    mise_total = float(sum(mises_utilisees.values()))
    gain_brut = 0.0
    mult_applique = 1
    if spin_result in mises_utilisees and mises_utilisees.get(spin_result,0) > 0:
        mult_applique = int(mult_manual)
        gain_brut = mises_utilisees[spin_result] * mult_applique
    gain_net = gain_brut - (mise_total - mises_utilisees.get(spin_result,0))
    new_bankroll = bankroll + gain_net
    return float(gain_net), float(gain_brut), float(mise_total), float(new_bankroll), int(mult_applique)

# -------------------------------
# Strategy chooser (single suggestion stored)
# -------------------------------
def choose_strategy_intelligent(history, bankroll):
    if bankroll <= critical_threshold_value:
        return "No-Bet", {k:0.0 for k in segments}

    unit = adjust_unit(bankroll)
    scale = unit / st.session_state.base_unit
    probs = compute_segment_probabilities(history)

    strategies = {}
    strategies["Martingale_1"] = {k:(unit if k=='1' else 0.0) for k in segments}
    strategies["God Mode"] = {'1':0.0,'2':3.0,'5':2.0,'10':1.0,'Cash Hunt':0.0,'Pachinko':0.0,'Coin Flip':0.0,'Crazy Time':0.0}
    strategies["God Mode + Bonus"] = {'1':0.0,'2':3.0,'5':2.0,'10':1.0,'Cash Hunt':1.0,'Pachinko':1.0,'Coin Flip':1.0,'Crazy Time':1.0}
    strategies["1 + Bonus"] = {'1':4.0,'2':0.0,'5':0.0,'10':0.0,'Cash Hunt':1.0,'Pachinko':1.0,'Coin Flip':1.0,'Crazy Time':1.0}

    scores = {}
    for name, mise in strategies.items():
        score = sum([probs.get(seg,0) for seg,amt in mise.items() if amt>0])
        scores[name] = score
    max_score = max(scores.values())
    candidates = [name for name,s in scores.items() if s>=max_score-1e-12]

    recent = history[-15:] if len(history)>=15 else history
    if not any(x in recent for x in ['Cash Hunt','Pachinko','Coin Flip','Crazy Time']):
        bonus_candidates = [c for c in candidates if ('Bonus' in c) or any(k in ['Cash Hunt','Pachinko','Coin Flip','Crazy Time'] and strategies[c][k]>0 for k in segments)]
        if bonus_candidates:
            candidates = bonus_candidates

    pick = random.choice(candidates)
    chosen_mises = strategies[pick]

    if pick=="Martingale_1":
        m = martingale_1_mise(st.session_state.base_unit)
        chosen_mises = {k:(m if k=='1' else 0.0) for k in segments}

    return pick, chosen_mises

# -------------------------------
# UI: manual history + buttons
# -------------------------------
st.title("Super Crazy Time Bot — (référence)")

st.header("Historique Spins (manuel) — entre les résultats ici")
def segment_buttons_grid(segments, cols_per_row=4):
    rows = (len(segments)+cols_per_row-1)//cols_per_row
    idx=0
    for r in range(rows):
        cols = st.columns(cols_per_row)
        for c in range(cols_per_row):
            if idx>=len(segments): break
            seg = segments[idx]
            if cols[c].button(seg,key=f"segbtn_{seg}_{idx}"):
                st.session_state.history.append(seg)
            idx+=1
segment_buttons_grid(segments,4)

col_a,col_b = st.columns([1,1])
with col_a:
    if st.button("↩ Suppr dernier historique"):
        if st.session_state.history: st.session_state.history.pop(); st.success("Dernier historique supprimé.")
with col_b:
    if st.button("🏁 Fin historique"):
        st.success(f"Historique enregistré ({len(st.session_state.history)} spins).")
        if st.session_state.history:
            name,mises = choose_strategy_intelligent(st.session_state.history,st.session_state.bankroll)
            st.session_state.last_suggestion_name=name
            st.session_state.last_suggestion_mises=mises
        else:
            st.session_state.last_suggestion_name=None
            st.session_state.last_suggestion_mises={}

st.subheader("Tableau Historique Manuel (sans simulation)")
if st.session_state.history:
    df_manual=pd.DataFrame({"Spin n°":list(range(1,len(st.session_state.history)+1)),"Segment":st.session_state.history})
    st.dataframe(df_manual,use_container_width=True)
else:
    st.write("Aucun spin manuel enregistré.")

# -------------------------------
# Multiplicateur manuel
# -------------------------------
st.subheader("⚡ Multiplicateur manuel (appliqué sur le segment sorti)")
mult_manual_input = st.number_input("Multiplier (manuel) — par défaut x1",min_value=1,max_value=200,value=int(st.session_state.mult_real_manual),step=1)
st.session_state.mult_real_manual = int(mult_manual_input)

# -------------------------------
# Display single suggestion
# -------------------------------
st.subheader("📊 Stratégie suggérée (prochaine mise)")
def display_next_suggestion():
    if st.session_state.last_suggestion_name:
        st.markdown(f"**Stratégie :** {st.session_state.last_suggestion_name}")
        st.markdown("**Mises proposées ( $ ) :**")
        st.write({k: round(v,2) for k,v in st.session_state.last_suggestion_mises.items()})
    else:
        st.write("Pas encore de suggestion. Appuie sur 'Fin historique' pour que le bot calcule la stratégie pour le 1er spin.")
display_next_suggestion()
