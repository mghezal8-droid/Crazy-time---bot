import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random

st.set_page_config(layout="wide")

# -------------------------------
# Initialisation session
# -------------------------------
if 'history' not in st.session_state:
    st.session_state.history = []
if 'live_history' not in st.session_state:
    st.session_state.live_history = []
if 'results_table' not in st.session_state:
    st.session_state.results_table = []
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
if 'mult_real_manual' not in st.session_state:
    st.session_state.mult_real_manual = None

# -------------------------------
# Segments
# -------------------------------
segments = ['1','2','5','10','Cash Hunt','Pachinko','Coin Flip','Crazy Time']

# -------------------------------
# Barre latérale (paramètres)
# -------------------------------
st.sidebar.header("Paramètres Crazy Time Bot")
initial_bankroll_input = st.sidebar.number_input(
    "Bankroll initial ($)", min_value=50.0, max_value=10000.0,
    value=float(st.session_state.initial_bankroll), step=1.0
)
if initial_bankroll_input != st.session_state.initial_bankroll:
    st.session_state.initial_bankroll = float(initial_bankroll_input)
    st.session_state.bankroll = float(initial_bankroll_input)

base_unit_input = st.sidebar.number_input(
    "Unité de base ($)", min_value=0.2, max_value=100.0,
    value=float(st.session_state.base_unit), step=0.1
)
if base_unit_input != st.session_state.base_unit:
    st.session_state.base_unit = float(base_unit_input)

critical_threshold_pct = st.sidebar.slider(
    "Seuil critique bankroll (%)", min_value=1, max_value=100,
    value=25, step=1
)
critical_threshold_value = float(st.session_state.initial_bankroll) * (critical_threshold_pct/100)

# -------------------------------
# Fonctions utilitaires
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

def process_spin_real(spin_result, mises_utilisees, bankroll, mult_manual=None):
    """Calcule le gain net, brut, total mise, bankroll après spin. 
       Multiplier standard = x1 sauf si mult_manual défini.
    """
    mult_table = {'1':1,'2':1,'5':1,'10':1,'Cash Hunt':1,'Pachinko':1,'Coin Flip':1,'Crazy Time':1}
    if mult_manual is not None:
        for k in mult_table.keys():
            mult_table[k] = mult_manual
    mise_total = sum(mises_utilisees.values())
    gain_brut = 0.0
    mult_applique = 0
    if spin_result in mises_utilisees and mises_utilisees[spin_result] > 0:
        mult_applique = mult_table[spin_result]
        gain_brut = mises_utilisees[spin_result] * mult_applique
    gain_net = gain_brut - (mise_total - mises_utilisees.get(spin_result, 0.0))
    new_bankroll = float(bankroll) + float(gain_net)
    return float(gain_net), float(gain_brut), float(mise_total), float(new_bankroll), mult_applique

def choose_strategy_intelligent(history, bankroll):
    if float(bankroll) <= float(critical_threshold_value):
        return "No-Bet", {k:0.0 for k in segments}

    unit = adjust_unit(bankroll)
    scale = unit / st.session_state.base_unit if st.session_state.base_unit > 0 else 1.0
    probs = compute_segment_probabilities(history)

    strategies = {}
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
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

    # Élimine les doubles suggestions
    max_prob = max([probs.get(seg,0) for seg in segments])
    candidates = [name for name,strat in strategies.items() if any(strat.get(seg,0)>0 and probs.get(seg,0)==max_prob for seg in segments)]
    if not candidates:
        candidates = list(strategies.keys())

    best_name = random.choice(candidates)
    best_mises = strategies[best_name]

    return best_name, best_mises

# -------------------------------
# UI boutons segments
# -------------------------------
st.header("Historique Spins (manuel)")
def segment_buttons_grid(segments, cols_per_row=4):
    rows = (len(segments) + cols_per_row - 1) // cols_per_row
    idx = 0
    for r in range(rows):
        cols = st.columns(cols_per_row)
        for c in range(cols_per_row):
            if idx >= len(segments):
                break
            seg = segments[idx]
            if cols[c].button(seg, key=f"segbtn_{seg}_{idx}"):
                st.session_state.history.append(seg)
            idx += 1

segment_buttons_grid(segments, cols_per_row=4)

act_col1, act_col2 = st.columns([1,1])
with act_col1:
    if st.button("↩ Suppr dernier", key="btn_suppr_hist"):
        if st.session_state.history:
            st.session_state.history.pop()
            st.success("Dernier historique supprimé.")
with act_col2:
    if st.button("🏁 Fin historique", key="btn_fin_hist"):
        st.success(f"Historique enregistré ({len(st.session_state.history)} spins). Le bot est prêt à suggérer.")
        if st.session_state.history:
            next_name, next_mises = choose_strategy_intelligent(st.session_state.history, st.session_state.bankroll)
            st.session_state.last_suggestion_name = next_name
            st.session_state.last_suggestion_mises = next_mises

st.subheader("Tableau Historique Manuel")
if st.session_state.history:
    df_manual = pd.DataFrame({
        "Spin n°": list(range(1, len(st.session_state.history)+1)),
        "Segment": st.session_state.history
    })
    st.dataframe(df_manual, use_container_width=True)
else:
    st.write("Aucun spin manuel enregistré.")

# -------------------------------
# Champ multiplicateur manuel uniquement
# -------------------------------
st.subheader("⚡ Multiplicateur top slot manuel (applique x1 par défaut)")
mult_manual_input = st.number_input(
    "x (manuel)", min_value=1, max_value=200,
    value=st.session_state.mult_real_manual or 1, step=1
)
st.session_state.mult_real_manual = mult_manual_input

# -------------------------------
# Affichage stratégie suggérée
# -------------------------------
st.subheader("📊 Stratégie suggérée (prochaine mise)")
def display_next_suggestion():
    if st.session_state.last_suggestion_name:
        st.markdown(f"**Stratégie :** {st.session_state.last_suggestion_name}")
        st.markdown("**Mises proposées :**")
        st.write({k: round(v,2) for k,v in st.session_state.last_suggestion_mises.items()})
    else:
        st.write("Pas encore de suggestion. Appuie sur 'Fin historique' ou enregistre un spin live pour calculer.")

display_next_suggestion()

# -------------------------------
# Mode Live (Enregistrer spin)
# -------------------------------
st.header("Spin Live")
spin_val = st.selectbox("Spin Sorti", segments)
live_col1, live_col2 = st.columns([1,1])

with live_col1:
    if st.button("Enregistrer Spin"):
        mises_for_spin = st.session_state.last_suggestion_mises.copy() if st.session_state.last_suggestion_mises else {}
        gain_net, gain_brut, mise_total, new_bankroll, mult_applique = process_spin_real(
            spin_val,
