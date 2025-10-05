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

# segments
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
    mult_table = {seg:1 for seg in segments}
    if mult_manual is not None:
        for seg in mult_table:
            mult_table[seg] = int(mult_manual)

    mise_total = float(sum(mises_utilisees.values()))
    gain_brut = 0.0
    mult_applique = 1
    if spin_result in mises_utilisees and mises_utilisees.get(spin_result,0) > 0:
        mult_applique = mult_table.get(spin_result,1)
        gain_brut = float(mises_utilisees[spin_result]) * float(mult_applique)
    gain_net = gain_brut - (mise_total - float(mises_utilisees.get(spin_result,0)))
    new_bankroll = float(bankroll) + float(gain_net)
    return float(gain_net), float(gain_brut), float(mise_total), float(new_bankroll), int(mult_applique)

# -------------------------------
# Strategy chooser
# -------------------------------
def choose_strategy_intelligent(history, bankroll):
    if float(bankroll) <= float(critical_threshold_value):
        return "No-Bet", {k:0.0 for k in segments}

    unit = adjust_unit(bankroll)
    scale = unit / st.session_state.base_unit if st.session_state.base_unit > 0 else 1.0
    probs = compute_segment_probabilities(history)

    strategies = {}
    strategies["Martingale_1"] = {k:(unit if k=='1' else 0.0) for k in segments}
    strategies["God Mode"] = {'1':0.0,'2':round(3*scale,2),'5':round(2*scale,2),'10':round(1*scale,2),
                              'Cash Hunt':0.0,'Pachinko':0.0,'Coin Flip':0.0,'Crazy Time':0.0}
    strategies["God Mode + Bonus"] = {'1':0.0,'2':round(3*scale,2),'5':round(2*scale,2),'10':round(1*scale,2),
                                     'Cash Hunt':round(1*scale,2),'Pachinko':round(1*scale,2),
                                     'Coin Flip':round(1*scale,2),'Crazy Time':round(1*scale,2)}
    strategies["1 + Bonus"] = {'1':round(4*scale,2),'2':0.0,'5':0.0,'10':0.0,
                               'Cash Hunt':round(1*scale,2),'Pachinko':round(1*scale,2),
                               'Coin Flip':round(1*scale,2),'Crazy Time':round(1*scale,2)}

    scores = {}
    for name,mise in strategies.items():
        score = sum([probs.get(seg,0.0) for seg,amt in mise.items() if amt>0])
        scores[name] = score

    max_score = max(scores.values())
    candidates = [name for name,s in scores.items() if s>=max_score-1e-12]

    pick = random.choice(candidates)
    chosen_mises = strategies[pick]
    if pick=="Martingale_1":
        m = martingale_1_mise(st.session_state.base_unit)
        chosen_mises = {k:(m if k=='1' else 0.0) for k in segments}
    return pick, chosen_mises

# -------------------------------
# UI: manual history buttons
# -------------------------------
st.title("Super Crazy Time Bot")
st.header("Historique Spins (manuel)")

def segment_buttons_grid(segments, cols_per_row=4):
    rows = (len(segments)+cols_per_row-1)//cols_per_row
    idx = 0
    for r in range(rows):
        cols = st.columns(cols_per_row)
        for c in range(cols_per_row):
            if idx >= len(segments):
                break
            seg = segments[idx]
            if cols[c].button(seg,key=f"segbtn_{seg}_{idx}"):
                st.session_state.history.append(seg)
            idx+=1

segment_buttons_grid(segments, cols_per_row=4)

col_a,col_b = st.columns([1,1])
with col_a:
    if st.button("↩ Suppr dernier historique"):
        if st.session_state.history:
            st.session_state.history.pop()
            st.success("Dernier historique supprimé.")
with col_b:
    if st.button("🏁 Fin historique"):
        st.success(f"Historique enregistré ({len(st.session_state.history)} spins).")
        if st.session_state.history:
            name,mises = choose_strategy_intelligent(st.session_state.history, st.session_state.bankroll)
            st.session_state.last_suggestion_name = name
            st.session_state.last_suggestion_mises = mises
        else:
            st.session_state.last_suggestion_name = None
            st.session_state.last_suggestion_mises = {}

st.subheader("Tableau Historique Manuel")
if st.session_state.history:
    df_manual = pd.DataFrame({"Spin n°":list(range(1,len(st.session_state.history)+1)),"Segment":st.session_state.history})
    st.dataframe(df_manual,use_container_width=True)
else:
    st.write("Aucun spin manuel enregistré.")

# -------------------------------
# Multiplicateur manuel
# -------------------------------
st.subheader("⚡ Multiplicateur manuel")
mult_manual_input = st.number_input("Multiplier (manuel) — par défaut x1", min_value=1,max_value=200,value=int(st.session_state.mult_real_manual),step=1)
st.session_state.mult_real_manual = int(mult_manual_input)

# -------------------------------
# Affichage stratégie suggérée
# -------------------------------
st.subheader("📊 Stratégie suggérée")
def display_next_suggestion():
    if st.session_state.last_suggestion_name:
        st.markdown(f"**Stratégie :** {st.session_state.last_suggestion_name}")
        st.markdown("**Mises proposées ( $ ) :**")
        st.write({k:round(v,2) for k,v in st.session_state.last_suggestion_mises.items()})
    else:
        st.write("Pas encore de suggestion.")

display_next_suggestion()

# -------------------------------
# Live spin
# -------------------------------
st.header("Spin Live")
spin_val = st.selectbox("Spin sorti", segments)
col_save,col_del = st.columns([1,1])
with col_save:
    if st.button("Enregistrer Spin"):
        mises_for_spin = st.session_state.last_suggestion_mises.copy() if st.session_state.last_suggestion_mises else {}
        if not mises_for_spin:
            tmp_name,mises_for_spin = choose_strategy_intelligent(st.session_state.history, st.session_state.bankroll)
            st.session_state.last_suggestion_name = tmp_name
            st.session_state.last_suggestion_mises = mises_for_spin

        gain_net,gain_brut,mise_total,new_bankroll,mult_applique = process_spin_real(
            spin_val,mises_for_spin,st.session_state.bankroll,st.session_state.mult_real_manual
        )

        if mises_for_spin.get('1',0.0)>0:
            if spin_val=='1' and gain_net>0:
                st.session_state.martingale_1_loss_streak=0
            else:
                st.session_state.martingale_1_loss_streak+=1

        st.session_state.history.append(spin_val)
        st.session_state.live_history.append(spin_val)
        st.session_state.last_gain=float(gain_net)
        st.session_state.bankroll=float(new_bankroll)

        st.session_state.results_table.append({
            "Spin #":len(st.session_state.results_table)+1,
            "Résultat":spin_val,
            "Stratégie utilisée":st.session_state.last_suggestion_name,
            "Mises $":{k:round(v,2) for k,v in mises_for_spin.items()},
            "Mise Totale":round(mise_total,2),
            "Gain Brut":round(gain_brut,2),
            "Gain Net":round(gain_net,2),
            "Multiplicateur appliqué":mult_applique,
            "Bankroll":round(new_bankroll,2)
        })

        next_name,next_mises = choose_strategy_intelligent(st.session_state.history,st.session_state.bankroll)
        st.session_state.last_suggestion_name = next_name
        st.session_state.last_suggestion_mises = next_mises
        display_next_suggestion()
        st.success(f"Spin enregistré : {spin_val} x{mult_applique} — Gain net: {round(gain_net,2)} — Bankroll: {round(new_bankroll,2)}")

with col_del:
    if st.button("Supprimer dernier live spin"):
        if st.session_state.live_history:
            st.session_state.live_history.pop()
        if st.session_state.results_table:
            st.session_state.results_table.pop()
        if st.session_state.history:
            st.session_state.history.pop()
        if st.session_state.martingale_1_loss_streak>0:
            st.session_state.martingale_1_loss_streak-=1
        st.warning("Dernier live spin supprimé.")
        next_name,next_mises = choose_strategy_intelligent(st.session_state.history,st.session_state.bankroll)
        st.session_state.last_suggestion_name=next_name
        st.session_state.last_suggestion_mises=next_mises
        display_next_suggestion()

# -------------------------------
# Tableau live + bankroll plot
# -------------------------------
st.subheader("📈 Historique des Spins Live")
if st.session_state.results_table:
    df_results=pd.DataFrame(st.session_state.results_table)
    st.dataframe(df_results,use_container_width=True)
    st.subheader("📊 Évolution de la Bankroll")
    fig,ax=plt.subplots()
    ax.plot(df_results["Spin #"],df_results["Bankroll"],marker='o',label='Bankroll')
    ax.axhline(y=st.session_state.initial_bankroll,color='gray',linestyle='--',label='Bankroll initiale')
    ax.set_xlabel("Spin #")
    ax.set_ylabel("Bankroll ($)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
else:
    st.write("Aucun spin live enregistré.")
