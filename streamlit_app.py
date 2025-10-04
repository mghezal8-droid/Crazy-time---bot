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
if 'mult_real' not in st.session_state:
    st.session_state.mult_real = 1

segments = ['1','2','5','10','Cash Hunt','Pachinko','Coin Flip','Crazy Time']

# -------------------------------
# Barre latérale
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

# Tableau historique manuel
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

def process_spin_real(spin_result, mises_utilisees, bankroll, mult_real):
    mult_table = {'1':2,'2':3,'5':6,'10':11,
                  'Cash Hunt':mult_real,
                  'Pachinko':mult_real,
                  'Coin Flip':mult_real,
                  'Crazy Time':mult_real}
    mise_total = sum(mises_utilisees.values())
    gain = 0.0
    if spin_result in mises_utilisees and mises_utilisees[spin_result]>0:
        gain = mises_utilisees[spin_result]*mult_table[spin_result]
    gain_net = gain - (mise_total - mises_utilisees.get(spin_result,0.0))
    new_bankroll = float(bankroll) + float(gain_net)
    return float(gain_net), float(mise_total), float(new_bankroll)

# -------------------------------
# Stratégie intelligente basée sur probabilités
# -------------------------------
def choose_strategy_intelligent(history, bankroll):
    if bankroll <= critical_threshold_value:
        return "No-Bet", {k:0.0 for k in segments}

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

    recent_history = history[-15:] if len(history)>=15 else history
    if not any(seg in recent_history for seg in ["Cash Hunt","Pachinko","Coin Flip","Crazy Time"]):
        bonus_strats = {name:strat for name,strat in strategies.items() if "Bonus" in name}
        if bonus_strats:
            strategies = bonus_strats

    best_name = random.choice(list(strategies.keys()))
    best_mises = strategies[best_name]

    return best_name, best_mises

# -------------------------------
# Suggestion stratégie intelligente
# -------------------------------
st.subheader("📊 Suggestion stratégie intelligente")
if st.session_state.history:
    strat_name, strat_mises = choose_strategy_intelligent(st.session_state.history, st.session_state.bankroll)
    st.write("Stratégie suggérée:", strat_name)
    st.write("Mises $:", {k:round(v,2) for k,v in strat_mises.items()})
else:
    st.write("Entrez l'historique manuel et appuyez sur 'Fin historique et commencer' pour générer une suggestion.")

# -------------------------------
# Mode Live
# -------------------------------
st.header("Spin Live")
spin_val = st.selectbox("Spin Sorti", segments)

# ✅ Multiplicateurs compacts (boutons rapides)
st.markdown("**Top Slot Multiplicateur :**")
cols_mult = st.columns(6)
preset_mults = [2,5,10,25,50,100]
for i, m in enumerate(preset_mults):
    if cols_mult[i].button(f"x{m}"):
        st.session_state.mult_real = m

# Champ manuel si autre valeur
mult_input = st.number_input("Ou entrer manuellement",1,200,value=st.session_state.mult_real,step=1)
st.session_state.mult_real = mult_input

mult_real = st.session_state.mult_real

live_col1,live_col2 = st.columns([1,1])
with live_col1:
    if st.button("Enregistrer Spin"):
        # ✅ On enregistre d'abord le spin
        st.session_state.history.append(spin_val)
        st.session_state.live_history.append(spin_val)

        # ✅ Puis on choisit la stratégie (plus de décalage)
        strat_name, strat_mises = choose_strategy_intelligent(st.session_state.history, st.session_state.bankroll)
        if strat_name=="No-Bet":
            strat_mises = {k:0.0 for k in segments}

        gain_net,mise_total,new_bankroll = process_spin_real(spin_val,strat_mises,st.session_state.bankroll,mult_real)

        st.session_state.last_gain = float(gain_net)
        st.session_state.bankroll = float(new_bankroll)
        st.session_state.results_table.append({
            "Spin #": len(st.session_state.results_table)+1,
            "Résultat": spin_val,
            "Stratégie": strat_name,
            "Mises $": {k:round(v,2) for k,v in strat_mises.items()},
            "Mise Totale": round(mise_total,2),
            "Gain Net": round(gain_net,2),
            "Bankroll": round(new_bankroll,2),
            "Multiplicateur": mult_real
        })
        st.success(f"Spin: {spin_val} x{mult_real} — Gain net: {round(gain_net,2)} — Bankroll: {round(new_bankroll,2)}")

with live_col2:
    if st.button("Supprimer dernier live spin"):
        if st.session_state.live_history:
            st.session_state.live_history.pop()
        if st.session_state.results_table:
            st.session_state.results_table.pop()
        if st.session_state.history:
            st.session_state.history.pop()
        st.warning("Dernier live spin supprimé.")

# -------------------------------
# Tableau live + graphique bankroll
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
