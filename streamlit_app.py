# streamlit_crazy_time_full.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# -----------------------------------
# 🔧 CONFIG INITIALE
# -----------------------------------
st.set_page_config(page_title="🎰 Crazy Time Tracker + ML + RTP", layout="wide")

SEGMENTS = ['1','2','5','10','Coin Flip','Cash Hunt','Pachinko','Crazy Time','Top Slot']
VAL_SEG = {'1':1,'2':2,'5':5,'10':10,'Coin Flip':1,'Cash Hunt':1,'Pachinko':1,'Crazy Time':1,'Top Slot':1}
THEO_COUNTS = {'1':21,'2':13,'5':7,'10':4,'Coin Flip':4,'Cash Hunt':2,'Pachinko':2,'Crazy Time':1,'Top Slot':1}
THEO_TOTAL = sum(THEO_COUNTS.values())

# -----------------------------------
# ⚙️ INIT session_state
# -----------------------------------
for key, default in [
    ("bankroll",150.0),
    ("initial_bankroll",150.0),
    ("live_history",[]),
    ("history",[]),
    ("results_table",[]),
    ("martingale_1_loss_streak",0),
    ("miss_streak",0),
    ("last_suggestion_name",None),
    ("last_suggestion_mises",{}),
    ("bonus_multiplier_assumption",10),
    ("mult_for_ev",1),
    ("show_history_table", True),
    ("ml_window", 10),
    ("rtp_weight", 0.5),
    ("ml_model", None)
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
    return "God Mode", {'2':3.0,'5':2.0,'10':1.0,'Top Slot':1.0}

def strategy_god_mode_bonus(bankroll):
    return "God Mode + Bonus", {'2':3.0,'5':2.0,'10':1.0,
                                'Coin Flip':1.0,'Cash Hunt':1.0,
                                'Pachinko':1.0,'Crazy Time':1.0,'Top Slot':1.0}

def strategy_1_bonus(bankroll):
    return "1 + Bonus", {'1':4.0,'Coin Flip':1.0,'Cash Hunt':1.0,'Pachinko':1.0,'Crazy Time':1.0,'Top Slot':1.0}

def strategy_only_numbers(bankroll):
    return "Only Numbers", {'1':3.0,'2':2.0,'5':1.0,'10':1.0,'Top Slot':1.0}

def strategy_all_but_1(bankroll):
    return "All but 1", {'2':2.0,'5':2.0,'10':2.0,'Coin Flip':1.0,'Cash Hunt':1.0,'Pachinko':1.0,'Crazy Time':1.0,'Top Slot':1.0}

STRATEGY_LIST = [strategy_martingale_1,strategy_god_mode,strategy_god_mode_bonus,
                 strategy_1_bonus,strategy_only_numbers,strategy_all_but_1]

# -----------------------------------
# 🧠 UTILITAIRES EV / PROBABILITÉS / RTP
# -----------------------------------
def theo_prob(segment):
    return THEO_COUNTS.get(segment,0)/THEO_TOTAL

def hist_prob(full_history, segment, window=300):
    if not full_history: return 0.0
    hist = full_history[-window:]
    return hist.count(segment)/len(hist)

def combined_prob(full_history, segment, window=300):
    p_hist = hist_prob(full_history, segment, window)
    p_theo = theo_prob(segment)
    return st.session_state.rtp_weight*p_theo + (1-st.session_state.rtp_weight)*p_hist

def expected_value_for_strategy(mises, full_history, multiplicateur, bankroll):
    mise_totale = sum(mises.values()) if mises else 0.0
    ev = 0.0
    for seg in SEGMENTS:
        p = combined_prob(full_history, seg)
        if seg in mises and mises[seg]>0:
            seg_val = VAL_SEG.get(seg, st.session_state.bonus_multiplier_assumption)
            payout = mises[seg]*(seg_val*multiplicateur)+mises[seg]
            net_if_hit = payout-mise_totale
        else:
            net_if_hit = -mise_totale
        ev += p*net_if_hit
    return ev

# -----------------------------------
# 🧠 ML Prediction
# -----------------------------------
def ml_predict(full_history):
    if len(full_history) < st.session_state.ml_window:
        return None
    X = []
    y = []
    for i in range(len(full_history)-st.session_state.ml_window):
        X.append([SEGMENTS.index(s) for s in full_history[i:i+st.session_state.ml_window]])
        y.append(SEGMENTS.index(full_history[i+st.session_state.ml_window]))
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X,y)
    st.session_state.ml_model = clf
    last_window = [SEGMENTS.index(s) for s in full_history[-st.session_state.ml_window:]]
    pred_idx = clf.predict([last_window])[0]
    return SEGMENTS[pred_idx]

# -----------------------------------
# 🧠 CHOIX STRATÉGIE INTELLIGENTE
# -----------------------------------
def choose_strategy_intelligent(full_history, bankroll, multiplicateur):
    ml_seg = ml_predict(full_history)
    # Martingale si perte récente
    if st.session_state.martingale_1_loss_streak>0:
        return strategy_martingale_1(bankroll, st.session_state.martingale_1_loss_streak)
    if st.session_state.miss_streak>=3:
        return strategy_martingale_1(bankroll, 0)
    candidates=[]
    for builder in STRATEGY_LIST:
        name, mises = builder(bankroll)
        if ml_seg and ml_seg in mises:
            mises[ml_seg] *= 1.5
        ev = expected_value_for_strategy(mises, full_history, multiplicateur, bankroll)
        candidates.append((name, mises, ev))
    best = max(candidates, key=lambda x:x[2])
    return best[0], best[1]

# -----------------------------------
# 💰 CALCUL GAIN
# -----------------------------------
def calcul_gain(mises, spin_result, multiplicateur):
    if not mises: return 0.0,0.0
    mise_totale = sum(mises.values())
    gain_brut = 0.0
    if spin_result in mises:
        seg_val = VAL_SEG.get(spin_result, st.session_state.bonus_multiplier_assumption)
        gain_brut = (mises[spin_result]*(seg_val*multiplicateur))+mises[spin_result]
    gain_net = gain_brut - mise_totale
    return float(gain_brut), float(gain_net)

# -----------------------------------
# 🧾 AFFICHAGE STRATÉGIE
# -----------------------------------
def display_next_suggestion():
    st.subheader("🎯 Prochaine stratégie suggérée")
    if st.session_state.last_suggestion_name and st.session_state.last_suggestion_mises:
        st.write(f"**Stratégie :** {st.session_state.last_suggestion_name}")
        st.table(pd.DataFrame.from_dict(st.session_state.last_suggestion_mises, orient='index', columns=['Mise $']))
    else:
        st.write("Aucune stratégie suggérée pour l’instant.")

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
            if cols[c].button(seg, key=f"hist_{seg}_{idx}"):
                st.session_state.history.append(seg)
            idx += 1
segment_buttons_grid(SEGMENTS[:-1])  # Top Slot séparé

col_a, col_b, col_c = st.columns([1,1,1])
with col_a:
    if st.button("↩ Supprimer dernier spin historique") and st.session_state.history:
        st.session_state.history.pop()
with col_b:
    if st.button("🔄 Réinitialiser historique manuel"):
        st.session_state.history=[]
with col_c:
    if st.button("🏁 Fin historique et commencer"):
        full_history = st.session_state.history + st.session_state.live_history
        next_name, next_mises = choose_strategy_intelligent(full_history, st.session_state.bankroll, st.session_state.mult_for_ev)
        st.session_state.last_suggestion_name = next_name
        st.session_state.last_suggestion_mises = next_mises
        display_next_suggestion()

# -----------------------------------
# Sidebar paramètres
# -----------------------------------
st.sidebar.header("Paramètres")
st.sidebar.number_input("Bankroll initial", value=st.session_state.initial_bankroll, key="initial_bankroll")
st.sidebar.number_input("Hypothèse multiplicateur bonus", value=st.session_state.bonus_multiplier_assumption, key="bonus_multiplier_assumption")
st.sidebar.slider("Seuil critique bankroll %", min_value=0, max_value=100, value=10, key="bankroll_critical_threshold")
st.sidebar.number_input("Multiplicateur manuel (pour EV)", min_value=1, max_value=200, value=st.session_state.mult_for_ev, key="mult_for_ev")
st.sidebar.number_input("Taille fenêtre ML", min_value=1, max_value=50, value=st.session_state.ml_window, key="ml_window")
st.sidebar.slider("Pondération RTP (%)", 0,100, int(st.session_state.rtp_weight*100), key="rtp_weight")
st.session_state.rtp_weight /= 100.0  # convert percent to fraction
st.sidebar.checkbox("Afficher tableau historique", value=st.session_state.show_history_table, key="show_history_table")

# -----------------------------------
# 🧮 SPINS LIVE
# -----------------------------------
st.title("🎡 Crazy Time Live Tracker")
col1, col2 = st.columns(2)
with col1:
    spin_val = st.selectbox("🎯 Résultat du spin :", SEGMENTS)
    mult_input = st.text_input("💥 Multiplicateur actuel (ex: x25 ou 25) :", "1")
    multiplicateur = float(mult_input.lower().replace('x','')) if mult_input else 1
with col2:
    if st.button("🎰 Enregistrer le spin live"):
        mises_for_spin = st.session_state.last_suggestion_mises or {}
        strategy_name = st.session_state.last_suggestion_name or "Unknown"
        gain_brut, gain_net = calcul_gain(mises_for_spin, spin_val, multiplicateur)
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
        # Update streaks
        bet_segments = [s for s,v in mises_for_spin.items() if v>0]
        st.session_state.miss_streak += 0 if spin_val in bet_segments else 1
        if strategy_name=="Martingale 1":
            st.session_state.martingale_1_loss_streak = 0 if gain_net>0 else st.session_state.martingale_1_loss_streak+1
        # Prochaine stratégie
        full_history = st.session_state.history + st.session_state.live_history
        next_name, next_mises = choose_strategy_intelligent(full_history, st.session_state.bankroll, st.session_state.mult_for_ev)
        st.session_state.last_suggestion_name = next_name
        st.session_state.last_suggestion_mises = next_mises
        display_next_suggestion()

# -----------------------------------
# 📈 Historique + Graphique
# -----------------------------------
if st.session_state.show_history_table and st.session_state.results_table:
    st.subheader("📈 Historique des Spins Live")
    df_results = pd.DataFrame(st.session_state.results_table)
    st.dataframe(df_results,use_container_width=True)

    st.subheader("💹 Évolution de la Bankroll")
    fig, ax = plt.subplots()
    ax.plot(df_results["Spin #"], df_results["Bankroll"], marker='o', label='Bankroll')
    ax.axhline(y=st.session_state.initial_bankroll, color='gray', linestyle='--', label='Bankroll initiale')
    ax.set_xlabel("Spin #")
    ax.set_ylabel("Bankroll ($)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# -----------------------------------
# ⚡ Simulation batch
# -----------------------------------
st.subheader("⚡ Simulation d'une stratégie sur tout l'historique")
strategy_choice = st.selectbox("Choisir une stratégie pour simuler :", [s.__name__ for s in STRATEGY_LIST])
if st.button("▶️ Lancer simulation"):
    temp_bankroll = st.session_state.initial_bankroll
    temp_results=[]
    full_hist = st.session_state.history + st.session_state.live_history
    for i, spin in enumerate(full_hist):
        # Appliquer stratégie choisie
        strat_func = next(s for s in STRATEGY_LIST if s.__name__==strategy_choice)
        name, mises = strat_func(temp_bankroll)
        gain_brut, gain_net = calcul_gain(mises, spin, 1)
        temp_bankroll += gain_net
        temp_results.append({"Spin #":i+1,"Stratégie":name,"Résultat":spin,"Mises $":mises,"Gain Net":gain_net,"Bankroll":temp_bankroll})
    df_sim = pd.DataFrame(temp_results)
    st.dataframe(df_sim,use_container_width=True)
    fig2, ax2 = plt.subplots()
    ax2.plot(df_sim["Spin #"], df_sim["Bankroll"], marker='o')
    ax2.axhline(y=st.session_state.initial_bankroll,color='gray',linestyle='--')
    ax2.set_xlabel("Spin #")
    ax2.set_ylabel("Bankroll simulée")
    ax2.grid(True)
    st.pyplot(fig2)
