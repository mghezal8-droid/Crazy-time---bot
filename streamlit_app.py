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
    st.session_state.mult_real_manual = 1
if 'martingale_1_loss_streak' not in st.session_state:
    st.session_state.martingale_1_loss_streak = 0

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
    """Calcule gain brut, gain net, mise totale et bankroll après spin"""
    mult_table = {seg:1 for seg in segments}  # x1 par défaut
    if mult_manual is not None:
        for k in mult_table.keys():
            mult_table[k] = mult_manual

    # Appliquer Martingale 1 sur le segment '1'
    mises_utilisees = mises_utilisees.copy()
    if '1' in mises_utilisees:
        mises_utilisees['1'] = st.session_state.base_unit * (2 ** st.session_state.martingale_1_loss_streak)

    mise_total = sum(mises_utilisees.values())
    gain_brut = 0.0
    mult_applique = 0
    if spin_result in mises_utilisees and mises_utilisees[spin_result] > 0:
        mult_applique = mult_table[spin_result]
        gain_brut = mises_utilisees[spin_result] * mult_applique
    gain_net = gain_brut - mise_total + mises_utilisees.get(spin_result,0.0)
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

    # Martingale 1 uniquement
    strategies["Martingale_1"] = {k:(unit if k=='1' else 0.0) for k in segments}

    # God Mode
    strategies["God Mode"] = {'1':0.0,'2':round(3*scale,2),'5':round(2*scale,2),'10':round(1*scale,2),
                              'Cash Hunt':0.0,'Pachinko':0.0,'Coin Flip':0.0,'Crazy Time':0.0}
    # God Mode + Bonus
    strategies["God Mode + Bonus"] = {'1':0.0,'2':round(3*scale,2),'5':round(2*scale,2),'10':round(1*scale,2),
                                     'Cash Hunt':round(1*scale,2),'Pachinko':round(1*scale,2),
                                     'Coin Flip':round(1*scale,2),'Crazy Time':round(1*scale,2)}
    # 1 + Bonus
    strategies["1 + Bonus"] = {'1':round(4*scale,2),'2':0.0,'5':0.0,'10':0.0,
                               'Cash Hunt':round(1*scale,2),'Pachinko':round(1*scale,2),
                               'Coin Flip':round(1*scale,2),'Crazy Time':round(1*scale,2)}

    # Toujours sélectionner UNE seule stratégie
    best_name = random.choice(list(strategies.keys()))
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
# Champ multiplicateur manuel
# -------------------------------
st.subheader("⚡ Multiplicateur top slot manuel (x1 par défaut)")
mult_manual_input = st.number_input(
    "x (manuel)", min_value=1, max_value=200,
    value=st.session_state.mult_real_manual, step=1
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
        st.write("Pas encore de suggestion. Appuie sur 'Fin historique' ou enregistre un spin live.")
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
            mises_for_spin,
            st.session_state.bankroll,
            st.session_state.mult_real_manual
        )

        st.session_state.history.append(spin_val)
        st.session_state.live_history.append(spin_val)
        st.session_state.last_gain = float(gain_net)
        st.session_state.bankroll = float(new_bankroll)
        st.session_state.results_table.append({
            "Spin #": len(st.session_state.results_table) + 1,
            "Résultat": spin_val,
            "Stratégie utilisée": st.session_state.last_suggestion_name,
            "Mises $": {k: round(v,2) for k,v in mises_for_spin.items()},
            "Mise Totale": round(mise_total,2),
            "Gain Brut": round(gain_brut,2),
            "Gain Net": round(gain_net,2),
            "Multiplicateur appliqué": mult_applique,
            "Bankroll": round(new_bankroll,2)
        })

        # Mettre à jour la Martingale 1
        if spin_val == '1' and gain_net > 0:
            st.session_state.martingale_1_loss_streak = 0
        else:
            st.session_state.martingale_1_loss_streak += 1

        # Recalculer suggestion
        next_name, next_mises = choose_strategy_intelligent(st.session_state.history, st.session_state.bankroll)
        st.session_state.last_suggestion_name = next_name
        st.session_state.last_suggestion_mises = next_mises
        display_next_suggestion()

        st.success(f"Spin enregistré : {spin_val} x{mult_applique} — Gain net: {round(gain_net,2)} — Bankroll: {round(new_bankroll,2)}")

with live_col2:
    if st.button("Supprimer dernier live spin"):
        if st.session_state.live_history:
            st.session_state.live_history.pop()
        if st.session_state.results_table:
            st.session_state.results_table.pop()
        if st.session_state.history:
            st.session_state.history.pop()
        if st.session_state.martingale_1_loss_streak > 0:
            st.session_state.martingale_1_loss_streak -= 1

        st.warning("Dernier live spin supprimé.")
        # Recalculer suggestion
        next_name, next_mises = choose_strategy_intelligent(st.session_state.history, st.session_state.bankroll)
        st.session_state.last_suggestion_name = next_name
        st.session_state.last_suggestion_mises = next_mises
        display_next_suggestion()

# -------------------------------
# Tableau live + graphique bankroll
# -------------------------------
st.subheader("📈 Historique des Spins Live")
if st.session_state.results_table:
    df_results = pd.DataFrame(st.session_state.results_table)
    st.dataframe(df_results, use_container_width=True)

    st.subheader("📊 Évolution Bankroll")
    fig, ax = plt.subplots()
    ax.plot(df_results["Spin #"], df_results["Bankroll"], marker='o', label='Bankroll')
    ax.axhline(y=st.session_state.initial_bankroll, color='gray', linestyle='--', label='Bankroll initiale')
    ax.set_xlabel("Spin #")
    ax.set_ylabel("Bankroll ($)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
else:
    st.write("Aucun spin live enregistré.")
