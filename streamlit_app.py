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

# stockage de la dernière suggestion (nom + mises) pour l'utiliser quand tu enregistres le spin
if 'last_suggestion_name' not in st.session_state:
    st.session_state.last_suggestion_name = None
if 'last_suggestion_mises' not in st.session_state:
    st.session_state.last_suggestion_mises = {}

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

# valeur par défaut pour preset multiplicateurs rapides
if 'mult_real_default' not in st.session_state:
    st.session_state.mult_real_default = 10

critical_threshold_pct = st.sidebar.slider(
    "Seuil critique bankroll (%)", min_value=1, max_value=100,
    value=25, step=1
)
critical_threshold_value = float(st.session_state.initial_bankroll) * (critical_threshold_pct/100)

# -------------------------------
# UI compacte : boutons segments (grille 4 colonnes) - mobile friendly
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
            # clé unique par segment + index pour éviter duplication d'état
            if cols[c].button(seg, key=f"segbtn_{seg}_{idx}"):
                st.session_state.history.append(seg)
            idx += 1

segment_buttons_grid(segments, cols_per_row=4)

# actions compacts
act_col1, act_col2 = st.columns([1,1])
with act_col1:
    if st.button("↩ Suppr dernier", key="btn_suppr_hist"):
        if st.session_state.history:
            st.session_state.history.pop()
            st.success("Dernier historique supprimé.")
with act_col2:
    if st.button("🏁 Fin historique", key="btn_fin_hist"):
        st.success(f"Historique enregistré ({len(st.session_state.history)} spins). Le bot est prêt à suggérer.")
        # --- CALCULER ET STOCKER LA STRAT POUR LE 1ER LIVE SPIN ---
        if st.session_state.history:
            next_name, next_mises = None, {}
            next_name, next_mises = None, {}
            # appel de la fonction intelligente (définie plus bas)
            next_name, next_mises = choose_strategy_intelligent(st.session_state.history, st.session_state.bankroll)
            st.session_state.last_suggestion_name = next_name
            st.session_state.last_suggestion_mises = next_mises
        else:
            st.warning("Historique vide — aucune suggestion calculée.")

# affichage du tableau historique manuel (sans simulation)
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
# UI compacte : multiplicateurs rapides + manuel
# -------------------------------
st.subheader("⚡ Multiplicateurs (Top slot) — boutons rapides")
preset_mults = [2,5,10,25,50,100]
cols_mult = st.columns(len(preset_mults))
for i,m in enumerate(preset_mults):
    if cols_mult[i].button(f"x{m}", key=f"mult_preset_{m}"):
        st.session_state.mult_real_default = m

# champ manuel compact
st.write("Ou choisis une valeur manuelle :")
mult_input = st.number_input("x (manuel)", min_value=1, max_value=200, value=int(st.session_state.mult_real_default), step=1, key="mult_input_compact")
st.session_state.mult_real_default = int(mult_input)
mult_real_for_spin = st.session_state.mult_real_default

# -------------------------------
# Fonctions probabilités & utilitaires
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
    # mult_real is a uniform top-slot multiplier used for bonus segments
    mult_table = {'1':2,'2':3,'5':6,'10':11,
                  'Cash Hunt':mult_real,
                  'Pachinko':mult_real,
                  'Coin Flip':mult_real,
                  'Crazy Time':mult_real}
    mise_total = sum(mises_utilisees.values())
    gain = 0.0
    if spin_result in mises_utilisees and mises_utilisees[spin_result] > 0:
        gain = mises_utilisees[spin_result] * mult_table[spin_result]
    gain_net = gain - (mise_total - mises_utilisees.get(spin_result, 0.0))
    new_bankroll = float(bankroll) + float(gain_net)
    return float(gain_net), float(mise_total), float(new_bankroll)

# -------------------------------
# Stratégie intelligente (basée sur prob + historique)
# -------------------------------
def choose_strategy_intelligent(history, bankroll):
    # no-bet si bankroll critique
    if float(bankroll) <= float(critical_threshold_value):
        return "No-Bet", {k:0.0 for k in segments}

    unit = adjust_unit(bankroll)
    scale = unit / st.session_state.base_unit if st.session_state.base_unit > 0 else 1.0
    probs = compute_segment_probabilities(history)

    strategies = {}
    # martingale sur le segment le plus probable
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

    # si aucun bonus dans les 15 derniers spins => favoriser stratégies avec bonus
    recent_history = history[-15:] if len(history) >= 15 else history
    if not any(seg in recent_history for seg in ["Cash Hunt","Pachinko","Coin Flip","Crazy Time"]):
        bonus_strats = {name:strat for name,strat in strategies.items() if "Bonus" in name}
        if bonus_strats:
            strategies = bonus_strats

    # choisir candidats : ceux qui misent sur le segment le plus probable (si possible), sinon toutes
    max_prob = max([probs.get(seg,0) for seg in segments])
    candidates = [name for name,strat in strategies.items() if any(strat.get(seg,0)>0 and probs.get(seg,0)==max_prob for seg in segments)]
    if not candidates:
        candidates = list(strategies.keys())

    # choix aléatoire contrôlé parmi candidats (variation)
    best_name = random.choice(candidates)
    best_mises = strategies[best_name]

    return best_name, best_mises

# -------------------------------
# Affichage de la suggestion courante (prochaine mise)
# -------------------------------
st.subheader("📊 Stratégie suggérée (prochaine mise)")
# IMPORTANT : n'**écrase** pas last_suggestion_mises ici — on affiche seulement ce qui a été stocké par "Fin historique"
if st.session_state.last_suggestion_name:
    st.markdown(f"**Stratégie :** {st.session_state.last_suggestion_name}")
    st.markdown("**Mises proposées :**")
    st.write({k: round(v,2) for k,v in st.session_state.last_suggestion_mises.items()})
else:
    st.write("Pas encore de suggestion. Appuie sur 'Fin historique' pour que le bot calcule la stratégie pour le 1er spin.")

# -------------------------------
# Mode Live (Enregistrer spin)
# -------------------------------
st.header("Spin Live")
spin_val = st.selectbox("Spin Sorti", segments)

live_col1, live_col2 = st.columns([1,1])
with live_col1:
    if st.button("Enregistrer Spin (utilise la suggestion stockée)"):
        # récupère la suggestion stockée (celle qui a été calculée pour CE spin)
        mises_for_spin = st.session_state.last_suggestion_mises.copy() if st.session_state.last_suggestion_mises else {}

        # s'il n'y a pas de suggestion (cas edge), on calcule une suggestion "on the fly" (mais normalement Fin historique doit l'avoir fait)
        if not mises_for_spin:
            tmp_name, mises_for_spin = choose_strategy_intelligent(st.session_state.history, st.session_state.bankroll)
            st.session_state.last_suggestion_name = tmp_name
            st.session_state.last_suggestion_mises = mises_for_spin

        # calcul du résultat du spin en utilisant les mises_for_spin (qui étaient suggérées POUR ce spin)
        gain_net, mise_total, new_bankroll = process_spin_real(spin_val, mises_for_spin, st.session_state.bankroll, mult_real_for_spin)

        # enregistrer le résultat (on ajoute le spin APRÈS le calcul)
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
            "Gain Net": round(gain_net,2),
            "Bankroll": round(new_bankroll,2),
            "Multiplicateur": mult_real_for_spin
        })

        # Après l'ajout du spin, CALCULER et STOCKER la suggestion POUR LE PROCHAIN SPIN
        next_name, next_mises = choose_strategy_intelligent(st.session_state.history, st.session_state.bankroll)
        st.session_state.last_suggestion_name = next_name
        st.session_state.last_suggestion_mises = next_mises

        st.success(f"Spin enregistré : {spin_val} x{mult_real_for_spin} — Gain net: {round(gain_net,2)} — Bankroll: {round(new_bankroll,2)}")

with live_col2:
    if st.button("Supprimer dernier live spin"):
        # suppression sécurisée
        if st.session_state.live_history:
            st.session_state.live_history.pop()
        if st.session_state.results_table:
            st.session_state.results_table.pop()
        if st.session_state.history:
            st.session_state.history.pop()
        st.warning("Dernier live spin supprimé.")
        # recalcule suggestion après suppression
        next_name, next_mises = choose_strategy_intelligent(st.session_state.history, st.session_state.bankroll)
        st.session_state.last_suggestion_name = next_name
        st.session_state.last_suggestion_mises = next_mises

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

# -------------------------------
# Tester une stratégie manuellement (compact)
# -------------------------------
st.subheader("⚡ Tester une stratégie manuellement")
strategy_choice = st.selectbox(
    "Choisir une stratégie",
    ["Martingale_1","Martingale_2","Martingale_5","Martingale_10",
     "God Mode","God Mode + Bonus","1 + Bonus","No-Bet"]
)

if st.button("Tester Stratégie (simulate)"):
    bankroll_test = st.session_state.initial_bankroll
    test_results = []
    history_test = st.session_state.history.copy()

    for i, spin in enumerate(history_test, start=1):
        # Construire les mises selon la stratégie
        if strategy_choice == "No-Bet":
            mises = {k:0.0 for k in segments}
        elif "Martingale" in strategy_choice:
            target = strategy_choice.split("_")[1]
            mises = {k:(st.session_state.base_unit if k == target else 0.0) for k in segments}
        elif strategy_choice == "God Mode":
            scale = adjust_unit(bankroll_test) / st.session_state.base_unit
            mises = {'1':0.0,'2':round(3*scale,2),'5':round(2*scale,2),'10':round(1*scale,2),
                     'Cash Hunt':0,'Pachinko':0,'Coin Flip':0,'Crazy Time':0}
        elif strategy_choice == "God Mode + Bonus":
            scale = adjust_unit(bankroll_test) / st.session_state.base_unit
            mises = {'1':0.0,'2':round(3*scale,2),'5':round(2*scale,2),'10':round(1*scale,2),
                     'Cash Hunt':round(1*scale,2),'Pachinko':round(1*scale,2),
                     'Coin Flip':round(1*scale,2),'Crazy Time':round(1*scale,2)}
        elif strategy_choice == "1 + Bonus":
            scale = adjust_unit(bankroll_test) / st.session_state.base_unit
            mises = {'1':round(4*scale,2),'2':0.0,'5':0.0,'10':0.0,
                     'Cash Hunt':round(1*scale,2),'Pachinko':round(1*scale,2),
                     'Coin Flip':round(1*scale,2),'Crazy Time':round(1*scale,2)}
        else:
            mises = {k:0.0 for k in segments}

        gain_net, mise_total, bankroll_test = process_spin_real(spin, mises, bankroll_test, mult_real_for_spin)
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

    # Graphique bankroll
    st.subheader("📊 Évolution bankroll (test stratégie)")
    fig, ax = plt.subplots()
    ax.plot(df_test["Spin #"], df_test["Bankroll"], marker='o', label='Bankroll (test)')
    ax.axhline(y=st.session_state.initial_bankroll, color='gray', linestyle='--', label='Bankroll initiale')
    ax.set_xlabel("Spin #")
    ax.set_ylabel("Bankroll ($)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
