import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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
    st.session_state.bankroll = 150.0           # bankroll courant
if 'initial_bankroll' not in st.session_state:
    st.session_state.initial_bankroll = 150.0   # bankroll initial (référence)
if 'base_unit' not in st.session_state:
    st.session_state.base_unit = 1.0
if 'last_gain' not in st.session_state:
    st.session_state.last_gain = 0.0

segments = ['1','2','5','10','Bonus']

# -------------------------------
# Barre latérale (widgets retournent une valeur — on ne lie PAS la key directement à session_state ici)
# -------------------------------
st.sidebar.header("Paramètres Crazy Time Bot")

initial_bankroll_input = st.sidebar.number_input(
    "Bankroll initial ($)",
    min_value=50.0,
    max_value=10000.0,
    value=float(st.session_state.initial_bankroll),
    step=1.0
)
# si l'utilisateur change la bankroll initiale via la sidebar, on met à jour
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
    "Seuil critique bankroll (%) (relatif au bankroll initial)",
    min_value=1, max_value=100, value=25, step=1
)
# seuil basé sur bankroll initial (plus stable)
critical_threshold_value = float(st.session_state.initial_bankroll) * (critical_threshold_pct/100)

# -------------------------------
# Entrée historique manuel (pas de simulation automatique)
# -------------------------------
st.header("Historique Spins (manuel) — entre les résultats ici")
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

# Afficher tableau historique manuel (SANS simulation) — demandé
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
    """Combine probabilité de base (répartition roue) + pondération historique (50/50)."""
    segment_count = {'1':1,'2':2,'5':2,'10':1,'Bonus':1}
    total_segments = sum(segment_count.values())
    base_prob = {k: v/total_segments for k,v in segment_count.items()}
    hist_weight = {k: (history.count(k)/len(history) if history else 0) for k in segment_count.keys()}
    # mix 50/50 base vs historique
    prob = {k: 0.5*base_prob[k] + 0.5*hist_weight[k] for k in segment_count.keys()}
    return prob

def adjust_unit(bankroll):
    """Ajuste l'unité en fonction du bankroll par rapport à initial_bankroll."""
    if bankroll >= 2.5 * st.session_state.initial_bankroll:
        return st.session_state.base_unit * 2.0
    elif bankroll <= 0.5 * st.session_state.initial_bankroll:
        return st.session_state.base_unit * 0.5
    return st.session_state.base_unit

def estimate_ev(mises, probs, bonus_mult):
    """Estime EV (gain net moyen par spin) pour un set de mises donné."""
    mult_table = {'1':2,'2':3,'5':6,'10':11}
    mise_total = sum(mises.values())
    ev = 0.0
    for seg in segments:
        bet = mises.get(seg, 0.0)
        if bet <= 0:
            continue
        payout = (bonus_mult if seg=='Bonus' else mult_table[seg])
        gain_if_hit = bet * payout
        gain_net_if_hit = gain_if_hit - (mise_total - bet)
        ev += probs[seg] * gain_net_if_hit
    return ev

def choose_strategy_intelligent(history, bankroll):
    """Retourne la meilleure stratégie (par EV estimé) ou No-Bet."""
    # Force no-bet si banque critique
    if float(bankroll) <= float(critical_threshold_value):
        return "No-Bet", {k:0.0 for k in segments}, 0.0

    unit = adjust_unit(bankroll)
    scale = unit / st.session_state.base_unit if st.session_state.base_unit>0 else 1.0

    probs = compute_segment_probabilities(history)

    # Construire stratégies (en $) (on applique 'scale' pour adapter aux unités dynamiques)
    strategies = {}

    # Martingale ciblée sur le segment le plus probable
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    target_segment = sorted_probs[0][0]
    mises_mart = {k: (unit if k==target_segment else 0.0) for k in segments}
    strategies[f"Martingale_{target_segment}"] = mises_mart

    # GodMode (scale applied)
    strategies["GodMode"] = {
        '1': 0.0,
        '2': round(3.0 * scale, 2),
        '5': round(2.0 * scale, 2),
        '10': round(1.0 * scale, 2),
        'Bonus': 0.0
    }

    # GodMode + Bonus (1$ par bonus -> 4$ total scaled)
    strategies["GodMode+Bonus"] = {
        '1': 0.0,
        '2': round(3.0 * scale, 2),
        '5': round(2.0 * scale, 2),
        '10': round(1.0 * scale, 2),
        'Bonus': round(4.0 * scale, 2)
    }

    # 1 + Bonus (4$ on 1 and 1$ each bonus -> total 8$ scaled)
    strategies["1+Bonus"] = {
        '1': round(4.0 * scale, 2),
        '2': 0.0,
        '5': 0.0,
        '10': 0.0,
        'Bonus': round(4.0 * scale, 2)
    }

    # Évaluer EV pour chaque stratégie
    evs = {}
    for name, mises in strategies.items():
        evs[name] = estimate_ev(mises, probs, bonus_multiplier_assumption)

    # Choisir la stratégie de plus grande EV
    best_name = max(evs.items(), key=lambda x: x[1])[0]
    best_ev = evs[best_name]
    best_mises = strategies[best_name]

    # Si EV <= 0 -> No-Bet
    if best_ev <= 0:
        return "No-Bet", {k:0.0 for k in segments}, best_ev

    return best_name, best_mises, best_ev

# -------------------------------
# Calcul/Exécution d'un spin
# -------------------------------
def process_spin(spin_result, multiplier, mises_utilisees, bankroll):
    """Calcule gain net, mise total, new bankroll, retourne valeurs."""
    mise_total = sum(mises_utilisees.values())
    gain = 0.0
    if spin_result in mises_utilisees and mises_utilisees[spin_result] > 0:
        if spin_result == 'Bonus':
            gain = mises_utilisees[spin_result] * multiplier
        else:
            mult_table = {'1':2,'2':3,'5':6,'10':11}
            gain = mises_utilisees[spin_result] * mult_table[spin_result]
    # gain_net selon ta formule : gain - (mise_total - mise_sur_segment)
    gain_net = gain - (mise_total - mises_utilisees.get(spin_result, 0.0))
    new_bankroll = float(bankroll) + float(gain_net)
    return float(gain_net), float(mise_total), float(new_bankroll)

# -------------------------------
# Suggestion actuelle (affichée avant live)
# -------------------------------
st.subheader("📊 Suggestion (basée sur l'historique manuel)")
if st.session_state.history:
    strat_name, strat_mises, strat_ev = choose_strategy_intelligent(st.session_state.history, st.session_state.bankroll)
    st.write("Stratégie suggérée:", strat_name)
    st.write("Mises proposées ( $ ): ", {k: round(v,2) for k,v in strat_mises.items()})
    st.write("EV estimé (par spin) :", round(strat_ev, 3))
else:
    st.write("Entrez l'historique manuel (boutons ci-dessus) puis appuie sur 'Fin historique et commencer' pour que le bot suggère.")

# -------------------------------
# Mode Live (enregistrer spins)
# -------------------------------
st.header("Spin Live")
spin_val = st.selectbox("Spin Sorti", segments)
multiplier_val = st.number_input("Multiplicateur réel (pour Bonus)", 1, 200, value=1, step=1)

live_col1, live_col2 = st.columns([1,1])
with live_col1:
    if st.button("Enregistrer Spin"):
        # on utilise l'historique manuel pour suggérer, comme demandé
        strat_name, strat_mises, strat_ev = choose_strategy_intelligent(st.session_state.history, st.session_state.bankroll)

        # si No-Bet -> mises nulles
        if strat_name == "No-Bet":
            strat_mises = {k:0.0 for k in segments}

        # calcul
        gain_net, mise_total, new_bankroll = process_spin(spin_val, multiplier_val, strat_mises, st.session_state.bankroll)

        # mettre à jour l'historique & bankroll & results_table
        st.session_state.history.append(spin_val)   # tu voulais garder history + live dans la même liste
        st.session_state.live_history.append(spin_val)
        st.session_state.last_gain = float(gain_net)

        # forcer float avant assignation pour éviter erreur Streamlit
        st.session_state.bankroll = float(new_bankroll)

        # enregistrer dans tableau live
        st.session_state.results_table.append({
            "Spin #": len(st.session_state.results_table) + 1,
            "Résultat": spin_val,
            "Stratégie": strat_name,
            "Mises $": {k: round(v,2) for k,v in strat_mises.items()},
            "Mise Totale": round(mise_total,2),
            "Gain Net": round(gain_net,2),
            "Bankroll": round(new_bankroll,2)
        })

        st.success(f"Spin: {spin_val} — Gain net: {round(gain_net,2)} — Bankroll: {round(new_bankroll,2)}")

with live_col2:
    if st.button("Supprimer dernier live spin"):
        # supprime du live_history + results_table + history si présent
        if st.session_state.live_history:
            st.session_state.live_history.pop()
        if st.session_state.results_table:
            st.session_state.results_table.pop()
        if st.session_state.history:
            # on suppose que le dernier élément de history correspond au dernier live si tu veux
            # sinon tu peux gérer différemment
            st.session_state.history.pop()
        st.warning("Dernier live spin supprimé.")

# -------------------------------
# Affichage tableau live (spin par spin) + courbe bankroll
# -------------------------------
st.subheader("📈 Historique des Spins Live (tableau)")
if st.session_state.results_table:
    df_results = pd.DataFrame(st.session_state.results_table)
    st.dataframe(df_results, use_container_width=True)

    # Graphe bankroll en temps réel
    st.subheader("📊 Évolution de la Bankroll (live)")
    fig, ax = plt.subplots()
    ax.plot(df_results["Spin #"], df_results["Bankroll"], marker='o', label='Bankroll')
    # ligne horizontale bankroll initiale
    ax.axhline(y=st.session_state.initial_bankroll, color='gray', linestyle='--', label='Bankroll initiale')
    ax.set_xlabel("Spin #")
    ax.set_ylabel("Bankroll ($)")
    ax.set_title("Évolution de la bankroll en temps réel")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
else:
    st.write("Aucun spin live enregistré encore.")

# -------------------------------
# Test manuel stratégie (bouton)
# -------------------------------
st.subheader("⚡ Tester une stratégie manuellement (ne change pas l'état)")
strategy_choice = st.selectbox(
    "Choisir une stratégie",
    ["Martingale_1","Martingale_2","Martingale_5","Martingale_10","GodMode","GodMode+Bonus","1+Bonus","No-Bet"]
)

if st.button("Tester Stratégie"):
    if strategy_choice == "No-Bet":
        mises = {k:0.0 for k in segments}
    elif "Martingale" in strategy_choice:
        target = strategy_choice.split("_")[1]
        mises = {k: (st.session_state.base_unit if k == target else 0.0) for k in segments}
    elif strategy_choice == "GodMode":
        scale = adjust_unit(st.session_state.bankroll) / st.session_state.base_unit
        mises = {'1':0.0,'2':round(3*scale,2),'5':round(2*scale,2),'10':round(1*scale,2),'Bonus':0.0}
    elif strategy_choice == "GodMode+Bonus":
        scale = adjust_unit(st.session_state.bankroll) / st.session_state.base_unit
        mises = {'1':0.0,'2':round(3*scale,2),'5':round(2*scale,2),'10':round(1*scale,2),'Bonus':round(4*scale,2)}
    elif strategy_choice == "1+Bonus":
        scale = adjust_unit(st.session_state.bankroll) / st.session_state.base_unit
        mises = {'1':round(4*scale,2),'2':0.0,'5':0.0,'10':0.0,'Bonus':round(4*scale,2)}
    else:
        mises = {k:0.0 for k in segments}
    st.info(f"Stratégie {strategy_choice} → Mises: {mises}")

# -------------------------------
# Fin du script
# -------------------------------
