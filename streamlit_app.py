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
# last suggestion stored and used when saving a live spin
if 'last_suggestion_name' not in st.session_state:
    st.session_state.last_suggestion_name = None
if 'last_suggestion_mises' not in st.session_state:
    st.session_state.last_suggestion_mises = {}
# martingale loss streak for '1' (Martingale_1)
if 'martingale_1_loss_streak' not in st.session_state:
    st.session_state.martingale_1_loss_streak = 0
# manual top-slot multiplicator (appliqué to the spin result)
if 'mult_real_manual' not in st.session_state:
    st.session_state.mult_real_manual = 1

# segments (numbers + separate bonus games)
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
    """Combine distribution (by #segments) and empirical frequency (history) equally."""
    segment_count = {'1':1,'2':2,'5':2,'10':1,'Cash Hunt':1,'Pachinko':1,'Coin Flip':1,'Crazy Time':1}
    total_segments = sum(segment_count.values())
    base_prob = {k: v/total_segments for k,v in segment_count.items()}
    hist_weight = {k: (history.count(k)/len(history) if history else 0) for k in segment_count.keys()}
    prob = {k: 0.5*base_prob[k] + 0.5*hist_weight[k] for k in segment_count.keys()}
    return prob

def adjust_unit(bankroll):
    """Adjust base unit depending on current bankroll vs initial (simple example)."""
    if bankroll >= 2.5 * st.session_state.initial_bankroll:
        return st.session_state.base_unit * 2.0
    elif bankroll <= 0.5 * st.session_state.initial_bankroll:
        return st.session_state.base_unit * 0.5
    return st.session_state.base_unit

def martingale_1_mise(base_unit):
    """Return current Martingale mise for '1' based on loss streak and protect against huge bets."""
    streak = st.session_state.martingale_1_loss_streak
    # avoid runaway: cap doubling to e.g. 10 steps
    cap = 10
    n = min(streak, cap)
    return base_unit * (2 ** n)

def process_spin_real(spin_result, mises_utilisees, bankroll, mult_manual):
    """
    Apply the manual multiplicator (default 1) to the spun segment.
    Returns: gain_net, gain_brut, mise_total, new_bankroll, multiplicateur_applique
    """
    # default mult: 1 for all segments, but we apply mult_manual uniformly to the spun segment
    # (as user requested: manual multiplier always applied on the spin result)
    mult_table = {seg:1 for seg in segments}
    # apply manual multiplier uniformly (so if mult_manual==1 -> default behavior)
    if mult_manual is not None:
        for seg in mult_table:
            mult_table[seg] = int(mult_manual)

    mise_total = float(sum(mises_utilisees.values()))
    gain_brut = 0.0
    mult_applique = 1
    if spin_result in mises_utilisees and mises_utilisees.get(spin_result, 0) > 0:
        mult_applique = mult_table.get(spin_result, 1)
        gain_brut = float(mises_utilisees[spin_result]) * float(mult_applique)
    # gain_net = gain_brut - (mise_total - mise_sur_segment)
    gain_net = gain_brut - (mise_total - float(mises_utilisees.get(spin_result, 0.0)))
    new_bankroll = float(bankroll) + float(gain_net)
    return float(gain_net), float(gain_brut), float(mise_total), float(new_bankroll), int(mult_applique)

# -------------------------------
# Strategy chooser (single suggestion stored)
# -------------------------------
def choose_strategy_intelligent(history, bankroll):
    """
    Build candidate strategies and pick a single suggestion based on probabilities.
    Selection criterion: maximize total hit-probability (sum of probs of segments we bet on).
    Ties broken randomly.
    """
    if float(bankroll) <= float(critical_threshold_value):
        return "No-Bet", {k:0.0 for k in segments}

    unit = adjust_unit(bankroll)
    scale = unit / st.session_state.base_unit if st.session_state.base_unit > 0 else 1.0
    probs = compute_segment_probabilities(history)

    strategies = {}
    # Martingale targeted on '1' (we will compute actual martingale mise later when applying)
    strategies["Martingale_1"] = {k:(unit if k=='1' else 0.0) for k in segments}

    # God Mode (fixed proportions, scaled)
    strategies["God Mode"] = {
        '1':0.0,
        '2': round(3.0 * scale,2),
        '5': round(2.0 * scale,2),
        '10': round(1.0 * scale,2),
        'Cash Hunt': 0.0,
        'Pachinko': 0.0,
        'Coin Flip': 0.0,
        'Crazy Time': 0.0
    }
    # God Mode + Bonus
    strategies["God Mode + Bonus"] = {
        '1':0.0,
        '2': round(3.0 * scale,2),
        '5': round(2.0 * scale,2),
        '10': round(1.0 * scale,2),
        'Cash Hunt': round(1.0 * scale,2),
        'Pachinko': round(1.0 * scale,2),
        'Coin Flip': round(1.0 * scale,2),
        'Crazy Time': round(1.0 * scale,2)
    }
    # 1 + Bonus
    strategies["1 + Bonus"] = {
        '1': round(4.0 * scale,2),
        '2': 0.0,
        '5': 0.0,
        '10': 0.0,
        'Cash Hunt': round(1.0 * scale,2),
        'Pachinko': round(1.0 * scale,2),
        'Coin Flip': round(1.0 * scale,2),
        'Crazy Time': round(1.0 * scale,2)
    }

    # Simple scoring: sum of probabilities of segments with stake > 0
    scores = {}
    for name, mise in strategies.items():
        score = sum([probs.get(seg, 0.0) for seg, amt in mise.items() if amt > 0])
        scores[name] = score

    max_score = max(scores.values())
    candidates = [name for name, s in scores.items() if s >= max_score - 1e-12]

    # Variation: if no bonus in recent history, give a small boost to strategies that include bonuses
    recent = history[-15:] if len(history) >= 15 else history
    if not any(x in recent for x in ['Cash Hunt','Pachinko','Coin Flip','Crazy Time']):
        bonus_candidates = [c for c in candidates if ('Bonus' in c) or ('Bonus' not in c and any(k in strategies[c] and k in ['Cash Hunt','Pachinko','Coin Flip','Crazy Time'] and strategies[c][k]>0 for k in segments))]
        if bonus_candidates:
            candidates = bonus_candidates

    pick = random.choice(candidates)
    chosen_mises = strategies[pick]

    # For Martingale_1, compute actual stake according to loss streak
    if pick == "Martingale_1":
        m = martingale_1_mise(st.session_state.base_unit)  # base_unit is the base $ unit
        chosen_mises = {k:(m if k=='1' else 0.0) for k in segments}

    return pick, chosen_mises

# -------------------------------
# UI: compact segment buttons (mobile-friendly)
# -------------------------------
st.title("Super Crazy Time Bot — (référence)")

st.header("Historique Spins (manuel) — entre les résultats ici")
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

col_a, col_b = st.columns([1,1])
with col_a:
    if st.button("↩ Suppr dernier historique"):
        if st.session_state.history:
            st.session_state.history.pop()
            st.success("Dernier historique supprimé.")
with col_b:
    if st.button("🏁 Fin historique"):
        st.success(f"Historique enregistré ({len(st.session_state.history)} spins). Le bot calcule la suggestion pour le 1er spin.")
        # When finishing history, compute and store suggestion for the *first* live spin
        if st.session_state.history:
            name, mises = choose_strategy_intelligent(st.session_state.history, st.session_state.bankroll)
            st.session_state.last_suggestion_name = name
            st.session_state.last_suggestion_mises = mises
        else:
            st.session_state.last_suggestion_name = None
            st.session_state.last_suggestion_mises = {}

# show manual history (no simulation)
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
# Multiplicateur manuel (appliqué toujours sur le spin result)
# -------------------------------
st.subheader("⚡ Multiplicateur manuel (appliqué sur le segment sorti)")
mult_manual_input = st.number_input("Multiplier (manuel) — par défaut x1", min_value=1, max_value=200, value=int(st.session_state.mult_real_manual), step=1)
st.session_state.mult_real_manual = int(mult_manual_input)

# -------------------------------
# Display single suggestion (only from stored session_state)
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

# -------------------------------
# Live spin recording (uses stored suggestion) - then recomputes suggestion for next spin
# -------------------------------
st.header("Spin Live — Enregistrement")
spin_val = st.selectbox("Spin sorti (à cliquer pour enregistrer)", segments)

col_save, col_del = st.columns([1,1])
with col_save:
    if st.button("Enregistrer Spin (utilise suggestion stockée)"):
        # mises to use for this spin = last stored suggestion (the one displayed)
        mises_for_spin = st.session_state.last_suggestion_mises.copy() if st.session_state.last_suggestion_mises else {}
        # If no suggestion stored (edge), compute on the fly using current history
        if not mises_for_spin:
            tmp_name, mises_for_spin = choose_strategy_intelligent(st.session_state.history, st.session_state.bankroll)
            st.session_state.last_suggestion_name = tmp_name
            st.session_state.last_suggestion_mises = mises_for_spin

        # compute result using the mises_for_spin that were suggested BEFORE the spin
        gain_net, gain_brut, mise_total, new_bankroll, mult_applique = process_spin_real(
            spin_val,
            mises_for_spin,
            st.session_state.bankroll,
            st.session_state.mult_real_manual
        )

        # update martingale loss streak for '1'
        # if we bet on '1' and we won, reset streak; if we bet on '1' and lost, increase streak
        if mises_for_spin.get('1', 0.0) > 0:
            if spin_val == '1' and gain_net > 0:
                st.session_state.martingale_1_loss_streak = 0
            else:
                # if bet on '1' and it didn't hit -> increment
                if spin_val != '1':
                    st.session_state.martingale_1_loss_streak += 1
                else:
                    # hit but maybe net <=0 (shouldn't happen normally), treat as win if net>0 else increment
                    if gain_net <= 0:
                        st.session_state.martingale_1_loss_streak += 1

        # record the spin (add to history AFTER using the suggestion)
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

        # compute and store suggestion for the next spin (so suggestion always refers to "next" spin)
        next_name, next_mises = choose_strategy_intelligent(st.session_state.history, st.session_state.bankroll)
        st.session_state.last_suggestion_name = next_name
        st.session_state.last_suggestion_mises = next_mises

        # show updated suggestion immediately
        display_next_suggestion()
        st.success(f"Spin enregistré : {spin_val} x{mult_applique} — Gain net: {round(gain_net,2)} — Bankroll: {round(new_bankroll,2)}")

with col_del:
    if st.button("Supprimer dernier live spin"):
        # remove last entries safely
        if st.session_state.live_history:
            st.session_state.live_history.pop()
        if st.session_state.results_table:
            st.session_state.results_table.pop()
        if st.session_state.history:
            st.session_state.history.pop()
        # adjust martingale streak conservatively
        if st.session_state.martingale_1_loss_streak > 0:
            st.session_state.martingale_1_loss_streak -= 1
        st.warning("Dernier live spin supprimé.")
        # recompute suggestion after deletion
        next_name, next_mises = choose_strategy_intelligent(st.session_state.history, st.session_state.bankroll)
        st.session_state.last_suggestion_name = next_name
        st.session_state.last_suggestion_mises = next_mises
        display_next_suggestion()

# -------------------------------
# Live table + bankroll plot
# -------------------------------
st.subheader("📈 Historique des Spins Live")
if st.session_state.results_table:
    df_results = pd.DataFrame(st.session_state.results_table)
    st.dataframe(df_results, use_container_width=True)

    st.subheader("📊 Évolution de la Bankroll")
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
# Manual strategy tester (simulate on current manual history) + bankroll curve
# -------------------------------
st.subheader("⚡ Tester une stratégie manuellement (simulate)")
strategy_choice = st.selectbox(
    "Choisir une stratégie",
    ["Martingale_1", "God Mode", "God Mode + Bonus", "1 + Bonus", "No-Bet"]
)

if st.button("Tester Stratégie (simulate)"):
    bankroll_test = float(st.session_state.initial_bankroll)
    test_results = []
    history_test = st.session_state.history.copy()
    for i, spin in enumerate(history_test, start=1):
        # build mise for chosen strategy (scale with adjust_unit(bankroll_test))
        scale = adjust_unit(bankroll_test) / st.session_state.base_unit
        if strategy_choice == "No-Bet":
            mises = {k:0.0 for k in segments}
        elif strategy_choice == "Martingale_1":
            # reset streak for this simulation (we could simulate based on stored streak, keep simple)
            mises = {k:(st.session_state.base_unit if k=='1' else 0.0) for k in segments}
        elif strategy_choice == "God Mode":
            mises = {'1':0.0,'2':round(3*scale,2),'5':round(2*scale,2),'10':round(1*scale,2),
                     'Cash Hunt':0.0,'Pachinko':0.0,'Coin Flip':0.0,'Crazy Time':0.0}
        elif strategy_choice == "God Mode + Bonus":
            mises = {'1':0.0,'2':round(3*scale,2),'5':round(2*scale,2),'10':round(1*scale,2),
                     'Cash Hunt':round(1*scale,2),'Pachinko':round(1*scale,2),
                     'Coin Flip':round(1*scale,2),'Crazy Time':round(1*scale,2)}
        elif strategy_choice == "1 + Bonus":
            mises = {'1':round(4*scale,2),'2':0.0,'5':0.0,'10':0.0,
                     'Cash Hunt':round(1*scale,2),'Pachinko':round(1*scale,2),
                     'Coin Flip':round(1*scale,2),'Crazy Time':round(1*scale,2)}
        else:
            mises = {k:0.0 for k in segments}

        gain_net, gain_brut, mise_total, bankroll_test, mult_applique = process_spin_real(spin, mises, bankroll_test, st.session_state.mult_real_manual)
        test_results.append({
            "Spin #": i,
            "Résultat": spin,
            "Mises": mises,
            "Mise Totale": mise_total,
            "Gain Brut": gain_brut,
            "Gain Net": gain_net,
            "Multiplicateur appliqué": mult_applique,
            "Bankroll": bankroll_test
        })

    df_test = pd.DataFrame(test_results)
    st.dataframe(df_test, use_container_width=True)

    fig2, ax2 = plt.subplots()
    ax2.plot(df_test["Spin #"], df_test["Bankroll"], marker='o', label='Bankroll (test)')
    ax2.axhline(y=st.session_state.initial_bankroll, color='gray', linestyle='--', label='Bankroll initiale')
    ax2.set_xlabel("Spin #")
    ax2.set_ylabel("Bankroll ($)")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)
