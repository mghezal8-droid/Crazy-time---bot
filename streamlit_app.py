# streamlit_app_suggestions.py
"""
Crazy Time — Full Streamlit app
- Fin historique => calcule et affiche stratégie suggérée pour prochain spin; stocke la suggestion.
- Enregistrer spin live => applique la stratégie suggérée précédente, calcule gain réel, met à jour bankroll,
  complète la ligne de strategies_table, puis calcule + stocke la suggestion pour le spin suivant.
- ML + RTP + Top Slot + plusieurs stratégies + simulation disponible.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# Config / constants
# ---------------------------
st.set_page_config(page_title="🎰 Crazy Time — Suggestion + Live", layout="wide")

SEGMENTS = ['1','2','5','10','Coin Flip','Cash Hunt','Pachinko','Crazy Time','Top Slot']
BONUSES = ['Coin Flip','Cash Hunt','Pachinko','Crazy Time']
VAL_SEG = {'1':1,'2':2,'5':5,'10':10}  # numeric segments fixed; bonuses handled as variable
THEO_COUNTS = {'1':21,'2':13,'5':7,'10':4,'Coin Flip':4,'Cash Hunt':2,'Pachinko':2,'Crazy Time':1,'Top Slot':1}
THEO_TOTAL = sum(THEO_COUNTS.values())

# RTP baselines for heuristic adjustments
RTP_BASE_LOW = 94.41
RTP_BASE_HIGH = 96.08

# ---------------------------
# Session state initialization
# ---------------------------
def ss_init(k,v):
    if k not in st.session_state:
        st.session_state[k] = v

ss_init("history", [])                # manual pre-live history entries (list of dicts or strings)
ss_init("live_history", [])           # list of dicts {"segment","mult"}
ss_init("results_table", [])          # recorded live spins (spin-by-spin)
ss_init("strategies_table", [])       # records of strategy suggestions + realized results
ss_init("bankroll", 200.0)
ss_init("initial_bankroll", st.session_state.bankroll)
ss_init("unit", 1.0)
ss_init("bonus_mult_assumption", 10)  # assumed multiplier for bonus EV calculations
ss_init("mult_for_ev", 1)
ss_init("ml_window", 50)
ss_init("ml_model", None)
ss_init("ml_trained", False)
ss_init("ml_boost", 1.5)
ss_init("rtp_weight_pct", 50)
ss_init("rtp_weight", st.session_state.rtp_weight_pct / 100.0)
ss_init("show_history_table", True)
ss_init("last_suggestion", None)      # holds last suggested dict
ss_init("miss_streak", 0)
ss_init("martingale_loss_streak", 0)

# ---------------------------
# Helper functions for history normalization
# ---------------------------
def normalize_entry(e):
    """Return (segment, mult) from history/live entries which may be string or dict."""
    if isinstance(e, dict):
        seg = e.get("segment")
        mult = float(e.get("mult", 1.0))
        return seg, mult
    else:
        return e, 1.0

def full_history_segments():
    segs = []
    for h in st.session_state.history:
        seg, _ = normalize_entry(h)
        segs.append(seg)
    for h in st.session_state.live_history:
        seg, _ = normalize_entry(h)
        segs.append(seg)
    return segs

# ---------------------------
# Probabilities & RTP helpers
# ---------------------------
def theo_prob(segment):
    return THEO_COUNTS.get(segment, 0) / THEO_TOTAL

def hist_prob(full_hist, segment, window=300):
    if not full_hist:
        return 0.0
    hist = full_hist[-window:]
    return hist.count(segment) / len(hist)

def combined_prob(full_hist, segment, window=300):
    w = st.session_state.rtp_weight
    p_theo = theo_prob(segment)
    p_hist = hist_prob(full_hist, segment, window)
    return w * p_theo + (1 - w) * p_hist

def compute_rtp_last_n(n=100):
    rows = st.session_state.results_table[-n:]
    if not rows:
        return None
    total_wagered = 0.0
    total_returned = 0.0
    for r in rows:
        mise = r.get("Mise Totale")
        if mise is None:
            mises = r.get("Mises $", {})
            if isinstance(mises, dict):
                mise = sum(mises.values())
            else:
                mise = 0.0
        gain_brut = r.get("Gain Brut", None)
        gain_net = r.get("Gain Net", None)
        if gain_brut is None and gain_net is not None:
            returned = mise + gain_net
        else:
            returned = gain_brut or 0.0
        total_wagered += float(mise or 0.0)
        total_returned += float(returned or 0.0)
    if total_wagered == 0:
        return None
    return (total_returned / total_wagered) * 100.0

# ---------------------------
# ML: train & predict for sliding window
# ---------------------------
def train_ml():
    segs = full_history_segments()
    if len(segs) <= st.session_state.ml_window:
        st.session_state.ml_trained = False
        return None
    idxs = [SEGMENTS.index(s) for s in segs]
    W = st.session_state.ml_window
    X = []
    y = []
    for i in range(len(idxs) - W):
        X.append(idxs[i:i+W])
        y.append(idxs[i+W])
    X = np.array(X); y = np.array(y)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X,y)
    st.session_state.ml_model = clf
    st.session_state.ml_trained = True
    return clf

def ml_predict_window():
    if not st.session_state.ml_trained or st.session_state.ml_model is None:
        return None, {}
    segs = full_history_segments()
    if len(segs) < st.session_state.ml_window:
        return None, {}
    last_window = [SEGMENTS.index(s) for s in segs[-st.session_state.ml_window:]]
    probs = st.session_state.ml_model.predict_proba([last_window])[0]
    per_seg = {SEGMENTS[i]: float(probs[i]) for i in range(len(probs))}
    pred = SEGMENTS[int(np.argmax(probs))]
    return pred, per_seg

# ---------------------------
# Strategy builders
# ---------------------------
def strat_martingale_1(bankroll, unit):
    return "Martingale 1", {'1': unit}

def strat_god_mode(bankroll, unit):
    return "God Mode", {'2': unit*2, '5': unit*1.5, '10': unit*1.0, 'Top Slot': unit*0.5}

def strat_god_mode_bonus(bankroll, unit):
    bets = {'2': unit*2, '5': unit*1.5, '10': unit*1.0}
    for b in BONUSES + ['Top Slot']:
        bets[b] = unit*1.0
    return "God Mode + Bonus", bets

def strat_1_plus_bonus(bankroll, unit):
    bets = {'1': unit*2}
    for b in BONUSES + ['Top Slot']:
        bets[b] = unit*1.0
    return "1+Bonus", bets

def strat_only_numbers(bankroll, unit):
    return "Only Numbers", {'1': unit*1.5, '2': unit*1.2, '5': unit*1.0, '10': unit*1.0, 'Top Slot': unit*0.5}

def strat_all_but_1(bankroll, unit):
    bets = {}
    for s in SEGMENTS:
        if s != '1':
            bets[s] = unit*1.0
    return "All but 1", bets

STRATEGIES = {
    "Martingale 1": strat_martingale_1,
    "God Mode": strat_god_mode,
    "God Mode + Bonus": strat_god_mode_bonus,
    "1+Bonus": strat_1_plus_bonus,
    "Only Numbers": strat_only_numbers,
    "All but 1": strat_all_but_1
}

# ---------------------------
# Compute bets given strategy, with ML & RTP adjustments
# ---------------------------
def compute_bets_for_strategy(strategy_name, bankroll, unit, rtp_last100=None, ml_pred=None, ml_probs=None):
    builder = STRATEGIES[strategy_name]
    name, base_bets = builder(bankroll, unit)
    bets = {seg: float(base_bets.get(seg, 0.0)) for seg in SEGMENTS}
    # RTP reaction
    if rtp_last100 is not None:
        if rtp_last100 < RTP_BASE_LOW:
            for b in BONUSES + ['Top Slot']:
                bets[b] *= 1.6
        elif rtp_last100 > RTP_BASE_HIGH:
            for n in ['1','2','5','10']:
                bets[n] *= 1.3
    # ML boost
    if ml_pred and ml_pred in bets and bets[ml_pred] > 0:
        bets[ml_pred] *= st.session_state.ml_boost
    scaling = 0.8 + st.session_state.rtp_weight  # gentle scaling
    bets = {k: round(v * scaling, 4) for k,v in bets.items()}
    return bets

# ---------------------------
# Expected value calc (used for comparing strategies)
# ---------------------------
def expected_value_for_strategy(mises, full_history, multiplicateur, bankroll):
    mise_totale = sum(mises.values()) if mises else 0.0
    ev = 0.0
    for seg in SEGMENTS:
        p = combined_prob(full_history, seg, window=st.session_state.ml_window)
        if seg in mises and mises[seg] > 0:
            if seg in VAL_SEG:
                seg_val = VAL_SEG[seg]
            else:
                seg_val = st.session_state.bonus_mult_assumption
            payout = mises[seg] * (seg_val * multiplicateur) + mises[seg]
            net_if_hit = payout - mise_totale
        else:
            net_if_hit = -mise_totale
        ev += p * net_if_hit
    return ev

# ---------------------------
# Gain calculation exactly as you requested:
# - numeric segments: gain_brut = (multiplier + 1) * mise  (i.e. 1 -> 2x, 2 -> 3x, etc.)
# - bonus segments: gain_brut = (bonus_mult * mise) + mise
# - Top Slot can be modeled as multiplier applied to the segment multiplicateur externally via multiplicateur param
# ---------------------------
def calcul_gain_from_bets(bets, spin_segment, multiplicateur=1.0, bonus_mult=None):
    mise_totale = sum(bets.values())
    if mise_totale <= 0:
        return 0.0, 0.0
    if bets.get(spin_segment, 0.0) <= 0:
        # total loss
        return 0.0, -mise_totale
    mise = bets[spin_segment]
    if spin_segment in VAL_SEG:
        # numeric case: factor = seg + 1 (as requested: 1 -> 2x, 2 -> 3x, 5 -> 6x, 10 -> 11x)
        seg_val = VAL_SEG[spin_segment]
        factor = seg_val + 1
        gain_brut = mise * factor * multiplicateur
    else:
        # bonus: bonus_mult provided (real), else use assumption
        bm = bonus_mult if bonus_mult is not None else st.session_state.bonus_mult_assumption
        gain_brut = mise * bm + mise  # (X * mise) + mise
        # If multiplicateur parameter used (Top Slot): apply multiplicateur to bonus effect
        if multiplicateur and multiplicateur != 1:
            # apply multiplicateur to the bonus part only, then add mise
            gain_brut = mise * (bm * multiplicateur) + mise
    gain_net = gain_brut - mise_totale
    return round(gain_brut, 4), round(gain_net, 4)

# ---------------------------
# Suggest best strategy (evaluates EV over STRATEGIES)
# returns dict with name, bets, ev
# ---------------------------
def suggest_strategy(full_history, bankroll, multiplicateur):
    # train ML on demand to get ml_pred
    if len(full_history) >= st.session_state.ml_window:
        train_ml()
    ml_pred, ml_probs = (None, {})
    if st.session_state.ml_trained:
        ml_pred, ml_probs = ml_predict_window()
    rtp100 = compute_rtp_last_n(100)
    best_name = None
    best_bets = {}
    best_ev = -1e12
    for name in STRATEGIES.keys():
        bets = compute_bets_for_strategy(name, bankroll, st.session_state.unit, rtp_last100=rtp100, ml_pred=ml_pred, ml_probs=ml_probs)
        ev = expected_value_for_strategy(bets, full_history, multiplicateur, bankroll)
        if ev > best_ev:
            best_ev = ev
            best_name = name
            best_bets = bets
    return {"name": best_name, "bets": best_bets, "ev": best_ev, "ml_pred": ml_pred, "ml_probs": ml_probs, "rtp100": rtp100}

# ---------------------------
# UI layout
# ---------------------------
st.title("🎰 Crazy Time — Bot avec suggestions & historique de stratégies")

col_l, col_r = st.columns([1,1])

with col_l:
    st.header("1) Historique manuel")
    st.write("Ajoute des spins manuellement (avant le live).")
    cols_per_row = 4
    rows = (len(SEGMENTS)+cols_per_row-1)//cols_per_row
    idx = 0
    for r in range(rows):
        cols = st.columns(cols_per_row)
        for c in range(cols_per_row):
            if idx >= len(SEGMENTS): break
            seg = SEGMENTS[idx]
            if cols[c].button(seg, key=f"hist_{seg}_{idx}"):
                st.session_state.history.append({"segment": seg, "mult": 1.0})
                st.success(f"Ajouté {seg} à l'historique manuel")
            idx += 1

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("↩ Supprimer dernier spin historique"):
            if st.session_state.history:
                popped = st.session_state.history.pop()
                st.warning(f"Supprimé: {popped}")
    with b2:
        if st.button("🔄 Réinitialiser historique manuel"):
            st.session_state.history = []
            st.success("Historique manuel vidé")
    with b3:
        if st.button("🏁 Fin historique et commencer"):
            # compute suggestion for next spin
            full_hist = full_history_segments()
            suggestion = suggest_strategy(full_hist, st.session_state.bankroll, st.session_state.mult_for_ev)
            # prepare strategy record (pre-filled; realized fields empty until spin occurs)
            next_spin_index = len(st.session_state.results_table) + 1
            rec = {
                "Spin #": next_spin_index,
                "Stratégie suggérée": suggestion["name"],
                "Mises $ (suggérées)": suggestion["bets"],
                "EV prévisionnel": round(suggestion["ev"],4),
                "Résultat réel": None,
                "Multiplier réel": None,
                "Gain Brut (réel)": None,
                "Gain Net (réel)": None,
                "Bankroll après": None
            }
            st.session_state.strategies_table.append(rec)
            st.session_state.last_suggestion = suggestion
            st.success(f"Suggestion calculée : {suggestion['name']} (EV {round(suggestion['ev'],4)})")

with col_r:
    st.header("2) Paramètres & ML")
    st.sidebar.header("Paramètres")
    st.session_state.initial_bankroll = st.sidebar.number_input("Bankroll initiale ($)", value=float(st.session_state.initial_bankroll))
    st.session_state.unit = st.sidebar.number_input("Unité de base ($)", value=float(st.session_state.unit))
    st.session_state.mult_for_ev = st.sidebar.number_input("Multiplicateur manuel (EV)", min_value=1, max_value=500, value=int(st.session_state.mult_for_ev))
    st.session_state.ml_window = st.sidebar.slider("Fenêtre ML (spins)", min_value=5, max_value=200, value=int(st.session_state.ml_window))
    st.session_state.ml_boost = st.sidebar.slider("ML boost (facteur)", min_value=1.0, max_value=3.0, value=float(st.session_state.ml_boost), step=0.1)
    st.session_state.rtp_weight_pct = st.sidebar.slider("Pondération RTP (%)", 0, 100, int(st.session_state.rtp_weight_pct))
    st.session_state.rtp_weight = st.session_state.rtp_weight_pct / 100.0
    st.sidebar.checkbox("Afficher tableau historique", value=st.session_state.show_history_table, key="show_history_table")
    st.write("ML entraîné :", st.session_state.ml_trained)
    if st.session_state.ml_trained:
        pseg, pprobs = ml_predict_window()
        st.write("Prédiction ML actuelle :", pseg)
        if pprobs:
            dfp = pd.DataFrame.from_dict(pprobs, orient='index', columns=['prob']).sort_values('prob', ascending=False)
            st.table((dfp*100).round(2))

st.markdown("---")

# ---------------------------
# Live spin area
# ---------------------------
st.header("3) Enregistrer spin live")
col1, col2, col3 = st.columns([1,1,1])
with col1:
    spin_val = st.selectbox("Résultat du spin (live) :", SEGMENTS)
with col2:
    mult_input = st.number_input("Multiplicateur (Top Slot / bonus) :", min_value=1.0, max_value=1000.0, value=1.0, step=1.0)
with col3:
    chosen_strategy_for_live = st.selectbox("Stratégie manuelle (override, sinon utilise suggestion) :", ["(use suggestion)"] + list(STRATEGIES.keys()))

if st.button("🎰 Enregistrer spin live"):
    # get bets: from suggestion if present, else compute best now
    if st.session_state.last_suggestion:
        suggested = st.session_state.last_suggestion
        chosen_name = suggested["name"]
        bets = suggested["bets"].copy()
    else:
        # compute suggestion on the fly
        full_hist = full_history_segments()
        sugg = suggest_strategy(full_hist, st.session_state.bankroll, st.session_state.mult_for_ev)
        chosen_name = sugg["name"]
        bets = sugg["bets"].copy()

    if chosen_strategy_for_live != "(use suggestion)":
        # override with selected strategy
        chosen_name = chosen_strategy_for_live
        rtp100 = compute_rtp_last_n(100)
        ml_pred, ml_probs = (None, {})
        if st.session_state.ml_trained:
            ml_pred, ml_probs = ml_predict_window()
        bets = compute_bets_for_strategy(chosen_name, st.session_state.bankroll, st.session_state.unit, rtp_last100=rtp100, ml_pred=ml_pred, ml_probs=ml_probs)

    # compute gain using provided multiplicateur for bonus or top slot
    # for bonus segments, multiplicateur passed as bonus_mult
    bonus_mult = None
    # if spin is a bonus segment, treat mult_input as the bonus multiplier
    if spin_val in BONUSES or spin_val == "Top Slot":
        bonus_mult = float(mult_input)

    gain_brut, gain_net = calcul_gain_from_bets(bets, spin_val, multiplicateur=float(mult_input), bonus_mult=bonus_mult)
    # update bankroll and results table
    st.session_state.bankroll = round(st.session_state.bankroll + gain_net, 4)
    rec = {
        "Spin #": len(st.session_state.results_table) + 1,
        "Stratégie": chosen_name,
        "Résultat": spin_val,
        "Multiplicateur": float(mult_input),
        "Mises $": bets,
        "Mise Totale": round(sum(bets.values()),4),
        "Gain Brut": gain_brut,
        "Gain Net": gain_net,
        "Bankroll": st.session_state.bankroll
    }
    st.session_state.results_table.append(rec)
    st.session_state.live_history.append({"segment": spin_val, "mult": float(mult_input)})

    # --- Update last strategies_table record (the one that was suggested for this spin) ---
    # find the last strategies_table entry with "Résultat réel" is None and spin index matches
    updated = False
    for srec in reversed(st.session_state.strategies_table):
        if srec.get("Résultat réel") is None:
            # fill it
            srec["Résultat réel"] = spin_val
            srec["Multiplier réel"] = float(mult_input)
            srec["Gain Brut (réel)"] = gain_brut
            srec["Gain Net (réel)"] = gain_net
            srec["Bankroll après"] = st.session_state.bankroll
            updated = True
            break
    if not updated:
        # No pre-suggestion row found — append a realized row
        srec = {
            "Spin #": rec["Spin #"],
            "Stratégie suggérée": chosen_name,
            "Mises $ (suggérées)": bets,
            "EV prévisionnel": None,
            "Résultat réel": spin_val,
            "Multiplier réel": float(mult_input),
            "Gain Brut (réel)": gain_brut,
            "Gain Net (réel)": gain_net,
            "Bankroll après": st.session_state.bankroll
        }
        st.session_state.strategies_table.append(srec)

    # update streaks
    bet_segments = [s for s,v in bets.items() if v>0]
    if spin_val in bet_segments:
        st.session_state.miss_streak = 0
    else:
        st.session_state.miss_streak += 1
    # martingale tracking
    if chosen_name == "Martingale 1":
        if gain_net > 0:
            st.session_state.martingale_loss_streak = 0
        else:
            st.session_state.martingale_loss_streak = st.session_state.get("martingale_loss_streak", 0) + 1

    st.success(f"Spin enregistré : {spin_val} x{mult_input} — Gain net: {gain_net} — Bankroll: {st.session_state.bankroll}")

    # --- Suggest next strategy automatically and append prefilled row ---
    full_hist = full_history_segments()
    suggestion_next = suggest_strategy(full_hist, st.session_state.bankroll, st.session_state.mult_for_ev)
    next_spin_index = len(st.session_state.results_table) + 1
    pre_rec = {
        "Spin #": next_spin_index,
        "Stratégie suggérée": suggestion_next["name"],
        "Mises $ (suggérées)": suggestion_next["bets"],
        "EV prévisionnel": round(suggestion_next["ev"],4),
        "Résultat réel": None,
        "Multiplier réel": None,
        "Gain Brut (réel)": None,
        "Gain Net (réel)": None,
        "Bankroll après": None
    }
    st.session_state.strategies_table.append(pre_rec)
    st.session_state.last_suggestion = suggestion_next

# ---------------------------
# Display probabilities & EV table
# ---------------------------
st.markdown("---")
st.subheader("Probabilités & EV (basé sur historique + live)")
full_hist = full_history_segments()
ev_rows = []
for s in SEGMENTS:
    p = combined_prob(full_hist, s, window=st.session_state.ml_window)
    bet = {seg: 0 for seg in SEGMENTS}; bet[s] = 1.0
    ev = expected_value_for_strategy(bet, full_hist, st.session_state.mult_for_ev, st.session_state.bankroll)
    ev_rows.append({"Segment": s, "Probabilité (%)": round(p*100,3), "EV (1$ mise)": round(ev,4)})
st.table(pd.DataFrame(ev_rows))

# ---------------------------
# Results table (live) + strategies table
# ---------------------------
st.markdown("---")
st.subheader("Historique des spins (live enregistrés)")
if st.session_state.results_table:
    df_results = pd.DataFrame(st.session_state.results_table)
    st.dataframe(df_results, use_container_width=True)
    # Bankroll chart
    fig, ax = plt.subplots()
    ax.plot(df_results["Spin #"], df_results["Bankroll"], marker='o')
    ax.axhline(y=st.session_state.initial_bankroll, color='gray', linestyle='--')
    ax.set_xlabel("Spin #"); ax.set_ylabel("Bankroll ($)")
    ax.grid(True)
    st.pyplot(fig)
else:
    st.info("Aucun spin live enregistré.")

st.markdown("---")
st.subheader("Historique des stratégies suggérées & résultats")
if st.session_state.strategies_table:
    df_strat = pd.DataFrame(st.session_state.strategies_table)
    st.dataframe(df_strat, use_container_width=True)
    # Optionally show a small performance summary
    realized = [r for r in st.session_state.strategies_table if r.get("Gain Net (réel)") is not None]
    if realized:
        total_real_gain = sum([r.get("Gain Net (réel)",0) for r in realized])
        st.write(f"Gain net total (stratégies réalisées) : {round(total_real_gain,4)}")
else:
    st.info("Aucune stratégie suggérée encore.")

# ---------------------------
# Simulation section (unchanged: simulate chosen strategy over full history starting from initial bankroll)
# ---------------------------
st.markdown("---")
st.subheader("Simulation — tester une stratégie sur toute la session (historique + live)")
sim_choice = st.selectbox("Choisir stratégie pour la simulation :", list(STRATEGIES.keys()))
if st.button("▶️ Lancer simulation (sur bankroll initiale)"):
    entries = []
    for h in st.session_state.history:
        seg, mult = normalize_entry(h)
        entries.append((seg, float(mult)))
    for h in st.session_state.live_history:
        seg, mult = normalize_entry(h)
        entries.append((seg, float(mult)))
    if not entries:
        st.warning("Aucun historique pour simuler.")
    else:
        sim_bank = float(st.session_state.initial_bankroll)
        sim_records = []
        sim_martingale = 0
        for i, (seg, mult) in enumerate(entries, start=1):
            rtp100 = compute_rtp_last_n(100)
            ml_pred, ml_probs = (None, {})
            if st.session_state.ml_trained:
                ml_pred, ml_probs = ml_predict_window()
            bets = compute_bets_for_strategy(sim_choice, sim_bank, st.session_state.unit, rtp_last100=rtp100, ml_pred=ml_pred, ml_probs=ml_probs)
            mise_tot = sum(bets.values())
            gain_brut, gain_net = calcul_gain_from_bets(bets, seg, multiplicateur=float(mult), bonus_mult=None)
            sim_bank = round(sim_bank + gain_net, 4)
            sim_records.append({
                "Spin #": i,
                "Segment": seg,
                "Mult": mult,
                "Mises $": bets,
                "Mise Totale": round(mise_tot,4),
                "Gain Brut": gain_brut,
                "Gain Net": gain_net,
                "Bankroll": sim_bank
            })
            if sim_choice == "Martingale 1":
                if gain_net > 0:
                    sim_martingale = 0
                else:
                    sim_martingale += 1
        df_sim = pd.DataFrame(sim_records)
        st.dataframe(df_sim, use_container_width=True)
        fig_s, ax_s = plt.subplots()
        ax_s.plot(df_sim["Spin #"], df_sim["Bankroll"], marker='o')
        ax_s.axhline(y=st.session_state.initial_bankroll, color='gray', linestyle='--')
        ax_s.set_xlabel("Spin #"); ax_s.set_ylabel("Bankroll simulée ($)")
        ax_s.grid(True)
        st.pyplot(fig_s)
        st.success(f"Simulation terminée — bankroll finale simulée : {round(sim_bank,4)}")

# End of file
