# streamlit_app_final_ml_rtp.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# -----------------------------------
# 🔧 CONFIG INITIALE
# -----------------------------------
st.set_page_config(page_title="🎰 Crazy Time Tracker + ML + RTP", layout="wide")

VAL_SEG = {'1':1,'2':2,'5':5,'10':10}
SEGMENTS = ['1','2','5','10','Coin Flip','Cash Hunt','Pachinko','Crazy Time']
THEO_COUNTS = {'1':21,'2':13,'5':7,'10':4,'Coin Flip':4,'Cash Hunt':2,'Pachinko':2,'Crazy Time':1}
THEO_TOTAL = sum(THEO_COUNTS.values())

# Baselines fournis
RTP_BASE_LOW = 94.41
RTP_BASE_HIGH = 96.08

# -----------------------------------
# ⚙️ INIT session_state
# -----------------------------------
for key, default in [
    ("bankroll",150.0),
    ("initial_bankroll",150.0),
    ("live_history",[]),
    ("history",[]),
    ("results_table",[]),   # stocke dicts détaillés (incl. "Mise Totale", "Gain Brut")
    ("martingale_1_loss_streak",0),
    ("miss_streak",0),
    ("last_suggestion_name",None),
    ("last_suggestion_mises",{}),
    ("bonus_multiplier_assumption",10),
    ("mult_for_ev",1),
    ("ml_model",None),
    ("label_encoder",None),
    ("show_history_table", True),
]:
    if key not in st.session_state:
        st.session_state[key] = default

MODEL_FILE = "crazytime_model.pkl"
ML_WINDOW = 5   # fenêtre séquence utilisée par le ML

# -----------------------------------
# 🧠 UTILITAIRES PROBABILITÉS / EV / RTP
# -----------------------------------
def theo_prob(segment):
    return THEO_COUNTS.get(segment,0)/THEO_TOTAL

def hist_prob(full_history, segment, window=300):
    if not full_history: return 0.0
    hist = full_history[-window:]
    return hist.count(segment)/len(hist)

def combined_prob(full_history, segment, window=300):
    return 0.5*(theo_prob(segment)+hist_prob(full_history, segment, window=window))

def expected_value_for_strategy(mises, full_history, multiplicateur, bankroll):
    mise_totale = sum(mises.values()) if mises else 0.0
    ev = 0.0
    for seg in SEGMENTS:
        p = combined_prob(full_history, seg)
        if seg in mises and mises[seg] > 0:
            seg_val = VAL_SEG.get(seg, st.session_state.bonus_multiplier_assumption)
            payout = mises[seg] * (seg_val * multiplicateur) + mises[seg]
            net_if_hit = payout - mise_totale
        else:
            net_if_hit = -mise_totale
        ev += p * net_if_hit
    return ev

def compute_rtp_last_n(n=100):
    """Calculates RTP (%) over the last n spins using results_table entries.
       RTP = (total returned to player / total wagered) * 100.
       Requires entries to include 'Gain Brut' and 'Mise Totale' (we store those on spin)."""
    tbl = st.session_state.results_table[-n:]
    if not tbl:
        return None
    total_returned = 0.0
    total_wagered = 0.0
    for entry in tbl:
        # If keys missing, try to infer
        gain_brut = entry.get("Gain Brut", 0.0)
        mise_totale = entry.get("Mise Totale", 0.0)
        # If mise_totale 0 but Mises $ exists, sum it
        if mise_totale == 0.0 and entry.get("Mises $"):
            mise_totale = sum(entry["Mises $"].values()) if isinstance(entry["Mises $"], dict) else 0.0
        total_returned += float(gain_brut)
        total_wagered += float(mise_totale)
    if total_wagered == 0:
        return None
    return (total_returned / total_wagered) * 100.0

# -----------------------------------
# 🎯 STRATÉGIES
# -----------------------------------
def strategy_martingale_1(bankroll, loss_streak):
    base_bet = 4.0
    mise_1 = base_bet * (2 ** loss_streak)
    return "Martingale 1", {'1': mise_1}

def strategy_god_mode(bankroll):
    return "God Mode", {'2':3.0,'5':2.0,'10':1.0}

def strategy_god_mode_bonus(bankroll):
    return "God Mode + Bonus", {'2':3.0,'5':2.0,'10':1.0,'Coin Flip':1.0,'Cash Hunt':1.0,'Pachinko':1.0,'Crazy Time':1.0}

def strategy_1_bonus(bankroll):
    return "1 + Bonus", {'1':4.0,'Coin Flip':1.0,'Cash Hunt':1.0,'Pachinko':1.0,'Crazy Time':1.0}

def strategy_only_numbers(bankroll):
    return "Only Numbers", {'1':3.0,'2':2.0,'5':1.0,'10':1.0}

# -----------------------------------
# 🧠 ML — préparation, entraînement auto (option 1), prédiction
# -----------------------------------
def prepare_ml_data(history, rtp_feature, window=ML_WINDOW):
    """Return X (list of vectors) and y (labels). X vectors = encoded window + [rtp_feature].
       history is list of segment strings."""
    X, y = [], []
    if len(history) <= window:
        return X, y
    le = LabelEncoder()
    le.fit(SEGMENTS)
    st.session_state.label_encoder = le
    encoded = le.transform(history)
    for i in range(len(encoded) - window):
        seq = list(encoded[i:i+window])
        # append rtp_feature (same for all samples) as normalized float (rtp/100)
        seq.append(float(rtp_feature)/100.0 if rtp_feature is not None else 0.0)
        X.append(seq)
        y.append(encoded[i+window])
    return X, y

def train_ml_model_auto():
    history = st.session_state.history + st.session_state.live_history
    # prefer to use full history as requested (option 2 earlier), but here user chose option1 (auto-train at launch)
    # The user requested "2" (use full history). We'll use full history (already composed).
    rtp_calc = compute_rtp_last_n(100)
    # If user provided manual override, use it (we will display input later)
    rtp_input = st.session_state.get("rtp_last100_manual", None)
    rtp_feature = rtp_input if rtp_input is not None else (rtp_calc if rtp_calc is not None else (RTP_BASE_LOW+RTP_BASE_HIGH)/2.0)
    X, y = prepare_ml_data(history, rtp_feature, window=ML_WINDOW)
    if not X:
        st.session_state.ml_model = None
        return None
    X_arr = np.array(X)
    y_arr = np.array(y)
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_arr, y_arr)
    st.session_state.ml_model = model
    # store label encoder already in session_state
    return model

def predict_next_segment_ml(rtp_feature):
    """Return predicted segment and probability + per-segment probs dict."""
    if st.session_state.ml_model is None or st.session_state.label_encoder is None:
        return None, 0.0, {}
    history = st.session_state.history + st.session_state.live_history
    if len(history) < ML_WINDOW:
        return None, 0.0, {}
    le = st.session_state.label_encoder
    encoded = le.transform(history[-ML_WINDOW:])
    vect = list(encoded) + [float(rtp_feature)/100.0 if rtp_feature is not None else 0.0]
    probs = st.session_state.ml_model.predict_proba([vect])[0]
    # map probs to segment names (label_encoder.classes_ gives in order of encoded indices)
    classes = st.session_state.label_encoder.inverse_transform(np.arange(len(st.session_state.label_encoder.classes_)))
    per_seg = {seg: float(probs[idx]) for idx, seg in enumerate(st.session_state.label_encoder.inverse_transform(np.arange(len(probs))))}
    pred_idx = np.argmax(probs)
    pred_seg = st.session_state.label_encoder.inverse_transform([pred_idx])[0]
    return pred_seg, float(probs[pred_idx]), per_seg

# -----------------------------------
# 🧠 STRATÉGIE : combine ML prediction + RTP adjustments + EV
# -----------------------------------
def adjust_mises_by_rtp_and_ml(mises, pred_seg, pred_prob, rtp_last100):
    """Modify mises dict in-place to reflect ML and RTP phases."""
    # If no RTP given -> no adjustment
    rtp = rtp_last100 if rtp_last100 is not None else (RTP_BASE_LOW+RTP_BASE_HIGH)/2.0
    # If cold (below baseline low) -> favor bonuses
    if rtp < RTP_BASE_LOW:
        # increase bonus segment stakes
        for b in ['Coin Flip','Cash Hunt','Pachinko','Crazy Time']:
            if b in mises:
                mises[b] = mises.get(b,0.0)*1.6
    # If hot -> favor numbers
    if rtp > RTP_BASE_HIGH:
        for n in ['1','2','5','10']:
            if n in mises:
                mises[n] = mises.get(n,0.0)*1.3
    # ML boost on predicted segment
    if pred_seg and pred_seg in mises:
        mises[pred_seg] = mises.get(pred_seg,0.0) * (1.0 + pred_prob)  # stronger boost if prob high
    return mises

def choose_strategy_intelligent(full_history, bankroll, multiplicateur, rtp_last100):
    # martingale logic priority
    if st.session_state.martingale_1_loss_streak > 0:
        return strategy_martingale_1(bankroll, st.session_state.martingale_1_loss_streak)
    if st.session_state.miss_streak >= 3:
        return strategy_martingale_1(bankroll, 0)

    # ML prediction (if model exists)
    pred_seg, pred_prob, per_seg = predict_next_segment_ml(rtp_last100)

    candidates = []
    for builder in [strategy_only_numbers, strategy_god_mode, strategy_god_mode_bonus, strategy_1_bonus]:
        name, mises = builder(bankroll)
        # make a shallow copy so original builder not mutated
        mises_copy = {k: float(v) for k,v in mises.items()}
        mises_adj = adjust_mises_by_rtp_and_ml(mises_copy, pred_seg, pred_prob, rtp_last100)
        ev = expected_value_for_strategy(mises_adj, full_history, multiplicateur, bankroll)
        candidates.append((name, mises_adj, ev))
    best = max(candidates, key=lambda x: x[2])
    return best[0], best[1], pred_seg, pred_prob, per_seg

# -----------------------------------
# 🔁 AUTO-TRAIN ML AU LANCEMENT (Option 1)
# -----------------------------------
# compute RTP from last 100 spins if possible
rtp_from_table = compute_rtp_last_n(100)
# input override for RTP (manual)
st.sidebar.header("RTP & paramètres")
manual_rtp = st.sidebar.number_input("🎰 RTP des 100 derniers spins (%) — saisie (laisser 0 pour utiliser calcul automatique)", min_value=0.0, max_value=100.0, value=float(round(rtp_from_table or 0.0,2)))
if manual_rtp > 0.0:
    st.session_state["rtp_last100_manual"] = manual_rtp
else:
    st.session_state["rtp_last100_manual"] = None

# Ensure we have at least something; train if history exists
if (st.session_state.history or st.session_state.live_history):
    train_ml_model_auto()

# -----------------------------------
# 💰 CALCUL GAIN (enregistr. spin)
# -----------------------------------
def calcul_gain(mises, spin_result, multiplicateur):
    if not mises:
        return 0.0, 0.0
    mise_totale = sum(mises.values())
    gain_brut = 0.0
    if spin_result in mises and mises[spin_result] > 0:
        seg_val = VAL_SEG.get(spin_result, st.session_state.bonus_multiplier_assumption)
        gain_brut = (mises[spin_result] * (seg_val * multiplicateur)) + mises[spin_result]
    gain_net = gain_brut - mise_totale
    return float(gain_brut), float(gain_net), float(mise_totale)

# -----------------------------------
# 🧾 AFFICHAGE STRATÉGIE & ML
# -----------------------------------
def display_next_suggestion(pred_seg=None, pred_prob=None, per_seg_probs=None):
    st.subheader("🎯 Prochaine stratégie suggérée")
    if st.session_state.last_suggestion_name and st.session_state.last_suggestion_mises:
        st.write(f"**Stratégie :** {st.session_state.last_suggestion_name}")
        st.table(pd.DataFrame.from_dict(st.session_state.last_suggestion_mises, orient='index', columns=['Mise $']))
    else:
        st.write("Aucune stratégie suggérée pour l’instant.")
    # ML prediction box
    if pred_seg:
        st.info(f"💡 ML Prediction: **{pred_seg}** — probabilité {round(pred_prob*100,1)}%")
    # show full per-segment probs if provided
    if per_seg_probs:
        df_ml = pd.DataFrame([{"Segment":k,"Probabilité (%)":round(v*100,2)} for k,v in per_seg_probs.items()]).sort_values("Probabilité (%)", ascending=False)
        st.subheader("🔎 Probabilités ML par segment")
        st.table(df_ml)

# -----------------------------------
# 🔘 HISTORIQUE MANUEL + affichage avant spins live
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
segment_buttons_grid(SEGMENTS)

col_a, col_b, col_c = st.columns([1,1,1])
with col_a:
    if st.button("↩ Supprimer dernier spin historique") and st.session_state.history:
        st.session_state.history.pop()
with col_b:
    if st.button("🔄 Réinitialiser historique manuel"):
        st.session_state.history=[]
with col_c:
    if st.button("🏁 Terminer historique"):
        full_history = st.session_state.history + st.session_state.live_history
        # decide rtp last100 to feed to strategy
        rtp_last100 = st.session_state.get("rtp_last100_manual") or compute_rtp_last_n(100) or (RTP_BASE_LOW+RTP_BASE_HIGH)/2.0
        next_name, next_mises, pred_seg, pred_prob, per_seg = choose_strategy_intelligent(full_history, st.session_state.bankroll, st.session_state["mult_for_ev"], rtp_last100)
        st.session_state.last_suggestion_name = next_name
        st.session_state.last_suggestion_mises = next_mises
        display_next_suggestion(pred_seg, pred_prob, per_seg)

# Affichage tableau historique manuel avant spins live
if st.session_state.show_history_table and st.session_state.history:
    st.subheader("📋 Historique Manuel Actuel")
    df_manual = pd.DataFrame({
        "#": range(1, len(st.session_state.history)+1),
        "Résultat": st.session_state.history
    })
    st.dataframe(df_manual, use_container_width=True)

# -----------------------------------
# Sidebar paramètres supplémentaires
# -----------------------------------
st.sidebar.markdown("---")
st.sidebar.write(f"RTP calculé (last100) : **{round(rtp_from_table,2) if rtp_from_table else 'N/A'}**")
st.sidebar.write(f"Baselines RTP : Low={RTP_BASE_LOW}%, High={RTP_BASE_HIGH}%")
mult_for_ev_input = st.sidebar.number_input("Multiplicateur manuel (pour EV)", min_value=1, max_value=200, value=st.session_state["mult_for_ev"], step=1)
st.session_state["mult_for_ev"] = mult_for_ev_input
bonus_ass = st.sidebar.number_input("Hypothèse multiplicateur bonus (EV)", min_value=1, max_value=1000, value=st.session_state.bonus_multiplier_assumption, step=1)
st.session_state.bonus_multiplier_assumption = int(bonus_ass)
st.sidebar.checkbox("Afficher tableau historique", value=st.session_state.show_history_table, key="show_history_table")

# -----------------------------------
# 🧮 SPINS LIVE + ENREGISTREMENT (avec sauvegarde Mise Totale)
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
        gain_brut, gain_net, mise_total = calcul_gain(mises_for_spin, spin_val, multiplicateur)
        new_bankroll = st.session_state.bankroll + gain_net
        st.session_state.bankroll = new_bankroll
        # store spin with Mise Totale
        st.session_state.live_history.append(spin_val)
        st.session_state.results_table.append({
            "Spin #": len(st.session_state.results_table) + 1,
            "Stratégie": strategy_name,
            "Résultat": spin_val,
            "Multiplicateur": multiplicateur,
            "Mises $": {k: round(v,2) for k,v in (mises_for_spin or {}).items()},
            "Mise Totale": round(mise_total,2),
            "Gain Brut": round(gain_brut,2),
            "Gain Net": round(gain_net,2),
            "Bankroll": round(new_bankroll,2)
        })

        # miss_streak logic
        bet_segments = [s for s,v in (mises_for_spin or {}).items() if v and v>0]
        if not bet_segments or spin_val not in bet_segments:
            st.session_state.miss_streak += 1
        else:
            st.session_state.miss_streak = 0

        # martingale handling
        if strategy_name == "Martingale 1":
            if gain_net > 0:
                st.session_state.martingale_1_loss_streak = 0
                st.session_state.miss_streak = 0
            else:
                st.session_state.martingale_1_loss_streak += 1

        # Next suggestion using fresh data
        full_history = st.session_state.history + st.session_state.live_history
        rtp_last100 = st.session_state.get("rtp_last100_manual") or compute_rtp_last_n(100) or (RTP_BASE_LOW+RTP_BASE_HIGH)/2.0
        next_name, next_mises, pred_seg, pred_prob, per_seg = choose_strategy_intelligent(full_history, st.session_state.bankroll, st.session_state["mult_for_ev"], rtp_last100)
        st.session_state.last_suggestion_name = next_name
        st.session_state.last_suggestion_mises = next_mises

        # retrain ML on the new full history automatically (auto-train each launch; retrain here keeps model up-to-date)
        train_ml_model_auto()

        display_next_suggestion(pred_seg, pred_prob, per_seg)
        st.success(f"Spin #{len(st.session_state.results_table)} enregistré : {spin_val} x{multiplicateur} — Gain net: {round(gain_net,2)}$ — Bankroll: {round(new_bankroll,2)}$")

# -----------------------------------
# 🔎 EV / Probabilités & Table augmentée live (inclut prob théorique/observée + EV + RTP)
# -----------------------------------
st.subheader("📊 Probabilités & EV pour tous les segments (live)")
full_history = st.session_state.history + st.session_state.live_history
ev_table = []
# compute observed probs over last 100 for display
last100 = full_history[-100:] if full_history else []
for seg in SEGMENTS:
    p_theo = theo_prob(seg)
    p_obs = (last100.count(seg) / len(last100)) if last100 else 0.0
    # EV for 1$ on this segment
    mises = {seg:1.0}
    ev = expected_value_for_strategy(mises, full_history, multiplicateur, st.session_state.bankroll)
    ev_table.append({
        "Segment": seg,
        "Prob Théorique (%)": round(p_theo*100,3),
        "Prob Observée (%)": round(p_obs*100,3),
        "EV (1$ mise)": round(ev,3)
    })
st.table(pd.DataFrame(ev_table))

# -----------------------------------
# 📈 Historique Spins Live enrichi + graphique (affiché si box cochée)
# -----------------------------------
if st.session_state.show_history_table and st.session_state.results_table:
    st.subheader("📈 Historique des Spins Live (enrichi)")
    # enrich the results table with prob_theo, prob_obs, ev_at_spin using the history until the spin (up to that index)
    enriched = []
    for i, entry in enumerate(st.session_state.results_table):
        # history until just before this spin
        hist_before = (st.session_state.history + [r["Résultat"] for r in st.session_state.results_table[:i]])
        p_theo = theo_prob(entry["Résultat"])
        p_obs = (hist_before.count(entry["Résultat"]) / len(hist_before)) if hist_before else 0.0
        # EV for a 1$ bet on that segment at that time
        ev = expected_value_for_strategy({entry["Résultat"]:1.0}, hist_before, entry.get("Multiplicateur",1), st.session_state.bankroll)
        enriched.append({
            "Spin #": entry["Spin #"],
            "Résultat": entry["Résultat"],
            "Prob Théorique (%)": round(p_theo*100,3),
            "Prob Observée (%)": round(p_obs*100,3),
            "EV (1$)": round(ev,3),
            "Mise Totale": entry.get("Mise Totale", 0.0),
            "Gain Net": entry.get("Gain Net", 0.0),
            "Bankroll": entry.get("Bankroll", 0.0),
            "Stratégie": entry.get("Stratégie", "")
        })
    df_enriched = pd.DataFrame(enriched)
    st.dataframe(df_enriched, use_container_width=True)

    st.subheader("💹 Évolution de la Bankroll")
    fig, ax = plt.subplots()
    ax.plot(df_enriched["Spin #"], df_enriched["Bankroll"], marker='o', label='Bankroll')
    ax.axhline(y=st.session_state.initial_bankroll, color='gray', linestyle='--', label='Bankroll initiale')
    ax.set_xlabel("Spin #")
    ax.set_ylabel("Bankroll ($)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# -----------------------------------
# 🔁 Simulation du bot sur l’historique (optionnel)
# -----------------------------------
if st.button("🧠 Appliquer le bot sur l’historique"):
    if not st.session_state.history:
        st.warning("⚠️ Ajoute d’abord des spins manuels avant d’appliquer le bot.")
    else:
        bankroll_sim = st.session_state.initial_bankroll
        simulated_results = []
        for i, spin_result in enumerate(st.session_state.history, start=1):
            full_history_temp = st.session_state.history[:i-1]
            rtp_temp = compute_rtp_last_n(100) or (st.session_state.get("rtp_last100_manual") or (RTP_BASE_LOW+RTP_BASE_HIGH)/2.0)
            strategy_name, mises, pred_seg, pred_prob, per_seg = choose_strategy_intelligent(full_history_temp, bankroll_sim, st.session_state["mult_for_ev"], rtp_temp)
            gain_brut, gain_net, mise_total = calcul_gain(mises, spin_result, 1)
            bankroll_sim += gain_net
            simulated_results.append({"Spin #": i, "Résultat": spin_result, "Stratégie": strategy_name, "Gain Net": round(gain_net,2), "Bankroll": round(bankroll_sim,2)})
        df_sim = pd.DataFrame(simulated_results)
        st.subheader("📊 Résultats du bot sur l’historique")
        st.dataframe(df_sim, use_container_width=True)
        st.success(f"✅ Simulation terminée — Bankroll finale : {round(bankroll_sim,2)}$")

# End of file
