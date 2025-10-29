# 🎰 CRAZY TIME BOT — version complète (RTP + ML + une stratégie + top slot intégré)
import streamlit as st
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# --------------------------------------------------
# ⚙️ CONFIG
# --------------------------------------------------
st.set_page_config(page_title="🎰 Crazy Time Bot", layout="wide")

# --------------------------------------------------
# 🧠 INITIALISATION DES VARIABLES
# --------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "bankroll" not in st.session_state:
    st.session_state.bankroll = 200.0
if "unit" not in st.session_state:
    st.session_state.unit = 1.0
if "strategy" not in st.session_state:
    st.session_state.strategy = "1+Bonus"
if "model" not in st.session_state:
    st.session_state.model = RandomForestClassifier()
if "trained" not in st.session_state:
    st.session_state.trained = False
if "rtp_weight" not in st.session_state:
    st.session_state.rtp_weight = 95.0  # valeur par défaut
if "ml_window" not in st.session_state:
    st.session_state.ml_window = 50  # fenêtre ML (nombre de spins récents à analyser)

# --------------------------------------------------
# 🧮 Normalisation du RTP (fix du bug)
# --------------------------------------------------
if st.session_state.rtp_weight > 1:
    st.session_state.rtp_weight /= 100.0

# --------------------------------------------------
# 🎛️ PARAMÈTRES PERSONNALISABLES
# --------------------------------------------------
st.sidebar.header("⚙️ Paramètres")
st.session_state.bankroll = st.sidebar.number_input("💰 Bankroll initiale", 50.0, 10000.0, st.session_state.bankroll)
st.session_state.unit = st.sidebar.number_input("🎯 Unité de base ($)", 0.1, 100.0, st.session_state.unit)
st.session_state.rtp_weight = st.sidebar.slider("📊 Pondération RTP (%)", 80, 100, int(st.session_state.rtp_weight * 100)) / 100.0
st.session_state.ml_window = st.sidebar.slider("🧠 Fenêtre ML (spins récents)", 10, 200, st.session_state.ml_window)

# --------------------------------------------------
# 🎡 SEGMENTS ET MULTIPLICATEURS
# --------------------------------------------------
SEGMENTS = ["1", "2", "5", "10", "Coin Flip", "Cash Hunt", "Pachinko", "Crazy Time"]
BONUSES = ["Coin Flip", "Cash Hunt", "Pachinko", "Crazy Time"]

# --------------------------------------------------
# 🧮 CALCUL PROBABILITÉS PONDÉRÉES
# --------------------------------------------------
def compute_segment_probabilities():
    base_probs = {"1": 21, "2": 13, "5": 7, "10": 4, "Coin Flip": 4, "Cash Hunt": 2, "Pachinko": 2, "Crazy Time": 1}
    total = sum(base_probs.values())
    probs = {k: v / total for k, v in base_probs.items()}

    # Influence de l’historique
    if len(st.session_state.history) > 0:
        recent = [x["segment"] for x in st.session_state.history[-st.session_state.ml_window:]]
        df = pd.DataFrame(recent, columns=["segment"])
        freq = df["segment"].value_counts(normalize=True).to_dict()
        for seg in probs:
            if seg in freq:
                probs[seg] = (probs[seg] + freq[seg] * st.session_state.rtp_weight) / (1 + st.session_state.rtp_weight)

    return probs

# --------------------------------------------------
# 🧠 MACHINE LEARNING (RandomForest)
# --------------------------------------------------
def train_model():
    if len(st.session_state.history) < 10:
        return
    df = pd.DataFrame(st.session_state.history)
    df["target"] = df["segment"].astype("category").cat.codes
    X = df.index.values.reshape(-1, 1)
    y = df["target"]
    st.session_state.model.fit(X, y)
    st.session_state.trained = True

def predict_next_segment():
    if not st.session_state.trained:
        return random.choice(SEGMENTS)
    X_pred = [[len(st.session_state.history)]]
    pred_idx = st.session_state.model.predict(X_pred)[0]
    mapping = pd.Series(pd.Categorical(st.session_state.history[-1]["segment"]).categories)
    return mapping[pred_idx] if pred_idx < len(mapping) else random.choice(SEGMENTS)

# --------------------------------------------------
# 🎯 CALCUL DES MISES (UNE SEULE STRATÉGIE)
# --------------------------------------------------
def compute_bets(probs):
    bets = {seg: 0 for seg in SEGMENTS}
    base = st.session_state.unit
    strat = st.session_state.strategy

    if strat == "1+Bonus":
        bets["1"] = base
        for b in BONUSES:
            bets[b] = base
    elif strat == "Martingale 1":
        bets["1"] = base * 2
    elif strat == "God Mode":
        bets["2"] = base
        bets["5"] = base
        bets["10"] = base
    elif strat == "Only Numbers":
        for s in ["1", "2", "5", "10"]:
            bets[s] = base
    elif strat == "All but 1":
        for s in SEGMENTS:
            if s != "1":
                bets[s] = base

    # Ajustement selon RTP
    for s in bets:
        bets[s] *= st.session_state.rtp_weight

    return bets

# --------------------------------------------------
# 🎮 TRAITEMENT SPIN LIVE
# --------------------------------------------------
def process_spin(segment, multiplier=1):
    bets = compute_bets(compute_segment_probabilities())
    total_bet = sum(bets.values())
    win = 0
    if segment in bets:
        if segment in ["1", "2", "5", "10"]:
            win = bets[segment] * int(segment) * multiplier
        else:
            win = bets[segment] * 25 * multiplier  # hypothèse bonus moyen
    net = win - total_bet
    st.session_state.bankroll += net
    st.session_state.history.append({"segment": segment, "mult": multiplier, "bet": total_bet, "win": win, "net": net, "bankroll": st.session_state.bankroll})

# --------------------------------------------------
# 📊 AFFICHAGE
# --------------------------------------------------
st.title("🎰 Crazy Time Bot — ML + RTP + Top Slot")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎡 Entrée manuelle du spin")
    for seg in SEGMENTS:
        if st.button(seg):
            mult = st.number_input("Multiplicateur Top Slot", 1, 100, 1, key=f"mult_{seg}")
            process_spin(seg, mult)
            st.success(f"Spin ajouté : {seg} ×{mult}")

    if st.button("🧠 Fin historique et commencer"):
        train_model()
        st.success("Modèle ML entraîné — Prédictions activées ✅")

with col2:
    st.subheader("📊 Prochain spin (prévision ML)")
    probs = compute_segment_probabilities()
    next_pred = predict_next_segment()
    st.write("🎯 Segment prédit :", next_pred)
    st.write(pd.DataFrame.from_dict(probs, orient="index", columns=["Probabilité"]).style.background_gradient(cmap="YlOrRd"))

# --------------------------------------------------
# 📈 TABLEAU + GRAPHIQUE
# --------------------------------------------------
if len(st.session_state.history) > 0:
    df = pd.DataFrame(st.session_state.history)
    st.subheader("📋 Historique des spins")
    st.dataframe(df)

    plt.figure(figsize=(8, 3))
    plt.plot(df["bankroll"], marker="o")
    plt.title("Évolution bankroll")
    plt.xlabel("Spin #")
    plt.ylabel("💰 Bankroll")
    st.pyplot(plt)

# --------------------------------------------------
# 🧪 SIMULATION BATCH
# --------------------------------------------------
st.subheader("🧪 Simulation d’une stratégie")
if st.button("Lancer la simulation"):
    temp_bankroll = st.session_state.bankroll
    for i in range(20):
        seg = random.choice(SEGMENTS)
        process_spin(seg)
    st.success("Simulation terminée ✅")
