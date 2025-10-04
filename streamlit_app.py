import streamlit as st
import random
import matplotlib.pyplot as plt

# =============================
# CONFIG INITIAL
# =============================
st.set_page_config(page_title="🎰 Super Crazy Time Bot", layout="wide")

# =============================
# INIT SESSION STATE
# =============================
if "bankroll" not in st.session_state:
    st.session_state.bankroll = 100
if "initial_bankroll" not in st.session_state:
    st.session_state.initial_bankroll = 100
if "history" not in st.session_state:
    st.session_state.history = []
if "last_suggestion_name" not in st.session_state:
    st.session_state.last_suggestion_name = None
if "last_suggestion_mises" not in st.session_state:
    st.session_state.last_suggestion_mises = None
if "results" not in st.session_state:
    st.session_state.results = []

# =============================
# SEGMENTS WHEEL
# =============================
segments = {
    "1": 21, "2": 13, "5": 7, "10": 4,
    "Coin Flip": 4, "Cash Hunt": 2, "Pachinko": 2, "Crazy Time": 1
}
total_segments = sum(segments.values())
probabilities = {k: v/total_segments for k, v in segments.items()}

# =============================
# STRATEGIES
# =============================
def get_strategies(bankroll):
    return {
        "God Mode": {"2": 3, "5": 2, "10": 1},
        "God Mode + Bonus": {"2": 3, "5": 2, "10": 1, "Coin Flip": 1, "Cash Hunt": 1, "Pachinko": 1, "Crazy Time": 1},
        "1 + Bonus": {"1": 4, "Coin Flip": 1, "Cash Hunt": 1, "Pachinko": 1, "Crazy Time": 1},
        "Martingale 1": {"1": 2}, 
        "No Bets": {}
    }

# Choix intelligent avec variation
def choose_strategy_intelligent(history, bankroll):
    strategies = get_strategies(bankroll)
    last_20 = history[-20:] if len(history) >= 20 else history
    counts = {k: last_20.count(k) for k in segments.keys()}

    # Favoriser bonus absents
    missing_bonus = [b for b in ["Coin Flip","Cash Hunt","Pachinko","Crazy Time"] if counts.get(b,0) == 0]
    if missing_bonus:
        return "1 + Bonus", strategies["1 + Bonus"]

    # Si beaucoup de "1"
    if counts.get("1", 0) > len(last_20) * 0.5:
        return "God Mode + Bonus", strategies["God Mode + Bonus"]

    # Sinon God Mode
    return "God Mode", strategies["God Mode"]

# =============================
# BANQUE / MISES
# =============================
def adjust_bets(strategy_dict, bankroll, initial_bankroll):
    factor = 1.0
    if bankroll >= 2.5 * initial_bankroll:
        factor = 1.5
    elif bankroll <= 0.5 * initial_bankroll:
        factor = 0.5
    return {k: v*factor for k,v in strategy_dict.items()}

def apply_spin_result(spin, mises, bankroll):
    gain = 0
    for seg, mise in mises.items():
        if spin == seg:
            if seg in ["1","2","5","10"]:
                gain += mise * (int(seg))
            else:
                gain += mise * 10  # simplification bonus
    bankroll = bankroll - sum(mises.values()) + gain
    return bankroll, gain

# =============================
# UI
# =============================
st.title("🎰 Super Crazy Time Bot")

st.sidebar.header("⚙️ Paramètres")
bankroll_input = st.sidebar.number_input("💰 Bankroll initiale", min_value=10, value=100, step=10)
if st.sidebar.button("🔄 Réinitialiser"):
    st.session_state.bankroll = bankroll_input
    st.session_state.initial_bankroll = bankroll_input
    st.session_state.history = []
    st.session_state.last_suggestion_name = None
    st.session_state.last_suggestion_mises = None
    st.session_state.results = []

# =============================
# HISTORIQUE
# =============================
st.subheader("📝 Historique manuel")

col1, col2 = st.columns([2,1])
with col1:
    spin_choice = st.selectbox("Résultat du spin :", list(segments.keys()), key="spin_choice")
with col2:
    if st.button("➕ Ajouter à l'historique"):
        st.session_state.history.append(spin_choice)

st.write("Historique actuel :", st.session_state.history)

# =============================
# FIN HISTORIQUE
# =============================
if st.button("🏁 Fin historique", key="btn_fin_hist"):
    st.success(f"Historique enregistré ({len(st.session_state.history)} spins). Le bot est prêt à suggérer.")
    # Calculer et stocker la stratégie pour le 1er live spin
    sugg_name, sugg_mises = choose_strategy_intelligent(st.session_state.history, st.session_state.bankroll)
    st.session_state.last_suggestion_name = sugg_name
    st.session_state.last_suggestion_mises = sugg_mises

# =============================
# SUGGESTION
# =============================
st.subheader("📊 Stratégie suggérée (prochain spin)")
if st.session_state.last_suggestion_name and st.session_state.last_suggestion_mises:
    st.markdown(f"**Stratégie :** {st.session_state.last_suggestion_name}")
    st.markdown("**Mises proposées :**")
    st.write({k: round(v,2) for k,v in st.session_state.last_suggestion_mises.items()})
else:
    st.write("Pas encore de suggestion. Appuie sur 'Fin historique' pour commencer.")

# =============================
# ENREGISTRER LIVE SPIN
# =============================
st.subheader("🎲 Enregistrer un live spin")
colA, colB = st.columns([2,1])
with colA:
    live_spin = st.selectbox("Résultat du spin live :", list(segments.keys()), key="live_spin")
with colB:
    if st.button("💾 Enregistrer spin live"):
        if st.session_state.last_suggestion_mises:
            # appliquer résultat
            new_bankroll, gain = apply_spin_result(
                live_spin,
                adjust_bets(st.session_state.last_suggestion_mises, st.session_state.bankroll, st.session_state.initial_bankroll),
                st.session_state.bankroll
            )
            st.session_state.bankroll = new_bankroll
            st.session_state.history.append(live_spin)
            st.session_state.results.append(new_bankroll)
            st.success(f"Résultat enregistré : {live_spin}, gain = {gain}, bankroll = {round(new_bankroll,2)}")

            # calculer la stratégie pour le prochain spin
            sugg_name, sugg_mises = choose_strategy_intelligent(st.session_state.history, st.session_state.bankroll)
            st.session_state.last_suggestion_name = sugg_name
            st.session_state.last_suggestion_mises = sugg_mises

# =============================
# GRAPHIQUE BANQUE
# =============================
if st.session_state.results:
    st.subheader("📈 Évolution de la bankroll")
    fig, ax = plt.subplots()
    ax.plot(st.session_state.results, marker="o")
    ax.set_title("Évolution de la bankroll")
    ax.set_xlabel("Nombre de spins")
    ax.set_ylabel("Bankroll")
    st.pyplot(fig)

# =============================
# BANQUE ACTUELLE
# =============================
st.sidebar.markdown(f"### 💰 Bankroll actuelle : {round(st.session_state.bankroll,2)}")
