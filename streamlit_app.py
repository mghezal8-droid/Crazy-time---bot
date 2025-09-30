import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="Crazy Time Bot", layout="wide")

# -----------------------
# CONSTANTES
# -----------------------
MIN_BETS = [0.2, 0.4, 1, 2, 4, 10]
WHEEL = {
    "1": (21, 2),
    "2": (13, 3),
    "5": (7, 6),
    "10": (4, 11),
    "CoinFlip": (4, 0),
    "Pachinko": (3, 0),
    "CashHunt": (2, 0),
    "CrazyTime": (1, 0)
}
BONUS_MULTIPLIERS = {"CoinFlip": 2.5, "Pachinko": 3, "CashHunt": 4, "CrazyTime": 5}

def adjust_to_minimum(stake):
    for m in MIN_BETS:
        if stake <= m:
            return m
    return MIN_BETS[-1]

# -----------------------
# BOT CLASS
# -----------------------
class CrazyTimeBot:
    def __init__(self, bankroll):
        self.bankroll = bankroll
        self.last_bonus = None
        self.martingale_step_1 = 0
        self.martingale_bet_1 = MIN_BETS[0]

    def suggest_bet(self, past_results):
        bet_suggestion = {}
        if not past_results:
            return bet_suggestion

        last_spin = past_results[-1]

        # Martingale sur 1
        if last_spin == "1":
            self.martingale_step_1 = 0
            self.martingale_bet_1 = MIN_BETS[0]
        else:
            self.martingale_step_1 += 1
            self.martingale_bet_1 = min(adjust_to_minimum(self.martingale_bet_1*2), self.bankroll)
        if self.bankroll >= self.martingale_bet_1:
            bet_suggestion["1"] = self.martingale_bet_1

        # God Mode sur 2,5,10 + bonus (sauf dernier bonus)
        god_targets = ["2","5","10"]
        remaining_bankroll = max(self.bankroll - sum(bet_suggestion.values()), 0)
        if remaining_bankroll >= MIN_BETS[0]:
            portion = remaining_bankroll / len(god_targets)
            for t in god_targets:
                bet_suggestion[t] = adjust_to_minimum(portion)
            for b in ["CoinFlip","Pachinko","CashHunt","CrazyTime"]:
                if b != self.last_bonus:
                    bet_suggestion[b] = adjust_to_minimum(portion / 2)

        # 1 + Bonus Combo
        remaining_bankroll = max(self.bankroll - sum(bet_suggestion.values()), 0)
        if remaining_bankroll >= MIN_BETS[0]:
            bet_suggestion["1"] = bet_suggestion.get("1",0)+MIN_BETS[0]
            for b in ["CoinFlip","Pachinko","CashHunt","CrazyTime"]:
                if b != self.last_bonus:
                    bet_suggestion[b] = bet_suggestion.get(b,0)+MIN_BETS[0]

        if sum(bet_suggestion.values()) < MIN_BETS[0]:
            return {}
        return bet_suggestion

    def apply_spin(self, spin_result, bet_suggestion):
        total_bet = sum(bet_suggestion.values())
        win_amount = 0.0
        hit = False
        for tgt, stake in bet_suggestion.items():
            if tgt == spin_result:
                if tgt in ["1","2","5","10"]:
                    multiplier = WHEEL[tgt][1]
                    win_amount += stake * multiplier
                else:
                    multiplier = BONUS_MULTIPLIERS.get(tgt,0)
                    win_amount += stake * multiplier
                hit = True

        result = {
            "timestamp": datetime.now().isoformat(timespec='seconds'),
            "spin": spin_result,
            "total_bet": round(total_bet,2),
            "win_amount": round(win_amount,2),
            "bankroll_before": round(self.bankroll,2),
            "bankroll_after": None,
            "outcome": "HIT" if hit else "LOSS"
        }

        self.bankroll = self.bankroll - total_bet + win_amount
        result["bankroll_after"] = round(self.bankroll,2)

        if spin_result in ["CoinFlip","Pachinko","CashHunt","CrazyTime"]:
            self.last_bonus = spin_result
        if spin_result == "1":
            self.martingale_step_1 = 0
            self.martingale_bet_1 = MIN_BETS[0]

        return result

# -----------------------
# SESSION STATE
# -----------------------
if "bot" not in st.session_state:
    st.session_state.bot = CrazyTimeBot(120)
if "history_df" not in st.session_state:
    st.session_state.history_df = pd.DataFrame(columns=[
        "timestamp","spin","total_bet","win_amount","bankroll_before","bankroll_after","outcome"
    ])
if "past_results" not in st.session_state:
    st.session_state.past_results = []

bot = st.session_state.bot
past_results = st.session_state.past_results

# -----------------------
# INTERFACE
# -----------------------
st.title("Crazy Time Bot 🎰")

# Réinitialisation
st.sidebar.header("Contrôles session")
if st.sidebar.button("Reset / Nouvelle session"):
    st.session_state.bot = CrazyTimeBot(120)
    st.session_state.history_df = st.session_state.history_df.iloc[0:0]
    st.session_state.past_results = []
    st.experimental_rerun()

st.sidebar.write(f"Bankroll actuelle: {bot.bankroll:.2f}$")
st.sidebar.write(f"Dernier bonus (exclu): {bot.last_bonus or '—'}")

# -----------------------
# Entrée de l'historique
# -----------------------
st.subheader("Entrée des résultats historiques")
spin_input = st.selectbox("Choisir le résultat du spin", ["", "1","2","5","10","CoinFlip","Pachinko","CashHunt","CrazyTime"])
if st.button("Ajouter au tableau"):
    if spin_input:
        past_results.append(spin_input)
        st.success(f"Résultat '{spin_input}' ajouté à l'historique.")
    else:
        st.warning("Veuillez sélectionner un résultat.")

st.write(f"Résultats saisis: {len(past_results)} spins")

# -----------------------
# Affichage et calcul après historique complet
# -----------------------
if past_results:
    st.subheader("Mises suggérées pour le prochain spin")
    suggestion = bot.suggest_bet(past_results)
    if not suggestion:
        st.info("Ne pas miser (aucune mise recommandée selon les règles / bankroll).")
    else:
        df_sugg = pd.DataFrame([{"Section":k,"Mise($)":round(v,2)} for k,v in suggestion.items()])
        st.table(df_sugg)

    # Appliquer un spin (simuler le dernier pour tester)
    if st.button("Appliquer le dernier spin saisi"):
        last_spin = past_results[-1]
        result = bot.apply_spin(last_spin, suggestion)
        st.session_state.history_df = pd.concat([st.session_state.history_df,pd.DataFrame([result])],ignore_index=True)
        st.success(f"Spin '{last_spin}' appliqué : {result['outcome']} — Bankroll: {result['bankroll_after']:.2f}$")

# -----------------------
# Historique
# -----------------------
st.subheader("Historique des spins")
st.dataframe(st.session_state.history_df.tail(20), use_container_width=True)

if not st.session_state.history_df.empty:
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(st.session_state.history_df["bankroll_after"].astype(float).values)
    ax.set_title("Courbe de bankroll")
    ax.set_xlabel("Spin index")
    ax.set_ylabel("Bankroll ($)")
    ax.grid(True)
    st.pyplot(fig)

    csv = st.session_state.history_df.to_csv(index=False).encode('utf-8')
    st.download_button("Télécharger l'historique CSV", data=csv, file_name="crazytime_history.csv", mime="text/csv")

st.caption("⚠️ Suggestions de paris uniquement. Utilisation à vos risques.")
