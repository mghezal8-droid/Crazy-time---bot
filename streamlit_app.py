import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="Crazy Time Bot - Stratégies Fixes", layout="wide")

# -----------------------
# CONSTANTES
# -----------------------
MIN_BETS = [0.2, 0.4, 1, 2, 4, 10]
SPINS = ["1","2","5","10","CoinFlip","Pachinko","CashHunt","CrazyTime"]

# Probabilités de la roue
WHEEL = {"1": 21, "2": 13, "5": 7, "10": 4,
         "CoinFlip": 4, "Pachinko": 3, "CashHunt": 2, "CrazyTime": 1}
TOTAL_SEGMENTS = sum(WHEEL.values())

# Multiplicateurs fixes
PAYOUTS = {"1":2,"2":3,"5":6,"10":11,
           "CoinFlip":2.5,"Pachinko":3,"CashHunt":4,"CrazyTime":5}

def adjust_to_minimum(stake):
    for m in MIN_BETS:
        if stake <= m:
            return m
    return MIN_BETS[-1]

# -----------------------
# BOT
# -----------------------
class CrazyTimeBot:
    def __init__(self, bankroll):
        self.bankroll = bankroll
        self.last_bonus = None
        self.martingale_step_1 = 0
        self.martingale_bet_1 = MIN_BETS[0]

    def choose_strategy(self, past_results):
        """Décide quelle stratégie utiliser en comparant fréquences vs attendues"""
        if not past_results:
            return "God Mode 2,5,10"  # défaut

        df = pd.Series(past_results).value_counts()
        freqs = {seg: df.get(seg,0)/len(past_results) for seg in WHEEL}
        expected = {seg: WHEEL[seg]/TOTAL_SEGMENTS for seg in WHEEL}

        # Martingale 1 si "1" sort moins que prévu
        if freqs["1"] < expected["1"]*0.8:
            return "Martingale 1"

        # God Mode 2,5,10 si gros chiffres manquent
        big_nums = ["2","5","10"]
        if sum(freqs[x] for x in big_nums) < sum(expected[x] for x in big_nums)*0.8:
            return "God Mode 2,5,10"

        # God Mode 2,5,10 + Bonus si gros chiffres ET bonus manquent
        bonus = ["CoinFlip","Pachinko","CashHunt","CrazyTime"]
        if (sum(freqs[x] for x in big_nums) < sum(expected[x] for x in big_nums)*0.9
            and sum(freqs[x] for x in bonus) < sum(expected[x] for x in bonus)*0.9):
            return "God Mode 2,5,10 + Bonus"

        # Sinon 1 + Bonus Combo par défaut
        return "1 + Bonus Combo"

    def suggest_bet(self, past_results):
        strat = self.choose_strategy(past_results)
        bets = {}

        if strat == "Martingale 1":
            # progression
            self.martingale_bet_1 = adjust_to_minimum(self.martingale_bet_1*2 if past_results and past_results[-1]!="1" else MIN_BETS[0])
            if self.bankroll >= self.martingale_bet_1:
                bets["1"] = self.martingale_bet_1

        elif strat == "God Mode 2,5,10":
            bets["2"] = adjust_to_minimum(2)
            bets["5"] = adjust_to_minimum(1)
            bets["10"] = adjust_to_minimum(1)

        elif strat == "God Mode 2,5,10 + Bonus":
            bets["2"] = adjust_to_minimum(2)
            bets["5"] = adjust_to_minimum(1)
            bets["10"] = adjust_to_minimum(1)
            for b in ["CoinFlip","Pachinko","CashHunt","CrazyTime"]:
                if b != self.last_bonus:
                    bets[b] = adjust_to_minimum(1)

        elif strat == "1 + Bonus Combo":
            bets["1"] = adjust_to_minimum(1)
            for b in ["CoinFlip","Pachinko","CashHunt","CrazyTime"]:
                if b != self.last_bonus:
                    bets[b] = adjust_to_minimum(1)

        return bets, strat

    def apply_spin(self, spin_result, bets):
        total_bet = sum(bets.values())
        win_amount = 0
        hit = False

        if spin_result in bets:
            win_amount = bets[spin_result] * PAYOUTS[spin_result]
            hit = True

        self.bankroll = self.bankroll - total_bet + win_amount

        if spin_result in ["CoinFlip","Pachinko","CashHunt","CrazyTime"]:
            self.last_bonus = spin_result

        return {
            "timestamp": datetime.now().isoformat(timespec='seconds'),
            "spin": spin_result,
            "strategy": bets,
            "total_bet": round(total_bet,2),
            "win_amount": round(win_amount,2),
            "bankroll": round(self.bankroll,2),
            "outcome": "HIT" if hit else "LOSS"
        }

# -----------------------
# SESSION STATE
# -----------------------
if "bot" not in st.session_state:
    st.session_state.bot = CrazyTimeBot(120)
if "past_results" not in st.session_state:
    st.session_state.past_results = []
if "history_df" not in st.session_state:
    st.session_state.history_df = pd.DataFrame()

bot = st.session_state.bot
past_results = st.session_state.past_results

# -----------------------
# INTERFACE
# -----------------------
st.title("Crazy Time Bot 🎡 - Stratégies intelligentes")

st.sidebar.header("Bankroll")
st.sidebar.write(f"{bot.bankroll:.2f} $")
st.sidebar.write(f"Dernier bonus exclu : {bot.last_bonus or '—'}")

# Entrée des spins
st.subheader("Entrée des spins")
cols = st.columns(4)
for idx, spin in enumerate(SPINS):
    if cols[idx%4].button(spin):
        past_results.append(spin)
        st.success(f"Spin ajouté : {spin}")

if past_results:
    st.write("Historique :", past_results)

# Suggestion
if past_results:
    bets, strat = bot.suggest_bet(past_results)
    st.subheader("Suggestion du bot")
    st.write(f"Stratégie choisie : **{strat}**")

    if bets:
        df = pd.DataFrame([{"Segment":k,"Mise($)":v} for k,v in bets.items()])
        st.table(df)
    else:
        st.info("Aucune mise suggérée.")

    if st.button("Appliquer le dernier spin saisi"):
        last_spin = past_results[-1]
        result = bot.apply_spin(last_spin, bets)
        st.session_state.history_df = pd.concat([st.session_state.history_df,pd.DataFrame([result])],ignore_index=True)
        st.success(f"Spin {last_spin} → {result['outcome']} | Bankroll : {result['bankroll']}$")

# Historique
if not st.session_state.history_df.empty:
    st.subheader("Historique")
    st.dataframe(st.session_state.history_df)

    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(st.session_state.history_df["bankroll"].astype(float).values)
    ax.set_title("Évolution bankroll")
    st.pyplot(fig)
