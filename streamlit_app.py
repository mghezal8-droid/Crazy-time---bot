import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="Crazy Time Bot Optimisé", layout="wide")

# -----------------------
# CONSTANTES
# -----------------------
MIN_BETS = [0.2, 0.4, 1, 2, 4, 10]
SPINS = ["1","2","5","10","CoinFlip","Pachinko","CashHunt","CrazyTime"]
WHEEL = {"1": (21,2),"2":(13,3),"5":(7,6),"10":(4,11),
         "CoinFlip":(4,0),"Pachinko":(3,0),"CashHunt":(2,0),"CrazyTime":(1,0)}
BONUS_MULTIPLIERS = {"CoinFlip":2.5,"Pachinko":3,"CashHunt":4,"CrazyTime":5}

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
        strategies_used = []

        if not past_results:
            return bet_suggestion, strategies_used

        last_spin = past_results[-1] if past_results else None

        # --- Martingale 1 ---
        if last_spin == "1":
            self.martingale_step_1 = 0
            self.martingale_bet_1 = MIN_BETS[0]
        else:
            self.martingale_step_1 += 1
            self.martingale_bet_1 = min(adjust_to_minimum(self.martingale_bet_1*2), self.bankroll)

        if self.bankroll >= self.martingale_bet_1:
            bet_suggestion["1"] = self.martingale_bet_1
            strategies_used.append("Martingale 1")

        # --- God Mode 2,5,10 ---
        god_targets = ["2","5","10"]
        remaining = max(self.bankroll - sum(bet_suggestion.values()), 0)
        if remaining >= MIN_BETS[0]:
            portion = remaining / len(god_targets)
            for t in god_targets:
                bet_suggestion[t] = adjust_to_minimum(portion)
            strategies_used.append("God Mode 2,5,10")

        # --- God Mode 2,5,10 + Bonus ---
        remaining = max(self.bankroll - sum(bet_suggestion.values()),0)
        if remaining >= MIN_BETS[0]:
            portion = remaining / (len(god_targets)+3)
            for t in god_targets:
                bet_suggestion[t] = adjust_to_minimum(portion)
            for b in ["CoinFlip","Pachinko","CashHunt"]:
                if b != self.last_bonus:
                    bet_suggestion[b] = adjust_to_minimum(portion)
            strategies_used.append("God Mode 2,5,10 + Bonus")

        # --- 1 + Bonus Combo ---
        remaining = max(self.bankroll - sum(bet_suggestion.values()),0)
        if remaining >= MIN_BETS[0]:
            bet_suggestion["1"] = bet_suggestion.get("1",0) + MIN_BETS[0]
            for b in ["CoinFlip","Pachinko","CashHunt","CrazyTime"]:
                if b != self.last_bonus:
                    bet_suggestion[b] = bet_suggestion.get(b,0) + MIN_BETS[0]
            strategies_used.append("1 + Bonus Combo")

        # Ajustement pour bankroll
        total_bet = sum(bet_suggestion.values())
        if total_bet > self.bankroll:
            scale = self.bankroll / total_bet
            for k in bet_suggestion:
                bet_suggestion[k] = max(MIN_BETS[0], round(bet_suggestion[k]*scale,2))

        return bet_suggestion, strategies_used

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
    st.session_state.history_df = pd.DataFrame(columns=["timestamp","spin","total_bet","win_amount","bankroll_before","bankroll_after","outcome"])
if "past_results" not in st.session_state:
    st.session_state.past_results = []
if "history_finished" not in st.session_state:
    st.session_state.history_finished = False

bot = st.session_state.bot
past_results = st.session_state.past_results
history_finished = st.session_state.history_finished

# -----------------------
# INTERFACE
# -----------------------
st.title("Crazy Time Bot Optimisé ✅")

st.sidebar.header("Contrôles")
if st.sidebar.button("Reset / Nouvelle session"):
    st.session_state.bot = CrazyTimeBot(120)
    st.session_state.history_df = st.session_state.history_df.iloc[0:0]
    st.session_state.past_results = []
    st.session_state.history_finished = False
    st.success("Session réinitialisée.")

st.sidebar.write(f"Bankroll: {bot.bankroll:.2f}$")
st.sidebar.write(f"Dernier bonus exclu: {bot.last_bonus or '—'}")

# -----------------------
# Entrée des spins
# -----------------------
if not history_finished:
    st.subheader("Entrée de l'historique des spins")
    cols = st.columns(4)
    for idx, spin in enumerate(SPINS):
        if cols[idx%4].button(spin):
            past_results.append(spin)
            st.success(f"Résultat '{spin}' ajouté à l'historique.")

    st.write(f"Total spins saisis: {len(past_results)}")
    if st.button("Historique terminé / Live spin"):
        st.session_state.history_finished = True
        st.success("Historique terminé. Maintenant le bot peut suggérer les mises.")
else:
    st.subheader("Mises suggérées pour le prochain spin")
    suggestion, strategies = bot.suggest_bet(past_results)
    if not suggestion:
        st.info("Aucune mise recommandée selon la stratégie / bankroll.")
    else:
        df_sugg = pd.DataFrame([{"Segment":k,"Mise($)":round(v,2)} for k,v in suggestion.items()])
        st.table(df_sugg)
        st.write("Stratégies appliquées :", ", ".join(strategies))

    if st.button("Appliquer le dernier spin saisi"):
        last_spin = past_results[-1]
        result = bot.apply_spin(last_spin, suggestion)
        st.session_state.history_df = pd.concat([st.session_state.history_df,pd.DataFrame([result])],ignore_index=True)
        st.success(f"Spin '{last_spin}' : {result['outcome']} — Bankroll: {result['bankroll_after']:.2f}$")

# -----------------------
# Historique et graphique
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

st.caption("⚠️ Suggestions uniquement. Pour analyse / simulation.")
