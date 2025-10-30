# streamlit_app_balanced_v3.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------
# CONFIG INITIALE
# -----------------------------------
st.set_page_config(page_title="🎰 Crazy Time — Bot ML+RTP (Enhanced)", layout="wide")

SEGMENTS = ['1','2','5','10','Coin Flip','Cash Hunt','Pachinko','Crazy Time','Top Slot']
BONUSES = ['Coin Flip','Cash Hunt','Pachinko','Crazy Time']
VAL_SEG = {'1':1,'2':2,'5':5,'10':10}
THEO_COUNTS = {'1':21,'2':13,'5':7,'10':4,'Coin Flip':4,'Cash Hunt':2,'Pachinko':2,'Crazy Time':1,'Top Slot':1}
THEO_TOTAL = sum(THEO_COUNTS.values())
RTP_BASE_LOW, RTP_BASE_HIGH = 94.41, 96.08

# -----------------------------------
# INIT STATE
# -----------------------------------
def ss(k,v): 
    if k not in st.session_state: st.session_state[k]=v
ss("history",[])
ss("live_history",[])
ss("results_table",[])
ss("strategies_table",[])
ss("bankroll",200.0)
ss("initial_bankroll",200.0)
ss("unit",1.0)
ss("bonus_mult_assumption",10)
ss("mult_for_ev",1)
ss("ml_window",50)
ss("ml_model",None)
ss("ml_trained",False)
ss("ml_boost",1.5)
ss("rtp_weight",0.5)
ss("last_suggestion",None)
ss("martingale_streak",0)

# -----------------------------------
# OUTILS
# -----------------------------------
def normalize_entry(e):
    if isinstance(e, dict): return e.get("segment"), float(e.get("mult",1.0))
    return e,1.0

def full_hist_segments(): 
    return [normalize_entry(h)[0] for h in st.session_state.history + st.session_state.live_history]

def theo_prob(s): return THEO_COUNTS.get(s,0)/THEO_TOTAL

def hist_prob(hist,s,window=300):
    if not hist: return 0
    return hist[-window:].count(s)/len(hist[-window:])

def combined_prob(hist,s,window=300):
    w=st.session_state.rtp_weight
    return w*theo_prob(s)+(1-w)*hist_prob(hist,s,window)

def compute_rtp_last_n(n=100):
    rows=st.session_state.results_table[-n:]
    if not rows: return None
    wager,ret=0,0
    for r in rows:
        mise=r.get("Mise Totale",sum(r.get("Mises $",{}).values()))
        gain=r.get("Gain Brut",0)
        if mise: wager+=mise
        ret+=gain
    return (ret/wager*100) if wager>0 else None

# -----------------------------------
# MACHINE LEARNING
# -----------------------------------
def train_ml():
    segs=full_hist_segments()
    if len(segs)<=st.session_state.ml_window: return
    idx=[SEGMENTS.index(s) for s in segs]
    X,y=[],[]
    W=st.session_state.ml_window
    for i in range(len(idx)-W):
        X.append(idx[i:i+W])
        y.append(idx[i+W])
    clf=RandomForestClassifier(n_estimators=100,random_state=42)
    clf.fit(X,y)
    st.session_state.ml_model=clf
    st.session_state.ml_trained=True

def ml_predict_window():
    if not st.session_state.ml_trained: return None,{}
    segs=full_hist_segments()
    if len(segs)<st.session_state.ml_window: return None,{}
    X=[SEGMENTS.index(s) for s in segs[-st.session_state.ml_window:]]
    p=st.session_state.ml_model.predict_proba([X])[0]
    probs={SEGMENTS[i]:float(p[i]) for i in range(len(p))}
    return SEGMENTS[int(np.argmax(p))],probs

# -----------------------------------
# STRATÉGIES
# -----------------------------------
def strat_martingale_1(bankroll,unit):
    mise = unit*(2**st.session_state.martingale_streak)
    return "Martingale 1",{'1':mise}

def strat_god_mode(bankroll,unit): return "God Mode",{'2':unit*2,'5':unit*1.5,'10':unit,'Top Slot':unit*0.5}
def strat_god_mode_bonus(bankroll,unit):
    b={'2':unit*2,'5':unit*1.5,'10':unit}
    for x in BONUSES+['Top Slot']: b[x]=unit
    return "God Mode + Bonus",b
def strat_1_bonus(bankroll,unit):
    b={'1':unit*2}
    for x in BONUSES+['Top Slot']: b[x]=unit
    return "1+Bonus",b
def strat_only_numbers(bankroll,unit): return "Only Numbers",{'1':unit*1.5,'2':unit*1.2,'5':unit,'10':unit,'Top Slot':unit*0.5}
def strat_all_but_1(bankroll,unit): 
    b={s:unit for s in SEGMENTS if s!='1'}
    return "All but 1",b

STRATS={
    "Martingale 1":strat_martingale_1,
    "God Mode":strat_god_mode,
    "God Mode + Bonus":strat_god_mode_bonus,
    "1+Bonus":strat_1_bonus,
    "Only Numbers":strat_only_numbers,
    "All but 1":strat_all_but_1
}

# -----------------------------------
# CALCUL DES MISES ET GAINS
# -----------------------------------
def compute_bets(name,bkr,unit,rtp100=None,ml_pred=None):
    _,b=STRATS[name](bkr,unit)
    bets={s:float(b.get(s,0)) for s in SEGMENTS}
    if rtp100:
        if rtp100<RTP_BASE_LOW:
            for s in BONUSES+['Top Slot']: bets[s]*=1.5
        elif rtp100>RTP_BASE_HIGH:
            for s in ['1','2','5','10']: bets[s]*=1.2
    if ml_pred and ml_pred in bets: bets[ml_pred]*=st.session_state.ml_boost
    return {k:round(v,2) for k,v in bets.items()}

def calcul_gain(bets,seg,mult=1.0,bonus_mult=None):
    mise=sum(bets.values())
    if seg not in bets or bets[seg]<=0: return 0,-mise
    mise_seg=bets[seg]
    if seg in VAL_SEG:
        factor=VAL_SEG[seg]+1
        gain=mise_seg*factor*mult
    else:
        bm=bonus_mult or st.session_state.bonus_mult_assumption
        gain=mise_seg*(bm*mult)+mise_seg
    return round(gain,2),round(gain-mise,2)

# -----------------------------------
# CHOIX AUTOMATIQUE STRATÉGIE (pondération 100 spins)
# -----------------------------------
def suggest_strategy(hist,bkr,mult):
    if len(hist)>=st.session_state.ml_window: train_ml()
    ml_pred,ml_probs=(None,{})
    if st.session_state.ml_trained: ml_pred,ml_probs=ml_predict_window()
    rtp100=compute_rtp_last_n(100)
    recent=hist[-100:]  # 🔥 tendance pondérée sur 100 spins
    counts={s:recent.count(s) for s in SEGMENTS}
    max_trend=max(counts.values()) if counts else 1
    trend_weight={s:(counts[s]/max_trend) for s in SEGMENTS}
    best_ev=-99999;best_name=None;best_bets={}
    for name in STRATS:
        bets=compute_bets(name,bkr,st.session_state.unit,rtp100,ml_pred)
        mise_tot=sum(bets.values())
        ev=0
        for s in SEGMENTS:
            p=combined_prob(hist,s)*trend_weight[s]
            if s in bets and bets[s]>0:
                val=VAL_SEG.get(s,st.session_state.bonus_mult_assumption)
                payout=bets[s]*(val*mult)+bets[s]
                net=payout-mise_tot
            else:
                net=-mise_tot
            ev+=p*net
        if ev>best_ev: best_ev,best_name,best_bets=ev,name,bets
    return {"name":best_name,"bets":best_bets,"ev":best_ev}

# -----------------------------------
# INTERFACE STREAMLIT
# -----------------------------------
st.title("🎰 Crazy Time — Bot ML+RTP (Enhanced)")

col1,col2=st.columns(2)

with col1:
    st.header("Historique manuel")
    for i,s in enumerate(SEGMENTS):
        if st.button(s,key=f"seg{i}",use_container_width=True):
            st.session_state.history.append({"segment":s,"mult":1.0})
    if st.session_state.history:
        df=pd.DataFrame([{"#":i+1,"Segment":h["segment"]} for i,h in enumerate(st.session_state.history)])
        st.dataframe(df,use_container_width=True)
    if st.button("🏁 Fin historique et commencer"):
        full=full_hist_segments()
        sug=suggest_strategy(full,st.session_state.bankroll,1)
        st.session_state.last_suggestion=sug
        st.success(f"Première stratégie suggérée : {sug['name']}")

with col2:
    st.header("Paramètres")
    st.session_state.initial_bankroll=st.number_input("Bankroll initiale",value=int(st.session_state.initial_bankroll))
    st.session_state.unit=st.number_input("Unité de base",value=int(st.session_state.unit))
    st.session_state.mult_for_ev=st.number_input("Multiplicateur EV",value=1)

st.markdown("---")
st.header("Spin live")
seg=st.selectbox("Résultat:",SEGMENTS)
mult=st.number_input("Multiplicateur:",value=1)
if st.button("🎰 Enregistrer"):
    sug=st.session_state.last_suggestion or suggest_strategy(full_hist_segments(),st.session_state.bankroll,1)
    bets=sug["bets"]
    gain_brut,gain_net=calcul_gain(bets,seg,mult)
    st.session_state.bankroll+=gain_net
    if seg=="1" and gain_net>0: st.session_state.martingale_streak=0
    elif "1" in bets: st.session_state.martingale_streak+=1
    st.session_state.results_table.append({
        "Spin #":len(st.session_state.results_table)+1,
        "Résultat":seg,"Mises":bets,"Gain Net":gain_net,"Bankroll":st.session_state.bankroll
    })
    st.session_state.live_history.append({"segment":seg,"mult":mult})
    next_sug=suggest_strategy(full_hist_segments(),st.session_state.bankroll,1)
    st.session_state.last_suggestion=next_sug
    st.success(f"{seg} — Gain net {gain_net}$")

# 🎯 Affichage de la stratégie suggérée juste sous le bouton
if st.session_state.last_suggestion:
    st.subheader("🎯 Stratégie suggérée pour le prochain spin :")
    s = st.session_state.last_suggestion
    st.markdown(f"**Nom :** {s['name']} — **EV estimé :** {round(s['ev'],2)}$")
    st.write(pd.DataFrame.from_dict(s['bets'], orient='index', columns=['Mise ($)']))

# ---- Tableau des résultats ----
if st.session_state.results_table:
    df=pd.DataFrame(st.session_state.results_table)
    st.dataframe(df,use_container_width=True)
    fig,ax=plt.subplots()
    ax.plot(df["Spin #"],df["Bankroll"],marker='o')
    ax.axhline(y=st.session_state.initial_bankroll,color='gray',ls='--')
    st.pyplot(fig)
