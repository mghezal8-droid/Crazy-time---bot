# streamlit_app_final.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="🎰 Crazy Time — Bot Auto Suggest", layout="wide")

SEGMENTS = ['1','2','5','10','Coin Flip','Cash Hunt','Pachinko','Crazy Time','Top Slot']
BONUSES = ['Coin Flip','Cash Hunt','Pachinko','Crazy Time']
VAL_SEG = {'1':1,'2':2,'5':5,'10':10}
THEO_COUNTS = {'1':21,'2':13,'5':7,'10':4,'Coin Flip':4,'Cash Hunt':2,'Pachinko':2,'Crazy Time':1,'Top Slot':1}
THEO_TOTAL = sum(THEO_COUNTS.values())
RTP_BASE_LOW, RTP_BASE_HIGH = 94.41, 96.08

# ---------------------------
# INIT
# ---------------------------
def ss_init(k,v): 
    if k not in st.session_state: st.session_state[k]=v

ss_init("history", [])
ss_init("live_history", [])
ss_init("results_table", [])
ss_init("strategies_table", [])
ss_init("bankroll", 200.0)
ss_init("initial_bankroll", 200.0)
ss_init("unit", 1.0)
ss_init("bonus_mult_assumption", 10)
ss_init("mult_for_ev", 1)
ss_init("ml_window", 50)
ss_init("ml_model", None)
ss_init("ml_trained", False)
ss_init("ml_boost", 1.5)
ss_init("rtp_weight", 0.5)
ss_init("last_suggestion", None)

# ---------------------------
# FUNCTIONS
# ---------------------------
def normalize_entry(e):
    if isinstance(e, dict):
        return e.get("segment"), float(e.get("mult",1.0))
    return e,1.0

def full_history_segments():
    segs=[]
    for h in st.session_state.history+st.session_state.live_history:
        seg,_=normalize_entry(h)
        segs.append(seg)
    return segs

def theo_prob(segment): return THEO_COUNTS.get(segment,0)/THEO_TOTAL
def hist_prob(full_hist,segment,window=300):
    if not full_hist: return 0
    hist=full_hist[-window:]
    return hist.count(segment)/len(hist)
def combined_prob(full_hist,segment,window=300):
    w=st.session_state.rtp_weight
    return w*theo_prob(segment)+(1-w)*hist_prob(full_hist,segment,window)

def compute_rtp_last_n(n=100):
    rows=st.session_state.results_table[-n:]
    if not rows: return None
    tot_mise=tot_ret=0
    for r in rows:
        mise=r.get("Mise Totale",sum(r.get("Mises $",{}).values()))
        gain=r.get("Gain Brut") or 0
        tot_mise+=mise; tot_ret+=gain
    return (tot_ret/tot_mise)*100 if tot_mise>0 else None

def train_ml():
    segs=full_history_segments()
    if len(segs)<=st.session_state.ml_window: return
    idx=[SEGMENTS.index(s) for s in segs]
    W=st.session_state.ml_window
    X,y=[],[]
    for i in range(len(idx)-W):
        X.append(idx[i:i+W]); y.append(idx[i+W])
    clf=RandomForestClassifier(n_estimators=100,random_state=42)
    clf.fit(X,y)
    st.session_state.ml_model=clf
    st.session_state.ml_trained=True

def ml_predict_window():
    if not st.session_state.ml_trained: return None,{}
    segs=full_history_segments()
    if len(segs)<st.session_state.ml_window: return None,{}
    last=[SEGMENTS.index(s) for s in segs[-st.session_state.ml_window:]]
    probs=st.session_state.ml_model.predict_proba([last])[0]
    per_seg={SEGMENTS[i]:float(probs[i]) for i in range(len(probs))}
    return SEGMENTS[int(np.argmax(probs))],per_seg

# --- Stratégies
def strat_martingale_1(b,u): return "Martingale 1",{'1':u}
def strat_god_mode(b,u): return "God Mode",{'2':u*2,'5':u*1.5,'10':u*1.0,'Top Slot':u*0.5}
def strat_god_mode_bonus(b,u):
    d={'2':u*2,'5':u*1.5,'10':u*1.0}
    for s in BONUSES+['Top Slot']: d[s]=u
    return "God Mode + Bonus",d
def strat_1_plus_bonus(b,u):
    d={'1':u*2}
    for s in BONUSES+['Top Slot']: d[s]=u
    return "1+Bonus",d
def strat_only_numbers(b,u): return "Only Numbers",{'1':u*1.5,'2':u*1.2,'5':u,'10':u,'Top Slot':u*0.5}
def strat_all_but_1(b,u): return "All but 1",{s:u for s in SEGMENTS if s!='1'}

STRATEGIES={
    "Martingale 1":strat_martingale_1,
    "God Mode":strat_god_mode,
    "God Mode + Bonus":strat_god_mode_bonus,
    "1+Bonus":strat_1_plus_bonus,
    "Only Numbers":strat_only_numbers,
    "All but 1":strat_all_but_1
}

def compute_bets_for_strategy(name,b,u,rtp_last100=None,ml_pred=None):
    builder=STRATEGIES[name]
    _,bets=builder(b,u)
    if rtp_last100:
        if rtp_last100<RTP_BASE_LOW:
            for s in BONUSES+['Top Slot']: bets[s]*=1.6
        elif rtp_last100>RTP_BASE_HIGH:
            for s in ['1','2','5','10']: bets[s]*=1.3
    if ml_pred and ml_pred in bets: bets[ml_pred]*=st.session_state.ml_boost
    return {k:int(v) for k,v in bets.items()}

def expected_value_for_strategy(bets,full_hist,mult,bkr):
    mise=sum(bets.values()); ev=0
    for seg in SEGMENTS:
        p=combined_prob(full_hist,seg,window=st.session_state.ml_window)
        if seg in bets and bets[seg]>0:
            seg_val=VAL_SEG.get(seg,st.session_state.bonus_mult_assumption)
            payout=bets[seg]*(seg_val*mult)+bets[seg]
            net=payout-mise
        else: net=-mise
        ev+=p*net
    return ev

def calcul_gain_from_bets(bets,spin,mult=1.0,bonus_mult=None):
    mise_tot=sum(bets.values())
    if mise_tot<=0: return 0,0
    if bets.get(spin,0)<=0: return 0,-mise_tot
    mise=bets[spin]
    if spin in VAL_SEG:
        factor=VAL_SEG[spin]+1
        gain_brut=mise*factor*mult
    else:
        bm=bonus_mult or st.session_state.bonus_mult_assumption
        gain_brut=mise*(bm*mult)+mise
    gain_net=gain_brut-mise_tot
    return round(gain_brut,2),round(gain_net,2)

def suggest_strategy(full_hist,bkr,mult):
    if len(full_hist)>=st.session_state.ml_window: train_ml()
    ml_pred,_=ml_predict_window()
    rtp100=compute_rtp_last_n(100)
    best_ev=-1e9;best={}
    for name in STRATEGIES:
        bets=compute_bets_for_strategy(name,bkr,st.session_state.unit,rtp100,ml_pred)
        ev=expected_value_for_strategy(bets,full_hist,mult,bkr)
        if ev>best_ev: best_ev=ev;best={"name":name,"bets":bets,"ev":ev}
    return best

# ---------------------------
# UI
# ---------------------------
st.title("🎰 Crazy Time — Auto Strategy Bot")

col1,col2=st.columns([1,1])

with col1:
    st.header("1️⃣ Historique manuel")
    for i,seg in enumerate(SEGMENTS):
        if st.button(seg,key=f"hist_{i}",use_container_width=True):
            st.session_state.history.append({"segment":seg,"mult":1.0})
    c1,c2,c3=st.columns(3)
    if c1.button("↩ Supprimer dernier"): 
        if st.session_state.history: st.session_state.history.pop()
    if c2.button("🔄 Réinitialiser"): 
        st.session_state.history=[]
    if c3.button("🏁 Fin historique"):
        full=full_history_segments()
        s=suggest_strategy(full,st.session_state.bankroll,st.session_state.mult_for_ev)
        rec={"Spin #":len(st.session_state.results_table)+1,"Stratégie suggérée":s["name"],
             "Mises $ (suggérées)":s["bets"],"EV prévisionnel":round(s["ev"],2)}
        st.session_state.strategies_table.append(rec)
        st.session_state.last_suggestion=s
        st.success(f"Suggestion calculée : {s['name']}")

with col2:
    st.header("2️⃣ Paramètres rapides")
    st.session_state.initial_bankroll=st.number_input("Bankroll initiale",value=200,step=1)
    st.session_state.unit=st.number_input("Unité de base ($)",value=1,step=1)
    st.session_state.mult_for_ev=st.number_input("Multiplicateur (EV)",value=1,step=1)

st.markdown("---")
st.header("3️⃣ Spin Live")

colL,colM,colR=st.columns([1,1,1])
spin=colL.selectbox("Résultat du spin",SEGMENTS)
mult=colM.number_input("Multiplicateur",value=1,step=1)
chosen=colR.selectbox("Stratégie (ou suggestion)",["(use suggestion)"]+list(STRATEGIES.keys()))

if st.button("🎰 Enregistrer spin",use_container_width=True):
    if st.session_state.last_suggestion: s=st.session_state.last_suggestion
    else: s=suggest_strategy(full_history_segments(),st.session_state.bankroll,1)
    name,bets=s["name"],s["bets"].copy()
    if chosen!="(use suggestion)": name=chosen; bets=compute_bets_for_strategy(name,st.session_state.bankroll,st.session_state.unit)
    bonus_mult=mult if spin in BONUSES+['Top Slot'] else None
    gain_brut,gain_net=calcul_gain_from_bets(bets,spin,mult,bonus_mult)
    st.session_state.bankroll+=gain_net
    rec={"Spin #":len(st.session_state.results_table)+1,"Stratégie":name,"Résultat":spin,
         "Multiplicateur":mult,"Mises $":bets,"Mise Totale":sum(bets.values()),
         "Gain Brut":gain_brut,"Gain Net":gain_net,"Bankroll":round(st.session_state.bankroll,2)}
    st.session_state.results_table.append(rec)
    st.session_state.live_history.append({"segment":spin,"mult":mult})

    # maj tableau stratégie
    if st.session_state.strategies_table and st.session_state.strategies_table[-1].get("Résultat réel") is None:
        st.session_state.strategies_table[-1].update({
            "Résultat réel":spin,"Multiplier réel":mult,"Gain Brut (réel)":gain_brut,
            "Gain Net (réel)":gain_net,"Bankroll après":round(st.session_state.bankroll,2)
        })
    else:
        st.session_state.strategies_table.append({
            "Spin #":rec["Spin #"],"Stratégie suggérée":name,"Mises $ (suggérées)":bets,
            "Résultat réel":spin,"Multiplier réel":mult,"Gain Net (réel)":gain_net,
            "Bankroll après":round(st.session_state.bankroll,2)
        })

    # Nouvelle suggestion
    s_next=suggest_strategy(full_history_segments(),st.session_state.bankroll,1)
    st.session_state.last_suggestion=s_next
    st.session_state.strategies_table.append({
        "Spin #":len(st.session_state.results_table)+1,
        "Stratégie suggérée":s_next["name"],
        "Mises $ (suggérées)":s_next["bets"],
        "EV prévisionnel":round(s_next["ev"],2)
    })
    st.success(f"Spin enregistré ({spin} x{mult}) → Gain net {gain_net}$ | Prochaine stratégie : {s_next['name']}")

st.markdown("---")
st.subheader("📊 Historique des spins")
if st.session_state.results_table:
    df=pd.DataFrame(st.session_state.results_table)
    st.dataframe(df,use_container_width=True)
    fig,ax=plt.subplots()
    ax.plot(df["Spin #"],df["Bankroll"],marker='o')
    ax.axhline(y=st.session_state.initial_bankroll,linestyle='--',color='gray')
    ax.set_xlabel("Spin #"); ax.set_ylabel("Bankroll ($)"); ax.grid(True)
    st.pyplot(fig)
else:
    st.info("Aucun spin live enregistré.")

st.markdown("---")
st.subheader("📈 Stratégies suggérées & résultats")
if st.session_state.strategies_table:
    st.dataframe(pd.DataFrame(st.session_state.strategies_table),use_container_width=True)
else:
    st.info("Aucune stratégie encore.")
