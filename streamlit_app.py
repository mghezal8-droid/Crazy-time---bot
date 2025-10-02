import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Crazy Time Bot", layout="wide")

# --- Initialisation session ---
if "history" not in st.session_state:
    st.session_state.history = []
if "bankroll" not in st.session_state:
    st.session_state.bankroll = 150.0
if "base_unit" not in st.session_state:
    st.session_state.base_unit = 1.0
if "current_mises" not in st.session_state:
    st.session_state.current_mises = {}
if "last_spin" not in st.session_state:
    st.session_state.last_spin = None
if "spin_results" not in st.session_state:
    st.session_state.spin_results = []
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []

# --- Segments de la roue ---
segments = ["1","2","5","10","Coin Flip","Cash Hunt","Pachinko","Crazy Time"]

# --- Ajustement mise dynamique ---
def adjust_mises(bankroll, base_unit):
    factor = 1
    if bankroll >= 2*150:
        factor = 2
    elif bankroll <= 0.5*150:
        factor = 0.5
    mises = {s: base_unit*factor for s in segments}
    return mises, {s: factor for s in segments}  # retourne également les unités

# --- Calcul gain net ---
def calculate_gain_net(winning_segment, mises, multiplicateur=1):
    mise_totale = sum(mises.values())
    mise_gagnante = mises.get(winning_segment, 0)
    gain_net = (mise_gagnante * multiplicateur) + mise_gagnante - (mise_totale - mise_gagnante)
    return gain_net, mise_totale

# --- Ajouter spin à l'historique ---
def add_spin_to_history(spin, mult):
    st.session_state.history.append({"spin": spin, "multiplicateur": mult})

# --- Calcul bankroll spin par spin ---
def compute_bankroll(history, base_unit):
    bankroll = st.session_state.bankroll
    records = []
    for entry in history:
        mises, units = adjust_mises(bankroll, base_unit)
        gain_net, mise_totale = calculate_gain_net(entry["spin"], mises, entry["multiplicateur"])
        bankroll += gain_net
        records.append({
            "Spin": entry["spin"],
            "Multiplicateur": entry["multiplicateur"],
            "Mises par segment ($)": mises,
            "Mises par segment (unités)": units,
            "Mise totale": mise_totale,
            "Gain net": gain_net,
            "Bankroll": bankroll
        })
    return records, bankroll

# --- Interface utilisateur ---
st.title("Crazy Time Bot")

# Sidebar paramètres
st.sidebar.header("Paramètres")
st.session_state.base_unit = st.sidebar.number_input("Unité de mise de base ($)", 0.2, 10.0, st.session_state.base_unit, 0.2)
st.session_state.bankroll = st.sidebar.number_input("Bankroll initial ($)", 50, 1000, st.session_state.bankroll)
bankroll_critique = st.sidebar.number_input("Seuil bankroll critique ($)", 10, 150, 30)

# Entrée historique
st.header("Ajouter historique des spins")
col1, col2 = st.columns([2,1])
with col1:
    for s in segments:
        if st.button(s, key=f"hist_{s}"):
            st.session_state.last_spin = s
with col2:
    multiplicateur_input = st.number_input("Multiplicateur du spin sélectionné", 1, 100, 1, 1)
    if st.button("Ajouter Spin avec multiplicateur"):
        if st.session_state.last_spin:
            add_spin_to_history(st.session_state.last_spin, multiplicateur_input)
            st.session_state.last_spin = None

# Bouton fin historique et commencer simulation
if st.button("Fin historique et calculer résultats"):
    results, bankroll_fin = compute_bankroll(st.session_state.history, st.session_state.base_unit)
    st.session_state.spin_results = results
    df = pd.DataFrame(results)
    st.subheader("Tableau des spins")
    st.dataframe(df)
    # Graphique bankroll
    st.subheader("Courbe Bankroll spin par spin")
    plt.figure(figsize=(10,4))
    plt.plot(df.index+1, df["Bankroll"], marker='o')
    plt.xlabel("Spin #")
    plt.ylabel("Bankroll ($)")
    plt.grid(True)
    st.pyplot(plt)
    # Alerte bankroll critique
    if results[-1]["Bankroll"] <= bankroll_critique:
        st.warning(f"Bankroll critique atteinte: {results[-1]['Bankroll']}$ - no-bet recommandé")

# --- Suggestions mises live ---
st.header("Live Spin - Entrer spin actuel")
col1, col2 = st.columns(len(segments))
for s in segments:
    if col1.button(s, key=f"live_{s}"):
        st.session_state.last_spin = s

mult_input = st.number_input("Multiplicateur manuel du spin", 1, 100, 1, 1)
if st.button("Valider Spin"):
    if st.session_state.last_spin:
        add_spin_to_history(st.session_state.last_spin, mult_input)
        results, bankroll_fin = compute_bankroll(st.session_state.history, st.session_state.base_unit)
        st.session_state.spin_results = results
        df = pd.DataFrame(results)
        st.subheader("Spin ajouté et bankroll mis à jour")
        st.dataframe(df)
        # Graphique
        plt.figure(figsize=(10,4))
        plt.plot(df.index+1, df["Bankroll"], marker='o')
        plt.xlabel("Spin #")
        plt.ylabel("Bankroll ($)")
        plt.grid(True)
        st.pyplot(plt)
        st.session_state.last_spin = None
        if results[-1]["Bankroll"] <= bankroll_critique:
            st.warning(f"Bankroll critique atteinte: {results[-1]['Bankroll']}$ - no-bet recommandé")

# --- Simulation batch ---
st.header("Simulation batch de variantes")
batch_units = st.multiselect("Sélectionner unités de mise de base pour batch", [0.2, 0.5, 1.0, 2.0, 5.0], default=[1.0])
if st.button("Lancer simulation batch"):
    batch_results = {}
    for unit in batch_units:
        recs, final_bankroll = compute_bankroll(st.session_state.history, unit)
        batch_results[f"Unité {unit}$"] = recs
    st.session_state.batch_results = batch_results
    for key, recs in batch_results.items():
        st.subheader(f"Résultats pour {key}")
        df_batch = pd.DataFrame(recs)
        st.dataframe(df_batch)
        plt.figure(figsize=(10,4))
        plt.plot(df_batch.index+1, df_batch["Bankroll"], marker='o')
        plt.xlabel("Spin #")
        plt.ylabel("Bankroll ($)")
        plt.title(key)
        plt.grid(True)
        st.pyplot(plt)
