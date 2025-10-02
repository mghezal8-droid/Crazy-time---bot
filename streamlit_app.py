# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Crazy Time Bot - Auto Strategy (Updated)", layout="wide")

# ------------------------
# SESSION INIT
# ------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {Spin, Résultat, Multiplicateur}
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame(columns=[
        "Spin", "Résultat", "Multiplicateur", "Total Mise", "Gain Net", "Bankroll", "Stratégie"
    ])
if "bankroll" not in st.session_state:
    st.session_state.bankroll = 150.0
if "initial_bankroll" not in st.session_state:
    st.session_state.initial_bankroll = float(st.session_state.bankroll)
if "base_unit" not in st.session_state:
    st.session_state.base_unit = 1.0  # unit value in $ (can be 1 or 0.5)
if "chosen_strategy" not in st.session_state:
    st.session_state.chosen_strategy = None
if "next_mises" not in st.session_state:
    st.session_state.next_mises = {}
if "mode_live" not in st.session_state:
    st.session_state.mode_live = False
if "last_gain" not in st.session_state:
    st.session_state.last_gain = 0.0
if "last_spin" not in st.session_state:
    st.session_state.last_spin = None

# ------------------------
# CONFIG / PAYOUTS
# ------------------------
# Using the formula you provided:
# When number appears: 1 -> 2x, 2 -> 3x, 5 -> 6x, 10 -> 11x
NUMBER_PAYOUT = {"1": 2, "2": 3, "5": 6, "10": 11}
BONUSES = ["CoinFlip", "Pachinko", "CashHunt", "CrazyTime"]
ALL_SEGMENTS = ["1", "2", "5", "10"] + BONUSES

# ------------------------
# STRATEGY DEFINITIONS (unit param in $)
# ------------------------
def strategy_martingale_1(unit):
    return {"1": unit, "2": 0.0, "5": 0.0, "10": 0.0,
            "CoinFlip": 0.0, "Pachinko": 0.0, "CashHunt": 0.0, "CrazyTime": 0.0}

def strategy_godmode_2_5_10(unit):
    return {"1": 0.0, "2": 2.0 * unit, "5": 1.0 * unit, "10": 1.0 * unit,
            "CoinFlip": 0.0, "Pachinko": 0.0, "CashHunt": 0.0, "CrazyTime": 0.0}

def strategy_godmode_2_5_10_bonus(unit):
    return {"1": 0.0, "2": 2.0 * unit, "5": 1.0 * unit, "10": 1.0 * unit,
            "CoinFlip": 0.2 * unit, "Pachinko": 0.2 * unit, "CashHunt": 0.2 * unit, "CrazyTime": 0.2 * unit}

def strategy_1_plus_bonus(unit):
    return {"1": 0.5 * unit, "2": 0.2 * unit, "5": 0.2 * unit, "10": 0.2 * unit,
            "CoinFlip": 0.2 * unit, "Pachinko": 0.2 * unit, "CashHunt": 0.2 * unit, "CrazyTime": 0.2 * unit}

STRATEGY_FUNCS = {
    "Martingale1": strategy_martingale_1,
    "GodMode2_5_10": strategy_godmode_2_5_10,
    "GodMode2_5_10_Bonus": strategy_godmode_2_5_10_bonus,
    "1+Bonus": strategy_1_plus_bonus
}

# ------------------------
# GAIN CALCULATIONS
# ------------------------
def calc_gain_net(result: str, mises: dict, mult: float):
    total_mise = round(sum(mises.values()), 6)
    mise_on_result = mises.get(result, 0.0)
    gain_brut = 0.0
    if result in NUMBER_PAYOUT:
        payout = NUMBER_PAYOUT[result]
        gain_brut = mise_on_result * payout + mise_on_result
        gain_net = gain_brut - (total_mise - mise_on_result)
    elif result in BONUSES:
        gain_brut = mise_on_result * mult + mise_on_result
        gain_net = gain_brut - (total_mise - mise_on_result)
    else:
        # fallback
        gain_net = - total_mise
    return total_mise, round(gain_brut, 6), round(gain_net, 6)

# ------------------------
# HISTORY -> PROBS -> CHOOSE STRATEGY
# ------------------------
def compute_probs_from_history(history):
    counts = {seg: 0 for seg in ALL_SEGMENTS}
    for h in history:
        res = h["Résultat"]
        if res in counts:
            counts[res] += 1
    total = max(len(history), 1)
    probs = {seg: counts[seg] / total for seg in ALL_SEGMENTS}
    return probs

def expected_net_for_strategy(mises: dict, probs: dict, avg_bonus_mult=1.0):
    expected = 0.0
    total_mise = sum(mises.values())
    for seg in ALL_SEGMENTS:
        n = mises.get(seg, 0.0)
        if seg in NUMBER_PAYOUT:
            payout = NUMBER_PAYOUT[seg]
            gain_brut = n * payout + n
            gain_net = gain_brut - (total_mise - n)
        else:
            gain_brut = n * avg_bonus_mult + n
            gain_net = gain_brut - (total_mise - n)
        expected += probs.get(seg, 0) * gain_net
    return expected

def choose_best_strategy(history, bankroll, unit):
    probs = compute_probs_from_history(history)
    best = None
    best_expect = -1e12
    for name, func in STRATEGY_FUNCS.items():
        mises = func(unit)
        # remove bet on last bonus
        last = history[-1]["Résultat"] if len(history) > 0 else None
        if last in BONUSES:
            mises = mises.copy()
            mises[last] = 0.0
        total = sum(mises.values())
        if total > 0 and total > bankroll:
            scale = bankroll / total
            mises = {k: v * scale for k, v in mises.items()}
        expect = expected_net_for_strategy(mises, probs, avg_bonus_mult=1.0)
        if expect > best_expect:
            best_expect = expect
            best = (name, mises, expect)
    return best

# ------------------------
# MARTINGALE NEXT MISES
# ------------------------
def next_mises_after_result(last_mises, last_gain, bankroll):
    warning = ""
    if last_gain is None or last_gain >= 0:
        return None, ""
    doubled = {k: v * 2.0 for k, v in last_mises.items()}
    total = sum(doubled.values())
    if total > bankroll and total > 0:
        scale = bankroll / total
        doubled = {k: v * scale for k, v in doubled.items()}
        warning = f"⚠️ Bankroll insuffisant pour doubler intégralement ({total:.2f}$). Mise ajustée à {bankroll:.2f}$."
    return doubled, warning

# ------------------------
# DYNAMIC UNIT ADJUSTMENT (optimize to bankroll)
# ------------------------
def adjust_unit_by_bankroll(unit, bankroll, initial_bankroll,
                            up_threshold=2.0, down_threshold=0.5,
                            up_factor=2.0, down_factor=0.5, min_unit=0.1, max_unit=100.0):
    """
    Adjust the per-segment unit according to bankroll vs initial_bankroll.
    - if bankroll >= initial_bankroll * up_threshold => unit *= up_factor
    - if bankroll <= initial_bankroll * down_threshold => unit *= down_factor
    This returns new unit (clamped).
    """
    new_unit = unit
    if initial_bankroll <= 0:
        return unit
    ratio = bankroll / initial_bankroll
    if ratio >= up_threshold:
        new_unit = unit * up_factor
    elif ratio <= down_threshold:
        new_unit = unit * down_factor
    new_unit = max(min_unit, min(max_unit, new_unit))
    return round(new_unit, 6)

# ------------------------
# UI: Sidebar controls
# ------------------------
st.sidebar.title("Paramètres")
st.sidebar.markdown("**Crazy Time Bot — historique → analyse → suggestions**")

# Bankroll input & initial recording
bankroll_input = st.sidebar.number_input("Bankroll initiale ($)", min_value=10.0, max_value=100000.0,
                                         value=float(st.session_state.bankroll), step=10.0)
if st.sidebar.button("Valider Bankroll initiale"):
    st.session_state.bankroll = float(bankroll_input)
    st.session_state.initial_bankroll = float(bankroll_input)
    st.success(f"Bankroll initiale: {st.session_state.initial_bankroll:.2f}$")

# unit selection (base)
unit_choice = st.sidebar.radio("Unité de base (par unité):", ("1.0 $ par unité (base ~13$)", "0.5 $ par unité (base ~6.5$)"))
st.session_state.base_unit = 1.0 if unit_choice.startswith("1.0") or unit_choice.startswith("1 ") else 0.5

st.sidebar.markdown("---")
st.sidebar.markdown("**Paramètres d'optimisation des mises**")
up_threshold = st.sidebar.number_input("Seuil augmentation unit (x initial)", value=2.0, step=0.1)
down_threshold = st.sidebar.number_input("Seuil réduction unit (x initial)", value=0.5, step=0.1)
up_factor = st.sidebar.number_input("Facteur augmentation unit", value=2.0, step=0.1)
down_factor = st.sidebar.number_input("Facteur réduction unit", value=0.5, step=0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Simulation batch** (cocher pour activer)")
batch_enabled = st.sidebar.checkbox("Activer simulation batch", value=False)
batch_units_text = st.sidebar.text_input("Unités test (virgule séparé), ex: 0.5,1,2", value="0.5,1.0,2.0")
if st.sidebar.button("Réinitialiser tout (clear history & results)"):
    st.session_state.history = []
    st.session_state.results_df = pd.DataFrame(columns=[
        "Spin", "Résultat", "Multiplicateur", "Total Mise", "Gain Net", "Bankroll", "Stratégie"
    ])
    st.session_state.bankroll = float(bankroll_input)
    st.session_state.initial_bankroll = float(bankroll_input)
    st.session_state.chosen_strategy = None
    st.session_state.next_mises = {}
    st.session_state.mode_live = False
    st.session_state.last_gain = 0.0
    st.session_state.last_spin = None
    st.experimental_rerun()

st.sidebar.markdown("---")
manual_mult = st.sidebar.number_input("Multiplicateur manuel (pour ajouter spins)", min_value=1.0, value=1.0, step=0.5)

st.sidebar.subheader("Ajouter au journal (Historique)")
for seg in ALL_SEGMENTS:
    if st.sidebar.button(f"{seg} ➕"):
        spin_num = len(st.session_state.history) + 1
        st.session_state.history.append({"Spin": spin_num, "Résultat": seg, "Multiplicateur": manual_mult})
        st.sidebar.success(f"Ajouté spin {spin_num}: {seg} x{manual_mult}")

if st.sidebar.button("🗑 Supprimer dernier historique"):
    if len(st.session_state.history) > 0:
        removed = st.session_state.history.pop()
        st.sidebar.info(f"Supprimé spin {removed['Spin']} {removed['Résultat']}")
    else:
        st.sidebar.warning("Rien à supprimer.")

# Finish history -> analyze
if st.sidebar.button("✅ Fin historique et commencer"):
    st.session_state.bankroll = float(bankroll_input)
    st.session_state.initial_bankroll = float(bankroll_input)
    # choose best strategy and set next_mises
    best = choose_best_strategy(st.session_state.history, st.session_state.bankroll, st.session_state.base_unit)
    if best is None:
        st.sidebar.error("Historique vide, impossible d'analyser.")
    else:
        name, mises, expect = best
        st.session_state.chosen_strategy = name
        # block the bonus that occurred on last history spin
        last = st.session_state.history[-1]["Résultat"] if len(st.session_state.history) > 0 else None
        if last in BONUSES:
            mises = mises.copy()
            mises[last] = 0.0
        # scale if over bankroll
        total = sum(mises.values())
        if total > st.session_state.bankroll and total > 0:
            scale = st.session_state.bankroll / total
            mises = {k: v * scale for k, v in mises.items()}
        st.session_state.next_mises = {k: round(v, 6) for k, v in mises.items()}
        st.session_state.mode_live = True
        st.session_state.results_df = pd.DataFrame()  # reset simulation results
        st.session_state.last_gain = 0.0
        st.session_state.last_spin = last
        st.success(f"Stratégie choisie : {name} | Mise totale initiale: {sum(mises.values()):.2f}$ | Expectation ≈ {expect:.4f}")

# ------------------------
# MAIN LAYOUT
# ------------------------
st.title("🎡 Crazy Time — Bot d'analyse & optimisation")

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Historique saisi (entrée brute)")
    if len(st.session_state.history) == 0:
        st.info("Aucun spin saisi — utilise la barre latérale pour ajouter des spins.")
    else:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
with col2:
    st.subheader("Paramètres courants")
    st.write(f"Bankroll active : **{st.session_state.bankroll:.2f}$** (initiale {st.session_state.initial_bankroll:.2f}$)")
    st.write(f"Base unit : **{st.session_state.base_unit:.2f}$**")
    if st.session_state.chosen_strategy:
        st.write(f"Stratégie choisie automatiquement : **{st.session_state.chosen_strategy}**")
        st.write("Mises suggérées (prochain spin) :")
        if st.session_state.next_mises:
            nm_df = pd.DataFrame([{"Segment": k, "Mise ($)": v, "Unité": round(v / st.session_state.base_unit if st.session_state.base_unit>0 else 0, 4)} for k, v in st.session_state.next_mises.items()])
            st.dataframe(nm_df, use_container_width=True)
            st.info(f"Mise totale suggérée : {sum(st.session_state.next_mises.values()):.2f}$")
        else:
            st.write("—")

st.markdown("---")

# ------------------------
# LIVE MODE
# ------------------------
if st.session_state.mode_live:
    st.subheader("🔴 Mode Live — entrer le résultat du spin")
    live_mult = st.number_input("Multiplicateur Live (pour bonus)", min_value=1.0, value=1.0, step=0.5)

    # Show next suggested
    st.write("Mises ACTUELLEMENT SUGGÉRÉES (utilisées pour le prochain spin):")
    if st.session_state.next_mises:
        display_nm = pd.DataFrame([{"Segment": k, "Mise ($)": round(v, 6), "Unites": round(v / st.session_state.base_unit if st.session_state.base_unit>0 else 0, 4)} for k, v in st.session_state.next_mises.items()])
        st.dataframe(display_nm, use_container_width=True)
        st.write(f"Total proposé : **{sum(st.session_state.next_mises.values()):.2f}$**")
    else:
        st.write("Aucune mise suggérée.")

    st.markdown("Clique sur le segment qui est sorti pour enregistrer le spin.")

    cols = st.columns(4)
    for i, seg in enumerate(ALL_SEGMENTS):
        if cols[i % 4].button(f"{seg}"):
            # Use current suggested mises (stakes user will place)
            mises_used = st.session_state.next_mises.copy()
            # block betting on bonus that was previous spin
            if st.session_state.last_spin in BONUSES:
                mises_used = mises_used.copy()
                mises_used[st.session_state.last_spin] = 0.0
            total_mise, gain_brut, gain_net = calc_gain_net(seg, mises_used, live_mult)
            new_bankroll = round(st.session_state.bankroll + gain_net, 6)

            # append result
            next_spin_index = len(st.session_state.results_df) + 1 if not st.session_state.results_df.empty else 1
            row = {
                "Spin": next_spin_index,
                "Résultat": seg,
                "Multiplicateur": live_mult,
                "Total Mise": round(total_mise, 6),
                "Gain Net": round(gain_net, 6),
                "Bankroll": round(new_bankroll, 6),
                "Stratégie": st.session_state.chosen_strategy
            }
            st.session_state.results_df = pd.concat([st.session_state.results_df, pd.DataFrame([row])], ignore_index=True)
            st.session_state.bankroll = new_bankroll
            st.session_state.last_gain = gain_net
            st.session_state.last_spin = seg

            # adjust unit by bankroll BEFORE computing next base strategy
            new_unit = adjust_unit_by_bankroll(
                st.session_state.base_unit,
                st.session_state.bankroll,
                st.session_state.initial_bankroll,
                up_threshold=up_threshold,
                down_threshold=down_threshold,
                up_factor=up_factor,
                down_factor=down_factor
            )
            # if unit changed, update base_unit (this will affect strategy base next)
            unit_changed = False
            if new_unit != st.session_state.base_unit:
                unit_changed = True
                st.session_state.base_unit = new_unit
                st.info(f"Unité ajustée automatiquement à {st.session_state.base_unit:.4f}$ selon bankroll.")

            # compute next mises (martingale doubling after loss)
            next_mises, warning = next_mises_after_result(st.session_state.next_mises, st.session_state.last_gain, st.session_state.bankroll)
            if next_mises is None:
                # reset to chosen strategy base using possibly updated unit
                base_func = STRATEGY_FUNCS[st.session_state.chosen_strategy]
                base_mises = base_func(st.session_state.base_unit)
                # block last bonus if necessary
                if st.session_state.last_spin in BONUSES:
                    base_mises = base_mises.copy()
                    base_mises[st.session_state.last_spin] = 0.0
                # scale if over bankroll
                total_tmp = sum(base_mises.values())
                if total_tmp > st.session_state.bankroll and total_tmp > 0:
                    sc = st.session_state.bankroll / total_tmp
                    base_mises = {k: v * sc for k, v in base_mises.items()}
                st.session_state.next_mises = {k: round(v, 6) for k, v in base_mises.items()}
            else:
                st.session_state.next_mises = {k: round(v, 6) for k, v in next_mises.items()}

            # critical bank alert: if bankroll less than X% of total suggested next_mises
            total_next = sum(st.session_state.next_mises.values())
            if total_next > 0 and st.session_state.bankroll < 0.5 * total_next:
                st.warning("⚠️ Bankroll critique : insuffisant (<50% de la mise totale suggérée) — envisager no-bet ou réduire unité.")

            st.success(f"Spin {next_spin_index} enregistré : {seg} x{live_mult} | Gain net: {gain_net:.2f}$ | Bankroll: {new_bankroll:.2f}$")
            if warning:
                st.warning(warning)

# ------------------------
# BATCH SIMULATION (if enabled)
# ------------------------
st.markdown("---")
st.subheader("🔬 Simulation batch (compare unités de base)")

if batch_enabled:
    units_list = []
    try:
        units_list = [float(u.strip()) for u in batch_units_text.split(",") if u.strip() != ""]
    except Exception:
        st.error("Format unités invalide. Utilise ex: 0.5,1,2")
        units_list = []

    if st.button("Lancer simulation batch"):
        if len(units_list) == 0:
            st.error("Aucune unité fournie.")
        else:
            # For each unit simulate running the *entered history* then play forward with current chosen strategy logic
            sim_results = []
            for unit in units_list:
                sim_bankroll = float(st.session_state.initial_bankroll)
                # choose best strategy for that unit using historical data
                best = choose_best_strategy(st.session_state.history, sim_bankroll, unit)
                if best is None:
                    st.error("Historique vide pour simuler.")
                    break
                name, mises, expect = best
                # ensure last bonus removed
                last_hist = st.session_state.history[-1]["Résultat"] if len(st.session_state.history) > 0 else None
                if last_hist in BONUSES:
                    mises = mises.copy()
                    mises[last_hist] = 0.0
                # apply sequence of historical spins (simulate) using chosen mises + martingale rules
                sim_bankroll_history = [sim_bankroll]
                last_mises = {k: round(v, 6) for k, v in mises.items()}
                last_gain_local = 0.0
                for h in st.session_state.history:
                    res = h["Résultat"]
                    mult = h["Multiplicateur"]
                    total_mise, gb, gn = calc_gain_net(res, last_mises, mult)
                    sim_bankroll = sim_bankroll + gn
                    sim_bankroll_history.append(sim_bankroll)
                    # compute next_mises via martingale rule
                    nm, _ = next_mises_after_result(last_mises, gn, sim_bankroll)
                    if nm is None:
                        # reset base
                        last_mises = {k: round(v, 6) for k, v in STRATEGY_FUNCS[name](unit).items()}
                        if res in BONUSES:
                            last_mises[res] = 0.0
                    else:
                        last_mises = {k: round(v, 6) for k, v in nm.items()}
                sim_results.append({
                    "unit": unit,
                    "final_bankroll": sim_bankroll,
                    "strategy": name,
                    "history_len": len(st.session_state.history)
                })
            # show summary
            st.write(pd.DataFrame(sim_results))
            # plot final bankroll comparison
            st.success("Simulation batch terminée.")

# ------------------------
# RESULTS TABLE & GRAPH
# ------------------------
st.markdown("---")
st.subheader("📊 Résultats / Simulation (depuis 'Fin historique et commencer')")

if not st.session_state.results_df.empty:
    st.dataframe(st.session_state.results_df.astype({"Spin": int}), use_container_width=True)
    st.subheader("📈 Bankroll (spin by spin)")
    try:
        chart_df = st.session_state.results_df.set_index("Spin")["Bankroll"].astype(float)
        st.line_chart(chart_df)
    except Exception:
        st.write("Erreur graphique — données manquantes.")
else:
    st.info("Aucune simulation commencée. Cliquez 'Fin historique et commencer' après avoir saisi l'historique.")

# ------------------------
# FOOTER NOTES
# ------------------------
st.markdown("---")
st.markdown(
    "- Calculs utilisés:\n"
    "  - Numéros (1/2/5/10): gain_brut = mise_on_seg * payout + mise_on_seg ; gain_net = gain_brut - (total_mise - mise_on_seg)\n"
    "  - Bonus: gain_brut = mise_on_bonus * multiplicateur_manual + mise_on_bonus ; gain_net = gain_brut - (total_mise - mise_on_bonus)\n"
    "  - Perte si segment non couvert = - total_mise\n"
    "- Martingale stricte: double toutes les mises après une perte, reset après gain. Ajustement proportionnel si doublage dépasse bankroll.\n"
    "- Optimisation unité: l'unité de mise s'ajuste automatiquement selon la progression du bankroll par rapport à la bankroll initiale (seuils et facteurs paramétrables dans la sidebar).\n"
    "- Simulation batch: teste différentes unités de base sur l'historique saisi.\n"
)
