# app.py  — Streamlit EV/Kelly калькулятор із S-кривою та Монте-Карло
import math, json
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
import math
from datetime import datetime

st.set_page_config(page_title="EV/Kelly Betting Calculator", page_icon="⚽", layout="centered")
if "bet_log" not in st.session_state:
    st.session_state["bet_log"] = []  # список записів журналу (dict)

# ======= ПРЕСЕТИ (можна редагувати в UI) =======
DEFAULT_HFA = {"EPL":0.13, "Serie A":0.15, "Ligue 1":0.14, "LaLiga":0.13, "Bundesliga":0.15}
# (alpha, beta) для S-кривої: P = 1/(1 + alpha * x^beta), x = D_fav_clean - 1
DEFAULT_SCURVE = {
    "EPL":{"home":(0.95,0.85), "away":(1.10,0.90)},
    "Serie A":{"home":(1.00,0.95), "away":(1.12,0.95)},
    "Ligue 1":{"home":(1.05,0.90), "away":(1.15,0.95)},
    "LaLiga":{"home":(0.98,0.90), "away":(1.10,0.92)},
    "Bundesliga":{"home":(0.96,0.88), "away":(1.08,0.90)}
}

# ======= СЛУЖБОВІ ФУНКЦІЇ =======
def de_vig_three(Dh, Dd, Da):
    """Очистити маржу для 1X2 і повернути чисті дека та імпліцитні ймовірності"""
    p = np.array([1/float(Dh), 1/float(Dd), 1/float(Da)], dtype=float)
    S = p.sum()
    if S == 0: S = 1e-9
    p_clean = p / S
    D_clean = 1.0 / p_clean
    return D_clean, p_clean

def de_vig_two(Da, Db):
    p = np.array([1/float(Da), 1/float(Db)], dtype=float)
    S = p.sum()
    if S == 0: S = 1e-9
    p_clean = p / S
    D_clean = 1.0 / p_clean
    return D_clean, p_clean

def s_curve_prob(D_fav_clean, alpha, beta):
    x = max(D_fav_clean - 1.0, 1e-9)
    return 1.0 / (1.0 + alpha * (x ** beta))

def build_lambdas(xG_h_tot, xGA_h_tot, M_h, xG_a_tot, xGA_a_tot, M_a, HFA=0.15,
                  form_mix=None, xpts_delta_home=0.0, xpts_delta_away=0.0,
                  key_att_home=True, key_att_away=True, key_def_home=True, key_def_away=True):
    # усереднені за матч
    xG_h_pg  = (xG_h_tot / max(M_h,1))
    xGA_h_pg = (xGA_h_tot/ max(M_h,1))
    xG_a_pg  = (xG_a_tot / max(M_a,1))
    xGA_a_pg = (xGA_a_tot/ max(M_a,1))

    lam_h = (xG_h_pg + xGA_a_pg)/2.0 + HFA
    lam_a = (xG_a_pg + xGA_h_pg)/2.0

    # корекції ключових гравців (акуратні)
    if not key_att_home: lam_h *= 0.90
    if not key_att_away: lam_a *= 0.90
    if not key_def_home: lam_a *= 1.05
    if not key_def_away: lam_h *= 1.05

    # легка корекція xPTS-дельтою (недо/пере-виконання)
    if xpts_delta_home > 2: lam_h += 0.03
    if xpts_delta_home < -2: lam_h -= 0.03
    if xpts_delta_away > 2: lam_a += 0.03
    if xpts_delta_away < -2: lam_a -= 0.03

    return max(lam_h, 0.05), max(lam_a, 0.05)

def simulate_bivariate_poisson(n, lam_h, lam_a, rho=0.07, rng=None):
    """Біваріатний Пуассон через спільний компонент; апроксимація λc=rho*sqrt(λh*λa)"""
    rng = rng or np.random.default_rng()
    lam_c = max(rho * math.sqrt(lam_h * lam_a), 0.0)
    lam1 = max(lam_h - lam_c, 1e-6)
    lam2 = max(lam_a - lam_c, 1e-6)
    g1 = rng.poisson(lam1, n)
    g2 = rng.poisson(lam2, n)
    gc = rng.poisson(lam_c, n)
    H = g1 + gc
    A = g2 + gc
    return H, A

def probs_from_scores(H, A, totals_lines, ah_lines):
    n = len(H)
    # 1X2
    p_home = float((H > A).mean())
    p_draw = float((H == A).mean())
    p_away = float((H < A).mean())

    # Totals
    totals = {}
    S = H + A
    for tag, (typ, line) in totals_lines.items():
        if typ == "U":
            totals[tag] = float((S <  line - 1e-9).mean())
        else:
            totals[tag] = float((S >  line + 1e-9).mean())

    # BTTS
    p_btts_yes = float(((H>=1) & (A>=1)).mean())
    p_btts_no  = 1.0 - p_btts_yes

    # AH (європейський/класичний, без повернень; для .25/.75 розщепимо нижче)
    ah = {}
    diff = H - A
    for tag, (side, line) in ah_lines.items():
        # євро-версія виплат: win/lose (без повернення)
        if side == "HOME":
            ah[tag] = float((diff > line).mean())
        else:
            ah[tag] = float((diff < -line).mean())
    return (p_home, p_draw, p_away), totals, (p_btts_yes, p_btts_no), ah

def implied_from_odds(D):  # без де-вигу, використовується для відображення
    return 1.0/float(D)

def value_ev_kelly(p_fair, D):
    p_imp = 1.0/float(D)
    value = p_fair - p_imp
    ev_per_uah = p_fair*(D-1.0) - (1.0 - p_fair)
    b = D - 1.0
    if b <= 0: return p_imp, value, ev_per_uah, 0.0
    kelly = ((b * p_fair) - (1.0 - p_fair)) / b
    kelly = max(kelly, 0.0)
    return p_imp, value, ev_per_uah, kelly

def scale_stakes_to_bank(stakes, bank, min_stake):
    stakes = np.array(stakes, dtype=float)
    # підтягуємо все нижче мінімуму до мінімуму (0 залишаємо 0)
    stakes = np.where((stakes>0) & (stakes<min_stake), min_stake, stakes)
    s = stakes.sum()
    if s <= bank or s==0: return stakes
    return stakes * (bank / s)

def conflict_key(row):
    mkt = row["market_type"]
    if mkt=="BTTS":
        return f"BTTS_{row['match']}"
    if mkt=="TOTAL":
        return f"TOTAL_{row['match']}"
    if mkt=="ML":
        return f"ML_{row['match']}"
    if mkt=="AH":
        return f"AH_{row['match']}"
    return f"GEN_{row['match']}"

# ======= UI =======
st.title("⚽ EV/Kelly калькулятор (Streamlit)")

with st.sidebar:
    st.subheader("Банкролл")
    BANK = st.number_input("Банк (грн)", value=100, step=10)
    MIN_STAKE = st.number_input("Мінімальна ставка (грн)", value=10, step=1)
    kelly_mode = st.radio("Режим Kelly", ["1/2 Kelly","Full Kelly","1/3 Kelly"], index=0)
    value_threshold = st.slider("Поріг Value (п.п.)", 0.0, 10.0, 3.0, 0.5)
    max_markets_per_match = st.slider("Макс. ринків на матч", 1, 5, 3)
    st.caption("Порада: 1/2 Kelly — менш волатильно.")

st.markdown("### 1) Додай/заповни матч")
colA, colB = st.columns(2)
with colA:
    league = st.selectbox("Ліга", list(DEFAULT_HFA.keys()), index=1)
    datetime_str = st.text_input("Дата/час (вільний формат)", "")
with colB:
    match = st.text_input("Матч (Home vs Away)", "Roma vs Inter")
    fav_side = st.selectbox("Фаворит для 1X2 prior", ["Home","Away"], index=0)

with st.expander("Коефіцієнти (десяткові) + де-вiгориш"):
    devig_on = st.checkbox("Очищати маржу (де-вiгориш)", value=True)
    st.markdown("**1X2**")
    D_home = st.number_input("D_home", value=3.39, step=0.01, format="%.2f")
    D_draw = st.number_input("D_draw", value=3.33, step=0.01, format="%.2f")
    D_away = st.number_input("D_away", value=2.30, step=0.01, format="%.2f")

    st.markdown("**Totals (U/O)** — формат: U2.5@1.76;O2.5@2.10 (через крапку з комою)")
    totals_str = st.text_input("Totals", "U2.5@1.76;O2.5@2.10")

    st.markdown("**AH (HOME/ AWAY)** — формат: HOME-1.5@2.17;AWAY+1.5@1.70")
    ah_str = st.text_input("AH", "")

    st.markdown("**BTTS**")
    D_btts_yes = st.number_input("D_BTTS_Yes", value=1.87, step=0.01, format="%.2f")
    D_btts_no  = st.number_input("D_BTTS_No",  value=1.93, step=0.01, format="%.2f")

with st.expander("Дані для симуляції (xG/xGA → λ)"):
    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.write("**Home (з Understat)**")
        xG_h_tot  = st.number_input("Home xG total",  value=7.22, step=0.01)
        xGA_h_tot = st.number_input("Home xGA total", value=5.05, step=0.01)
        M_h       = st.number_input("Home Matches",   value=6, step=1)
        xPTS_h    = st.number_input("Home xPTS (опц.)", value=9.98, step=0.01)
        PTS_h     = st.number_input("Home PTS (опц.)",  value=15.0, step=1.0)
        key_att_h = st.checkbox("Ключовий атакер у строю (home)", True)
        key_def_h = st.checkbox("Ключовий захисник у строю (home)", True)
    with col2:
        st.write("**Away (з Understat)**")
        xG_a_tot  = st.number_input("Away xG total",  value=17.03, step=0.01)
        xGA_a_tot = st.number_input("Away xGA total", value=4.73, step=0.01)
        M_a       = st.number_input("Away Matches",   value=6, step=1)
        xPTS_a    = st.number_input("Away xPTS (опц.)", value=14.41, step=0.01)
        PTS_a     = st.number_input("Away PTS (опц.)",  value=18.0, step=1.0)
        key_att_a = st.checkbox("Ключовий атакер у строю (away)", True)
        key_def_a = st.checkbox("Ключовий захисник у строю (away)", True)

    rho = st.slider("Кореляція атак ρ", 0.0, 0.2, 0.07, 0.01)
    HFA = st.number_input(f"HFA для {league}", value=float(DEFAULT_HFA[league]), step=0.01, format="%.2f")

with st.expander("S-крива (prior для 1X2)"):
    side_key = "home" if fav_side=="Home" else "away"
    alpha_def, beta_def = DEFAULT_SCURVE.get(league, DEFAULT_SCURVE["Serie A"])[side_key]
    alpha = st.number_input("α", value=float(alpha_def), step=0.01, format="%.2f")
    beta  = st.number_input("β", value=float(beta_def),  step=0.01, format="%.2f")
    blend = st.slider("Вага prior у бленді 1X2 (λ_blend)", 0.0, 1.0, 0.4, 0.05)

# ======= Обробка рядків Totals/AH =======
def parse_totals(s):
    items={}
    s=s.strip()
    if not s: return items
    parts = [p.strip() for p in s.split(";") if p.strip()]
    for p in parts:
        try:
            side, rest = p[0], p[1:]
            line_str, odd_str = rest.split("@")
            key = f"{side}{line_str}"
            items[key]=(side, float(line_str))
            items[key+"_ODDS"]=float(odd_str)
        except:
            pass
    return items

def parse_ah(s):
    items={}
    s=s.strip()
    if not s: return items
    parts = [p.strip() for p in s.split(";") if p.strip()]
    for p in parts:
        try:
            side, rest = p.split("@")[0].split("-")[0], p.split("@")[0]
            odd = float(p.split("@")[1])
            # rest прикш: HOME-1.5, AWAY+1.0
            if "HOME" in rest:
                line = float(rest.split("HOME")[1])
                side="HOME"
            else:
                line = float(rest.split("AWAY")[1])
                side="AWAY"
            key = f"{side}{line:+.2f}"
            items[key]=(side, float(line))
            items[key+"_ODDS"]=odd
        except:
            pass
    return items

totals_def = parse_totals(totals_str)
ah_def     = parse_ah(ah_str)

# ======= РОЗРАХУНОК =======
if st.button("🔎 Розрахувати"):
    # 1) де-вiгориш
    Dh, Dd, Da = D_home, D_draw, D_away
    if devig_on:
        (Dh_c, Dd_c, Da_c), (pH_c, pD_c, pA_c) = de_vig_three(Dh, Dd, Da)
    else:
        Dh_c, Dd_c, Da_c = Dh, Dd, Da
        pH_c, pD_c, pA_c = implied_from_odds(Dh), implied_from_odds(Dd), implied_from_odds(Da)

    # 2) λ-параметри
    xpts_delta_h = (xPTS_h - PTS_h) if PTS_h>0 else 0.0
    xpts_delta_a = (xPTS_a - PTS_a) if PTS_a>0 else 0.0
    lam_h, lam_a = build_lambdas(
        xG_h_tot, xGA_h_tot, M_h, xG_a_tot, xGA_a_tot, M_a, HFA=HFA,
        xpts_delta_home=xpts_delta_h, xpts_delta_away=xpts_delta_a,
        key_att_home=key_att_h, key_att_away=key_att_a,
        key_def_home=key_def_h, key_def_away=key_def_a
    )

    # 3) Монте-Карло
    N = 50000
    H, A = simulate_bivariate_poisson(N, lam_h, lam_a, rho=rho)
    # підготовити словники ліній→ймовірностей
    totals_lines = {k:v for k,v in totals_def.items() if not k.endswith("_ODDS")}
    ah_lines     = {k:v for k,v in ah_def.items() if not k.endswith("_ODDS")}
    (pH_mc, pD_mc, pA_mc), totals_mc, (p_btts_yes, p_btts_no), ah_mc = probs_from_scores(H, A, totals_lines, ah_lines)

    # 4) prior + блендинг для 1X2
    Dfav = Dh_c if fav_side=="Home" else Da_c
    p_prior_fav = s_curve_prob(Dfav, alpha, beta)
    if fav_side=="Home":
        p_home_fair = blend*p_prior_fav + (1-blend)*pH_mc
        p_away_fair = pA_mc   # без змін
    else:
        p_away_fair = blend*p_prior_fav + (1-blend)*pA_mc
        p_home_fair = pH_mc
    # корекція нічиєї (на основі MC)
    p_draw_fair = max(0.0, min(1.0, pD_mc))
    # нормалізація
    ssum = p_home_fair + p_draw_fair + p_away_fair
    if ssum>0:
        p_home_fair, p_draw_fair, p_away_fair = p_home_fair/ssum, p_draw_fair/ssum, p_away_fair/ssum

    # 5) збираємо всі ринки із fair / implied / value / EV / Kelly
    rows = []

    # ML
    for name, D_out, p_fair in [
        (f"ML Home", D_home, p_home_fair),
        (f"ML Draw", D_draw, p_draw_fair),
        (f"ML Away", D_away, p_away_fair),
    ]:
        p_imp, value, ev1, kelly = value_ev_kelly(p_fair, D_out)
        rows.append(dict(
            league=league, match=match, market=f"{name}", market_type="ML",
            odds=D_out, fair_pct=round(p_fair*100,2), implied_pct=round(p_imp*100,2),
            value_pct=round(value*100,2), ev_per_uah=round(ev1,3), kelly_frac=kelly
        ))

    # BTTS
    for name, D_out, p_fair in [
        ("BTTS Yes", D_btts_yes, p_btts_yes),
        ("BTTS No",  D_btts_no,  p_btts_no),
    ]:
        p_imp, value, ev1, kelly = value_ev_kelly(p_fair, D_out)
        rows.append(dict(
            league=league, match=match, market=name, market_type="BTTS",
            odds=D_out, fair_pct=round(p_fair*100,2), implied_pct=round(p_imp*100,2),
            value_pct=round(value*100,2), ev_per_uah=round(ev1,3), kelly_frac=kelly
        ))

    # Totals
    for key,(typ, line) in totals_lines.items():
        D_out = totals_def.get(key+"_ODDS", 1.0)
        p_fair = totals_mc.get(key, 0.0)
        p_imp, value, ev1, kelly = value_ev_kelly(p_fair, D_out)
        rows.append(dict(
            league=league, match=match, market=f"{typ}{line}", market_type="TOTAL",
            odds=D_out, fair_pct=round(p_fair*100,2), implied_pct=round(p_imp*100,2),
            value_pct=round(value*100,2), ev_per_uah=round(ev1,3), kelly_frac=kelly
        ))

    # AH (євро стиль; для .25/.75 у MVP так само оцінюємо як наближено win/lose)
    for key,(side,line) in ah_lines.items():
        D_out = ah_def.get(key+"_ODDS", 1.0)
        p_fair = ah_mc.get(key, 0.0)
        p_imp, value, ev1, kelly = value_ev_kelly(p_fair, D_out)
        rows.append(dict(
            league=league, match=match, market=f"{side}{line:+.2f}", market_type="AH",
            odds=D_out, fair_pct=round(p_fair*100,2), implied_pct=round(p_imp*100,2),
            value_pct=round(value*100,2), ev_per_uah=round(ev1,3), kelly_frac=kelly
        ))

    df = pd.DataFrame(rows)
    # фільтр за Value
    df["pass_value"] = df["value_pct"] >= value_threshold
    st.markdown("### 2) Ринки (Fair/Value/EV/Kelly)")
    st.dataframe(df, hide_index=True, use_container_width=True)

    # 6) портфель: беремо тільки позитивний Kelly і value-прохід
    df_sel = df[(df["kelly_frac"]>0) & (df["pass_value"])]
    if df_sel.empty:
        st.warning("Немає ринків, що проходять поріг Value та Kelly>0. Підкрути вхідні дані.")
    else:
        # чорнові ставки за Kelly режимом
        factor = {"Full Kelly":1.0, "1/2 Kelly":0.5, "1/3 Kelly":1/3}[kelly_mode]
        raw_stakes = (df_sel["kelly_frac"].values * factor * BANK)
        stakes = scale_stakes_to_bank(raw_stakes, BANK, MIN_STAKE)

        # обмеження: макс ринків на матч + антиконфлікти
        df_sel = df_sel.copy()
        df_sel["stake_uah"] = np.round(stakes, 2)

        # прибираємо конфлікти та надлишки на матч
        final_rows=[]
        by_match={}
        taken_keys=set()
        for i, r in df_sel.sort_values("ev_per_uah", ascending=False).iterrows():
            if r["stake_uah"]<=0: continue
            m = r["match"]
            by_match[m]=by_match.get(m,0)
            # ліміт ринків на матч
            if by_match[m] >= max_markets_per_match:
                continue
            # конфлікти
            ck = conflict_key(r)
            if ck in taken_keys and ("TOTAL" in r["market_type"] or "BTTS" in r["market_type"] or "ML" in r["market_type"]):
                continue
            taken_keys.add(ck)
            by_match[m]+=1
            final_rows.append(r.to_dict())

        if not final_rows:
            st.warning("Після застосування правил (ліміти/конфлікти) портфель порожній.")
        else:
            port = pd.DataFrame(final_rows)
            # зріз по банку
            if port["stake_uah"].sum() > BANK:
                port["stake_uah"] = scale_stakes_to_bank(port["stake_uah"].values, BANK, MIN_STAKE)

            port["ev_uah"] = np.round(port["ev_per_uah"] * port["stake_uah"], 2)
            port["ROI_%"]  = np.round(100.0 * port["ev_uah"].sum() / max(port["stake_uah"].sum(),1), 2)

       # === 3) Купон (портфель) ===
st.markdown("### 3) Купон (портфель)")

show_cols = [
    "league","match","market","odds",
    "fair_pct","implied_pct","value_pct",
    "stake_uah","ev_per_uah","ev_uah"
]

if len(port) == 0:
    st.info("Поки що порожньо. Додайте матчі і натисніть «Розрахувати».")
else:
    # Таблиця купона
    st.dataframe(port[show_cols], hide_index=True, use_container_width=True)

    # Експорт купона в CSV
    csv = port[show_cols].to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 Завантажити CSV (купон)", data=csv, file_name="coupon.csv", mime="text/csv")

    # Базовий підсумок
    total_stake = float(port["stake_uah"].sum())
    total_ev    = float(port["ev_uah"].sum())
    roi         = (total_ev / total_stake * 100.0) if total_stake > 0 else 0.0
    st.info(f"Σ ставка: {total_stake:.2f} грн | Σ EV: +{total_ev:.2f} грн | ROI ~ {roi:.1f}%")

    # ---- ПІДСУМОК СЕСІЇ ----
    n_bets    = int(len(port))
    avg_value = float(port["value_pct"].mean()) if n_bets > 0 else 0.0
    avg_odds  = float(port["odds"].mean()) if n_bets > 0 else 0.0

    # 95% CI для ROI (припускаємо незалежність ординарів)
    var_terms = []
    if n_bets > 0:
        for _, r in port.iterrows():
            p   = float(r["fair_pct"]) / 100.0
            b   = float(r["odds"]) - 1.0
            ev1 = float(r["ev_per_uah"])                 # EV per 1 грн
            var1 = p*(b**2) + (1.0-p)*(1.0) - (ev1**2)   # Var(X) для 1 грн
            var_terms.append((float(r["stake_uah"])**2) * var1)

        portfolio_var = sum(var_terms)
        se_roi = (math.sqrt(portfolio_var) / total_stake * 100.0) if total_stake > 0 else 0.0
        ci_low  = roi - 1.96*se_roi
        ci_high = roi + 1.96*se_roi
    else:
        se_roi = 0.0
        ci_low = ci_high = roi

    st.markdown("#### Підсумок сесії")
    st.write(
        f"**Σ ставка:** {total_stake:.2f} грн  |  **Σ EV:** +{total_ev:.2f} грн  |  "
        f"**ROI:** {roi:.2f}%  (95% CI: {ci_low:.2f}% … {ci_high:.2f}%)  |  "
        f"**К-сть ставок:** {n_bets}  |  **Сер. Value:** {avg_value:.2f}%  |  **Сер. коеф.:** {avg_odds:.2f}"
    )

    # ---- ЖУРНАЛ СЕСІЇ ----
    st.session_state.setdefault("bet_log", [])
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for _, r in port.iterrows():
        st.session_state["bet_log"].append({
            "timestamp": ts,
            "league": r["league"],
            "match": r["match"],
            "market": r["market"],
            "odds": float(r["odds"]),
            "fair_pct": float(r["fair_pct"]),
            "implied_pct": float(r["implied_pct"]),
            "value_pct": float(r["value_pct"]),
            "stake_uah": float(r["stake_uah"]),
            "ev_per_uah": float(r["ev_per_uah"]),
            "ev_uah": float(r["ev_uah"]),
            "session_roi_pct": float(roi)
        })

    with st.expander("📒 Журнал сесії (усі обчислені купони)"):
        log_df = pd.DataFrame(st.session_state.get("bet_log", []))
        if len(log_df) == 0:
            st.caption("Журнал порожній — розрахуй хоча б один купон.")
        else:
            st.dataframe(log_df, hide_index=True, use_container_width=True)
            csv_log = log_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇️ Завантажити CSV (журнал)", data=csv_log, file_name="session_log.csv", mime="text/csv")
            if st.button("🧹 Очистити журнал"):
                st.session_state["bet_log"] = []
                st.experimental_rerun()


# ---------- ВІДОБРАЖЕННЯ ЖУРНАЛУ + ЕКСПОРТ ----------
with st.expander("📒 Журнал сесії (усі обчислені купони)"):
    if st.session_state["bet_log"]:
        log_df = pd.DataFrame(st.session_state["bet_log"])
        st.dataframe(log_df, hide_index=True, use_container_width=True)
        csv_log = log_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Завантажити CSV (журнал)", data=csv_log, file_name="session_log.csv", mime="text/csv")
        if st.button("🧹 Очистити журнал"):
            st.session_state["bet_log"] = []
            st.experimental_rerun()
    else:
        st.caption("Журнал порожній — розрахуй хоча б один купон.")
        csv = port[show_cols].to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Завантажити CSV (купон)", data=csv, file_name="coupon_ev_kelly.csv", mime="text/csv")

    st.caption(f"λ_home={lam_h:.2f}, λ_away={lam_a:.2f} • p(ML) MC: H={pH_mc:.2%}, D={pD_mc:.2%}, A={pA_mc:.2%}")
