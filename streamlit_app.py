import os, time, datetime as dt
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="Dr Bleou Hybrid Pro v3", page_icon="📈", layout="wide")

from app.indicators import compute_indicators
from app.strategy import hybrid_pro, momentum_breakout, mean_revert, combine_signals
from app.backtest import run_backtest
from app.bitget_client import fetch_ohlcv, fetch_positions, fetch_balance, place_order
from app.alerts import alert_drawdown, alert_liquidation, alert_margin, alert_high_fees
from app.router import route_orders
from app.utils import tg_send

# ---------- Auth ----------
import yaml, streamlit_authenticator as stauth
with open("auth_config.yaml") as f:
    config = yaml.safe_load(f)
authenticator = stauth.Authenticate(
    config['credentials'], config['cookie']['name'], config['cookie']['key'], config['cookie']['expiry_days']
)
name, authentication_status, username = authenticator.login("Login", "main")
if not authentication_status:
    if authentication_status is False:
        st.error("Login incorrect.")
    st.stop()
authenticator.logout("Logout", "sidebar")
st.sidebar.success(f"Connecté: {name}")

st.title("📈 Dr Bleou Hybrid Pro – v3")

# ---------- Sidebar ----------
default_symbols = [
    "BTC/USDT:USDT","ETH/USDT:USDT","BNB/USDT:USDT","SOL/USDT:USDT","XRP/USDT:USDT",
    "ADA/USDT:USDT","DOGE/USDT:USDT","AVAX/USDT:USDT","MATIC/USDT:USDT","DOT/USDT:USDT",
    "LINK/USDT:USDT","ATOM/USDT:USDT","LTC/USDT:USDT","TRX/USDT:USDT","FIL/USDT:USDT",
    "SHIB/USDT:USDT","PEPE/USDT:USDT","SUI/USDT:USDT","ARB/USDT:USDT","OP/USDT:USDT",
]
symbols = st.sidebar.multiselect("Univers (20 max)", options=default_symbols, default=default_symbols, max_selections=20)
timeframe = st.sidebar.selectbox("Timeframe", ["15m","1h","4h","1d"], index=1)
lookback = st.sidebar.slider("Historique (barres)", 400, 2000, 1000, 100)
mode_live = st.sidebar.radio("Mode", ["Paper Trading","Live (Bitget)"], index=0)

st.sidebar.subheader("Fenêtre US & Frais")
session_start = st.sidebar.number_input("Début UTC", 0, 23, 9)
session_end = st.sidebar.number_input("Fin UTC", 1, 24, 15)
fee_windows = st.sidebar.text_input("Frais élevés (plages '9-10,13-14')", "0-1")

st.sidebar.subheader("Stratégies actives")
use_hybrid = st.sidebar.checkbox("Hybrid Pro", True)
use_momo  = st.sidebar.checkbox("Momentum/Breakout", True)
use_mr    = st.sidebar.checkbox("Mean-Revert", True)

st.sidebar.subheader("Risque & Router")
capital = st.sidebar.number_input("Capital (Paper)", 1000, 1_000_000, 10000, 500)
risk_budget = st.sidebar.number_input("Budget risque (USD) pour router", 100, 100000, 1000, 100)
leverage = st.sidebar.slider("Levier (live)", 1, 20, 5, 1)

st.sidebar.subheader("Alertes & Telegram")
enable_alerts = st.sidebar.checkbox("Activer alertes", True)
dd_thresh = st.sidebar.slider("Seuil Drawdown %", 5, 50, 15, 1) / 100.0
liq_prox = st.sidebar.slider("Proximité liquidation %", 1, 20, 5, 1) / 100.0
min_free = st.sidebar.slider("Ratio marge libre min %", 5, 80, 20, 5) / 100.0
enable_tg_conf = st.sidebar.checkbox("Confirms Telegram (live)", True)

def parse_fee_windows(s):
    wins = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a,b = part.split("-")
            try:
                wins.append((int(a), int(b)))
            except:
                pass
    return wins

@st.cache_data(ttl=300)
def load_symbol(symbol, timeframe, limit):
    data = fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = compute_indicators(df)
    return df

tab_dash, tab_strat, tab_router, tab_alerts, tab_live = st.tabs(
    ["Dashboard","Stratégies","Order Router","Alertes","Exécution Live"]
)

with tab_dash:
    left, right = st.columns([2,1])
    with left:
        sym = st.selectbox("Symbol à visualiser", options=symbols, index=0)
        df = load_symbol(sym, timeframe, lookback).copy()

        # Build signals by selected strategies
        sigs = []
        if use_hybrid:
            sigs.append(("Hybrid", hybrid_pro(df, session_start=session_start, session_end=session_end)))
        if use_momo:
            sigs.append(("Momentum", momentum_breakout(df)))
        if use_mr:
            sigs.append(("MeanRevert", mean_revert(df)))

        # Combine equal weights
        if sigs:
            comb = combine_signals({k:v for k,v in sigs})
            df["signal"] = comb
        else:
            df["signal"] = 0

        fig = px.line(df, x="timestamp", y="close", title=f"{sym} – Prix")
        st.plotly_chart(fig, use_container_width=True)

        bt_df, summary = run_backtest(df, capital=capital, risk_per_trade=0.01)
        fig2 = px.line(bt_df, x="timestamp", y="equity", title="Courbe d'equity (paper backtest)")
        st.plotly_chart(fig2, use_container_width=True)

    with right:
        st.subheader("📊 Métriques")
        st.metric("Trades", summary["trades"])
        st.metric("Sharpe (approx)", f"{summary['sharpe']:.2f}")
        st.metric("Sortino (approx)", f"{summary['sortino']:.2f}")
        st.metric("Max Drawdown", f"{summary['max_drawdown']:.2%}")
        st.metric("Equity finale", f"{summary['final_equity']:.2f}")

with tab_strat:
    st.write("Aperçu des signaux par stratégie (dernière barre) :")
    rows = []
    for s in symbols[:20]:
        d = load_symbol(s, timeframe, lookback)
        sigs = {}
        if use_hybrid: sigs["Hybrid"] = int(hybrid_pro(d, session_start=session_start, session_end=session_end).iloc[-1])
        if use_momo:   sigs["Momentum"] = int(momentum_breakout(d).iloc[-1])
        if use_mr:     sigs["MeanRevert"] = int(mean_revert(d).iloc[-1])
        combined = combine_signals({k: pd.Series([v], index=d.index[-1:]) for k,v in sigs.items()})
        rows.append({"symbol": s, **sigs, "Combined": int(combined.iloc[-1])})
    st.dataframe(pd.DataFrame(rows))

with tab_router:
    st.write("Router: combine signaux, calcule poids par corrélation, propose tailles.")
    returns_df = {}
    prices = {}
    final_signals = {}
    for s in symbols[:20]:
        try:
            d = load_symbol(s, timeframe, 600)
            prices[s] = float(d["close"].iloc[-1])
            d["ret"] = d["close"].pct_change().fillna(0)
            returns_df[s] = d["ret"]
            sigs = []
            if use_hybrid: sigs.append(("Hybrid", hybrid_pro(d, session_start=session_start, session_end=session_end)))
            if use_momo:   sigs.append(("Momentum", momentum_breakout(d)))
            if use_mr:     sigs.append(("MeanRevert", mean_revert(d)))
            final = combine_signals({k:v for k,v in sigs}) if sigs else pd.Series(0, index=d.index)
            final_signals[s] = final
        except Exception:
            continue
    ret_df = pd.DataFrame(returns_df).dropna()
    actions = route_orders({k:v for k,v in final_signals.items()}, prices, ret_df, risk_budget_usd=risk_budget)
    st.write(pd.DataFrame(actions))

with tab_alerts:
    st.write("Évalue les conditions d’alerte.")
    # Simule equity via backtest sur premier symbole
    if symbols:
        d = load_symbol(symbols[0], timeframe, lookback)
        bt_df, _ = run_backtest(d, capital=capital, risk_per_trade=0.01)
        eq = bt_df["equity"]
    else:
        eq = pd.Series(dtype=float)
    msgs = []
    ddm = alert_drawdown(eq, max_dd_threshold=-dd_thresh)
    if ddm: msgs.append(ddm)
    hour_utc = dt.datetime.utcnow().hour
    wins = parse_fee_windows(fee_windows)
    fee_msg = alert_high_fees(hour_utc, wins)
    if fee_msg: msgs.append(fee_msg)
    st.write("Alertes (paper):", msgs if msgs else "Aucune")

with tab_live:
    if mode_live != "Live (Bitget)":
        st.warning("Passe en mode Live (Bitget) dans la sidebar pour exécuter.")
    else:
        st.success("Mode live actif.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔁 Scanner & Router (prévisualisation)"):
            st.session_state["preview"] = True
    with col2:
        if st.button("🚀 Exécuter ordres (router)"):
            st.session_state["execute"] = True

    if st.session_state.get("preview"):
        st.subheader("Prévisualisation des actions")
        # recalcul actions
        returns_df = {}
        prices = {}
        final_signals = {}
        for s in symbols[:20]:
            try:
                d = load_symbol(s, timeframe, 600)
                prices[s] = float(d["close"].iloc[-1])
                d["ret"] = d["close"].pct_change().fillna(0)
                returns_df[s] = d["ret"]
                sigs = []
                if use_hybrid: sigs.append(("Hybrid", hybrid_pro(d, session_start=session_start, session_end=session_end)))
                if use_momo:   sigs.append(("Momentum", momentum_breakout(d)))
                if use_mr:     sigs.append(("MeanRevert", mean_revert(d)))
                final = combine_signals({k:v for k,v in sigs}) if sigs else pd.Series(0, index=d.index)
                final_signals[s] = final
            except Exception:
                continue
        ret_df = pd.DataFrame(returns_df).dropna()
        actions = route_orders(final_signals, prices, ret_df, risk_budget_usd=risk_budget)
        st.dataframe(pd.DataFrame(actions))

    if st.session_state.get("execute") and mode_live == "Live (Bitget)":
        st.subheader("Exécution Live")
        # Execute the same actions
        returns_df = {}
        prices = {}
        final_signals = {}
        for s in symbols[:20]:
            try:
                d = load_symbol(s, timeframe, 600)
                prices[s] = float(d["close"].iloc[-1])
                d["ret"] = d["close"].pct_change().fillna(0)
                returns_df[s] = d["ret"]
                sigs = []
                if use_hybrid: sigs.append(("Hybrid", hybrid_pro(d, session_start=session_start, session_end=session_end)))
                if use_momo:   sigs.append(("Momentum", momentum_breakout(d)))
                if use_mr:     sigs.append(("MeanRevert", mean_revert(d)))
                final = combine_signals({k:v for k,v in sigs}) if sigs else pd.Series(0, index=d.index)
                final_signals[s] = final
            except Exception:
                continue
        ret_df = pd.DataFrame(returns_df).dropna()
        actions = route_orders(final_signals, prices, ret_df, risk_budget_usd=risk_budget)

        results = []
        for a in actions:
            side = "buy" if a["side"]=="buy" else "sell"
            try:
                r = place_order(a["symbol"], side, a["qty"], type_="market", leverage=int(leverage))
                results.append({"symbol": a["symbol"], "side": a["side"], "qty": a["qty"], "status":"OK"})
                if enable_tg_conf:
                    tg_send(f"*Ordre* {a['symbol']} {a['side']} qty={a['qty']:.4f}\nPrix≈{a['price']:.4f}  Levier×{int(leverage)}")
            except Exception as e:
                results.append({"symbol": a["symbol"], "side": a["side"], "qty": a["qty"], "status":f"ERR {e}"})
        st.dataframe(pd.DataFrame(results))

        # Post-trade PnL recap (approx en lisant positions)
        pos = fetch_positions()
        bal = fetch_balance()
        st.write("Positions live (résumé brut):", pos)
        st.write("Balance (USDT):", bal)
        if enable_tg_conf:
            tg_send(f"✅ Exécution terminée — Positions: {len(pos)} | Balance USDT: total={bal.get('total',0)} free={bal.get('free',0)}")

# Footer note
st.caption("N’oublie pas: clés en Secrets/ENV, utilisateur par défaut à supprimer, chiffrement local avant stockage.")
