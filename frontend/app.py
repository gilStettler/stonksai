from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ------------------------------------------------------------------------------
# Page
# ------------------------------------------------------------------------------
st.set_page_config(page_title="StonksAI – Volatility", layout="wide")
PLOTLY_CONFIG = {"displayModeBar": False}

# ------------------------------------------------------------------------------
# Config (secrets first, then env, then default)
# ------------------------------------------------------------------------------


def _secret_get(key: str, default: Any = None) -> Any:
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


# Prefer docker-safe env var first:
# - Docker Compose: API_BASE_URL=http://backend:8000
# - Local (no docker): API_BASE_URL=http://127.0.0.1:8000
API_BASE = (
    _secret_get("API_BASE_URL")
    or _secret_get("API_BASE")
    or _secret_get("API_URL")
    or os.getenv("API_BASE_URL")
    or os.getenv("API_BASE")
    or os.getenv("API_URL")
    or "http://127.0.0.1:8000"
).rstrip("/")

# Token naming:
# - VT_API_KEY in .env / docker compose
# - alternatively API_KEY
API_KEY = (
    _secret_get("VT_API_KEY")
    or _secret_get("API_KEY")
    or os.getenv("VT_API_KEY")
    or os.getenv("API_KEY")
    or ""
).strip()

# ------------------------------------------------------------------------------
# HTTP helpers
# ------------------------------------------------------------------------------


def _headers() -> Dict[str, str]:
    if not API_KEY:
        return {}
    return {"Authorization": f"Bearer {API_KEY}"}


def _show_auth_error(status_code: int, body: str):
    st.error(
        f"🔒 Backend Auth fehlgeschlagen ({status_code}).\n\n"
        f"**API_BASE:** `{API_BASE}`  \n"
        f"**API_KEY geladen:** `{bool(API_KEY)}`\n\n"
        "Fix-Checklist:\n"
        "- Docker: Frontend ENV `VT_API_KEY` ist gesetzt (z.B. `vt_live_admin_123`)\n"
        "- Backend ENV `VT_API_KEYS` enthält genau diesen Key\n"
        "- Frontend ruft in Docker `http://backend:8000` auf (nicht `localhost`)\n"
    )
    if body:
        st.code(body[:2000])
    st.stop()


def _show_backend_down(err: Exception):
    st.error(
        "🚫 Backend nicht erreichbar.\n\n"
        f"**API_BASE:** `{API_BASE}`\n\n"
        "Typische Ursachen:\n"
        "- Backend läuft nicht\n"
        "- falscher Host/Port\n"
        "- in Docker muss API_BASE z.B. `http://backend:8000` sein (nicht `localhost`)\n\n"
        f"Fehler: `{type(err).__name__}: {err}`"
    )
    st.stop()


@st.cache_data(ttl=30, show_spinner=False)
def api_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{API_BASE}{path}"
    try:
        r = requests.get(url, params=params or {}, headers=_headers(), timeout=30)
    except requests.exceptions.RequestException as e:
        _show_backend_down(e)

    if r.status_code in (401, 403):
        _show_auth_error(r.status_code, r.text)

    r.raise_for_status()
    return r.json()


def api_post(path: str) -> Dict[str, Any]:
    url = f"{API_BASE}{path}"
    try:
        r = requests.post(url, headers=_headers(), timeout=300)
    except requests.exceptions.RequestException as e:
        _show_backend_down(e)

    if r.status_code in (401, 403):
        _show_auth_error(r.status_code, r.text)

    r.raise_for_status()
    return r.json()


# ------------------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------------------


def fmt_float(x: Any, nd: int = 4) -> str:
    if x is None:
        return "—"
    try:
        xf = float(x)
        if pd.isna(xf):
            return "—"
        return f"{xf:.{nd}f}"
    except Exception:
        return "—"


def fmt_pct(x: Any, nd: int = 1, arrow: Optional[str] = None) -> str:
    if x is None:
        return "—"
    try:
        xf = float(x)
        if pd.isna(xf):
            return "—"
        a = arrow or ""
        return f"{a} {xf:+.{nd}f}%"
    except Exception:
        return "—"


def risk_badge(label: str) -> str:
    if label == "High":
        return "🔴 High"
    if label == "Low":
        return "🟢 Low"
    return "🟠 Medium"


def delta_percent(delta_abs: Any, base: Any) -> Optional[float]:
    if delta_abs is None or base is None:
        return None
    try:
        d = float(delta_abs)
        b = float(base)
        if b == 0 or pd.isna(d) or pd.isna(b):
            return None
        return (d / b) * 100.0
    except Exception:
        return None


# ------------------------------------------------------------------------------
# DataFrame helpers
# ------------------------------------------------------------------------------


def sanitize_records_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).copy()
    if "target_date" in df.columns:
        df["target_date"] = pd.to_datetime(df["target_date"], errors="coerce").dt.date.astype(str)
    for c in ["forecast_value", "confidence_low", "confidence_high", "actual_value", "last_known_vol"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def filter_trading_days(df: pd.DataFrame, date_col: str = "target_date") -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return df
    dt = pd.to_datetime(df[date_col], errors="coerce")
    wd = dt.dt.weekday
    return df.loc[wd < 5].copy()


# ------------------------------------------------------------------------------
# Plot builder
# ------------------------------------------------------------------------------


def build_prediction_vs_actual_figure(
    df_bt: pd.DataFrame,
    symbol: str,
    live_pred: Optional[Dict[str, Any]] = None,
) -> go.Figure:
    fig = go.Figure()

    df_bt = filter_trading_days(df_bt)

    # live pred weekday check
    if live_pred:
        try:
            live_dt = pd.to_datetime(live_pred.get("target_date"), errors="coerce")
            if live_dt is pd.NaT or live_dt.weekday() >= 5:
                live_pred = None
        except Exception:
            live_pred = None

    if df_bt.empty and not live_pred:
        fig.update_layout(
            height=360,
            margin=dict(l=40, r=20, t=40, b=30),
            title=dict(text=f"{symbol} – Prediction vs Actual (no data)", font=dict(size=14), x=0.01),
        )
        return fig

    if not df_bt.empty:
        df_bt = df_bt.sort_values("target_date").reset_index(drop=True)
        x = df_bt["target_date"].tolist()

        # Confidence band if available
        if "confidence_low" in df_bt.columns and "confidence_high" in df_bt.columns:
            ok_band = df_bt["confidence_low"].notna() & df_bt["confidence_high"].notna()
            if ok_band.any():
                x_ok = df_bt.loc[ok_band, "target_date"].tolist()
                y_low = df_bt.loc[ok_band, "confidence_low"].tolist()
                y_high = df_bt.loc[ok_band, "confidence_high"].tolist()
                if len(x_ok) >= 2:
                    fig.add_trace(go.Scatter(x=x_ok, y=y_high, line=dict(width=0), showlegend=False, hoverinfo="skip"))
                    fig.add_trace(
                        go.Scatter(
                            x=x_ok,
                            y=y_low,
                            fill="tonexty",
                            line=dict(width=0),
                            name="Confidence Band",
                            hoverinfo="skip",
                        )
                    )

        # Prediction line
        if "forecast_value" in df_bt.columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df_bt["forecast_value"].tolist(),
                    mode="lines+markers",
                    name="Prediction (T+1)",
                    hovertemplate="Prediction: %{y:.4f}<br>%{x}<extra></extra>",
                )
            )

        # Actual line if available
        if "actual_value" in df_bt.columns and df_bt["actual_value"].notna().any():
            ok = df_bt["actual_value"].notna()
            x_a = df_bt.loc[ok, "target_date"].tolist()
            y_a = df_bt.loc[ok, "actual_value"].tolist()
            mode = "lines+markers" if len(x_a) >= 2 else "markers"
            fig.add_trace(
                go.Scatter(
                    x=x_a,
                    y=y_a,
                    mode=mode,
                    marker=dict(size=8),
                    name="Actual Volatility",
                    hovertemplate="Actual: %{y:.4f}<br>%{x}<extra></extra>",
                )
            )

    # Live pred marker
    if live_pred:
        live_date = str(pd.to_datetime(live_pred.get("target_date")).date())
        live_val = live_pred.get("forecast_value", None)

        if df_bt.empty or ("target_date" not in df_bt.columns) or (live_date not in set(df_bt["target_date"].tolist())):
            fig.add_trace(
                go.Scatter(
                    x=[live_date],
                    y=[live_val],
                    mode="markers",
                    marker=dict(size=12, symbol="diamond"),
                    name="Live Prediction",
                    hovertemplate="Live Prediction: %{y:.4f}<br>%{x}<extra></extra>",
                )
            )

    fig.update_layout(
        height=360,
        margin=dict(l=40, r=20, t=40, b=30),
        title=dict(
            text=f"{symbol} – Prediction (T+1) vs Actual (last 5 trading days) + Live",
            font=dict(size=14),
            x=0.01,
        ),
        legend=dict(orientation="h", y=1.10, x=1, xanchor="right"),
    )
    fig.update_yaxes(title_text="Volatility")
    fig.update_xaxes(title_text="Date", rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


# ------------------------------------------------------------------------------
# Session init
# ------------------------------------------------------------------------------
if "nonce" not in st.session_state:
    st.session_state["nonce"] = 0

# ------------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("Config")
    st.caption(f"API_BASE: {API_BASE}")
    st.caption(f"API_KEY loaded: {'yes' if bool(API_KEY) else 'no'}")

    st.divider()
    st.header("Actions")

    colA, colB = st.columns(2)
    with colA:
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.session_state["nonce"] += 1
            st.rerun()

    with colB:
        auto = st.toggle("Auto refresh", value=False, help="Refresh every 60s")

    if auto:
        st.caption("Auto refresh active (60s)")
        st.autorefresh(interval=60_000, key="auto_refresh")

    st.divider()
    st.header("Admin (optional)")
    st.caption("Nur wenn dein API Key im Backend als `:admin` gesetzt ist.")

    if st.button("Run Ingest"):
        res = api_post("/v1/admin/jobs/ingest")
        st.success(f"Return code: {res.get('returncode')}")
        st.code((res.get("stdout") or "")[-4000:] + "\n" + (res.get("stderr") or "")[-2000:])
        st.cache_data.clear()
        st.session_state["nonce"] += 1
        st.rerun()

    if st.button("Run Forecast"):
        res = api_post("/v1/admin/jobs/forecast")
        st.success(f"Return code: {res.get('returncode')}")
        st.code((res.get("stdout") or "")[-4000:] + "\n" + (res.get("stderr") or "")[-2000:])
        st.cache_data.clear()
        st.session_state["nonce"] += 1
        st.rerun()


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
st.title("StonksAI – Volatility Forecast")

# Fetch list of stocks
stocks = api_get("/v1/stocks", {"_nonce": st.session_state["nonce"]})
symbols: List[str] = stocks.get("symbols", []) or []
last_updated = stocks.get("last_updated")

st.caption(f"Last updated: {last_updated}")

if not symbols:
    st.info("No symbols available yet. Run ingest + forecast jobs first.")
    st.stop()

tab1, tab2 = st.tabs(["📈 Dashboard", "🏁 Overview"])

# ------------------------------------------------------------------------------
# Dashboard tab
# ------------------------------------------------------------------------------
with tab1:
    symbol = st.selectbox("Stock", symbols, key="symbol_select")

    # Force refresh when symbol changes
    prev_symbol = st.session_state.get("_prev_symbol")
    if prev_symbol and prev_symbol != symbol:
        st.cache_data.clear()
        st.session_state["nonce"] += 1
    st.session_state["_prev_symbol"] = symbol

    payload = api_get("/v1/forecast/latest", {"symbol": symbol, "_nonce": st.session_state["nonce"]})
    record = payload.get("record", {}) or {}
    derived = payload.get("derived", {}) or {}

    # Core metrics
    pred_val = record.get("forecast_value")
    last_vol = record.get("last_known_vol")
    last_date = record.get("last_known_date")
    tgt_date = record.get("target_date")

    risk_label = (derived.get("risk", {}) or {}).get("label", "Medium")
    delta_abs = derived.get("delta_vol")
    delta_arrow = derived.get("delta_arrow")

    band = (derived.get("confidence_band") or {})
    band_low = band.get("low")
    band_high = band.get("high")

    dpct = delta_percent(delta_abs, last_vol)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Target date", tgt_date or "—")
    c2.metric("Prediction (T+1)", fmt_float(pred_val, 6))
    c3.metric("Risk", risk_badge(risk_label))
    c4.metric("Δ vs last realized", fmt_pct(dpct, 1, delta_arrow))

    st.write(
        f"**Confidence band:** {fmt_float(band_low, 6)} – {fmt_float(band_high, 6)}  \n"
        f"**Last known date:** {last_date or '—'} | **Last known EWMA vol:** {fmt_float(last_vol, 6)}"
    )

    # Backtests (last 5 trading days)
    st.subheader("Prediction vs Actual (last 5 trading days) + Live Prediction")
    bt_payload = api_get(
        "/v1/forecast/backtests",
        {"symbol": symbol, "days": 5, "_nonce": st.session_state["nonce"]},
    )
    df_bt = sanitize_records_df(bt_payload.get("records", []))

    fig = build_prediction_vs_actual_figure(df_bt, symbol, live_pred=record if record else None)

    # Unique key forces Streamlit to re-render correctly
    plot_key = f"pred_actual::{symbol}::{st.session_state['nonce']}::{len(df_bt)}"
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key=plot_key)

# ------------------------------------------------------------------------------
# Overview tab
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("All symbols overview")

    rows: List[Dict[str, Any]] = []
    for s in symbols:
        try:
            p = api_get("/v1/forecast/latest", {"symbol": s, "_nonce": st.session_state["nonce"]})
            d = p.get("derived", {}) or {}
            r = p.get("record", {}) or {}

            dp = delta_percent(d.get("delta_vol"), r.get("last_known_vol"))
            rows.append(
                {
                    "symbol": s,
                    "risk": risk_badge((d.get("risk", {}) or {}).get("label", "Medium")),
                    "Δ%": fmt_pct(dp, 1, d.get("delta_arrow")),
                    "prediction": r.get("forecast_value"),
                    "target_date": r.get("target_date"),
                    "last_vol": r.get("last_known_vol"),
                }
            )
        except Exception:
            continue

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
