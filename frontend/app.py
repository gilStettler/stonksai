import os
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import timedelta


# -----------------------------
# Page setup + compact typography
# -----------------------------
st.set_page_config(page_title="VolaTrade Insights", layout="wide")

st.markdown(
    """
    <style>
      h1 { font-size: 1.6rem; margin-bottom: 0.2rem; }
      h2 { font-size: 1.25rem; margin-top: 0.6rem; }
      h3 { font-size: 1.05rem; margin-top: 0.6rem; }
      .small { font-size: 0.9rem; opacity: 0.85; }
      div[data-testid="stMetricValue"] { font-size: 1.2rem; }
      div[data-testid="stMetricLabel"] { font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

st.title("VolaTrade Insights")
st.caption("Next-day volatility forecast · realized range · confidence band · risk label")


# -----------------------------
# Config (secrets.toml OR env)
# -----------------------------
API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://127.0.0.1:8000"))
API_KEY = st.secrets.get("API_KEY", os.getenv("API_KEY", ""))

if not API_KEY:
    st.warning(
        "API_KEY missing.\n\n"
        "Create frontend/.streamlit/secrets.toml:\n\n"
        "API_URL='http://127.0.0.1:8000'\n"
        "API_KEY='vt_live_free_abc'"
    )
    st.stop()


# -----------------------------
# API helpers (SUSTAINABLE CACHE)
# -----------------------------
def _headers(api_key: str):
    return {"Authorization": f"Bearer {api_key}"}


@st.cache_data(ttl=30)
def api_get_cached(api_url: str, api_key: str, path: str, params_items: tuple) -> dict:
    params = dict(params_items)
    r = requests.get(
        f"{api_url}{path}",
        params=params,
        headers=_headers(api_key),
        timeout=30,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"API error {r.status_code}: {r.text}")
    return r.json()


def api_get(path: str, params: dict | None = None) -> dict:
    if params is None:
        params = {}
    params_items = tuple(sorted(params.items()))
    return api_get_cached(API_URL, API_KEY, path, params_items)


def api_post(path: str) -> dict:
    r = requests.post(
        f"{API_URL}{path}",
        headers=_headers(API_KEY),
        timeout=180,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"API error {r.status_code}: {r.text}")
    return r.json()


# -----------------------------
# Helpers
# -----------------------------
def safe_progress_value(pos):
    """st.progress accepts only [0..1]. Clamp + neutral fallback."""
    try:
        if pos is None:
            return 0.5
        v = float(pos)
        if v != v:  # NaN
            return 0.5
        return max(0.0, min(1.0, v))
    except Exception:
        return 0.5


def _as_iso_list(dt_series):
    # Plotly is much more stable with ISO strings than pandas Timestamp objects
    return [pd.to_datetime(x).strftime("%Y-%m-%d") for x in dt_series]


def risk_badge(label: str) -> str:
    if label == "High":
        return "🔴 High"
    if label == "Low":
        return "🟢 Low"
    return "🟡 Medium"


def delta_badge(delta, arrow):
    if delta is None or arrow is None:
        return "n/a"
    try:
        return f"{arrow} {float(delta):.4f}"
    except Exception:
        return "n/a"


def sanitize_history_df(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["target_date"] = pd.to_datetime(df.get("target_date"), errors="coerce")
    for c in ["forecast_value", "confidence_low", "confidence_high", "actual_value"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["target_date", "forecast_value"]).sort_values("target_date")
    return df


def points_to_realized_df(points):
    """Convert backend range_5d.points into a DF with date + realized."""
    if not points:
        return pd.DataFrame(columns=["date", "realized"])
    df = pd.DataFrame(points).copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "realized"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["realized"] = pd.to_numeric(df["realized"], errors="coerce")
    df = df.dropna(subset=["date", "realized"]).sort_values("date")
    return df


# -----------------------------
# Plot builders
# -----------------------------
def build_5d_window_figure(points, symbol: str, forecast_value: float | None):
    """
    points: list of {date, realized} from backend (CSV-based).
    Shows:
      - green band between min/max
      - realized line
      - forecast marker at T+1
    """
    df = points_to_realized_df(points)
    if df.empty or len(df) < 2:
        return None

    x_dates = _as_iso_list(df["date"])
    x_rev = list(reversed(x_dates))
    y_real = df["realized"].astype(float).tolist()

    rmin, rmax = float(df["realized"].min()), float(df["realized"].max())

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_dates + x_rev,
        y=[rmax]*len(x_dates) + [rmin]*len(x_dates),
        fill="toself",
        fillcolor="rgba(40, 167, 69, 0.15)",
        line=dict(width=0),
        name="5D Range",
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=x_dates,
        y=y_real,
        mode="lines+markers",
        name="Realized (EWMA)",
        hovertemplate="Realized: %{y:.4f}<br>%{x}<extra></extra>",
    ))

    if forecast_value is not None:
        t1_iso = (pd.to_datetime(df["date"].iloc[-1]) + timedelta(days=1)).strftime("%Y-%m-%d")
        fig.add_trace(go.Scatter(
            x=[t1_iso],
            y=[float(forecast_value)],
            mode="markers",
            marker=dict(size=12, symbol="diamond"),
            name="Forecast (T+1)",
            hovertemplate="Forecast: %{y:.4f}<br>%{x}<extra></extra>",
        ))

    fig.update_layout(
        height=240,
        margin=dict(l=40, r=20, t=35, b=25),
        title=dict(text=f"{symbol} – 5-Day Realized Range + Forecast", font=dict(size=13), x=0.01),
        font=dict(size=11),
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
    )
    fig.update_yaxes(title_text="Volatility", tickfont=dict(size=10))
    fig.update_xaxes(title_text="Date", tickfont=dict(size=10))
    return fig


def build_history_with_realized_figure(df_forecast: pd.DataFrame, realized_points, symbol: str):
    """
    Forecast history (JSON) + last 5 realized volatility points (CSV)
    and show forecast vs realized in the SAME chart.

    - Forecast: line + markers
    - Confidence band: fill between low/high (if available)
    - Realized: last 5 realized points (EWMA) as line + markers
    """
    fig = go.Figure()

    # --- Realized series (last 5 points)
    df_r = points_to_realized_df(realized_points)
    if not df_r.empty:
        x_r = _as_iso_list(df_r["date"])
        y_r = df_r["realized"].astype(float).tolist()
        fig.add_trace(go.Scatter(
            x=x_r,
            y=y_r,
            mode="lines+markers",
            name="Realized (EWMA, last 5D)",
            hovertemplate="Realized: %{y:.4f}<br>%{x}<extra></extra>",
        ))

    # --- Forecast series (history)
    x_f = _as_iso_list(df_forecast["target_date"])
    y_f = df_forecast["forecast_value"].astype(float).tolist()

    # Confidence band
    if {"confidence_low", "confidence_high"}.issubset(df_forecast.columns):
        ok = df_forecast["confidence_low"].notna() & df_forecast["confidence_high"].notna()
        ok_list = ok.tolist()

        x_ok = [x_f[i] for i, v in enumerate(ok_list) if v]
        y_low = df_forecast.loc[ok, "confidence_low"].astype(float).tolist()
        y_high = df_forecast.loc[ok, "confidence_high"].astype(float).tolist()

        if len(x_ok) == len(y_low) == len(y_high) and len(x_ok) > 1:
            fig.add_trace(go.Scatter(
                x=x_ok,
                y=y_high,
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=x_ok,
                y=y_low,
                fill="tonexty",
                fillcolor="rgba(99,110,250,0.15)",
                line=dict(width=0),
                name="Confidence Band",
                hoverinfo="skip",
            ))

    fig.add_trace(go.Scatter(
        x=x_f,
        y=y_f,
        mode="lines+markers",
        name="Forecast (history)",
        hovertemplate="Forecast: %{y:.4f}<br>%{x}<extra></extra>",
    ))

    # Optional: actual_value (if present in JSON)
    if "actual_value" in df_forecast.columns and df_forecast["actual_value"].notna().any():
        ok2 = df_forecast["actual_value"].notna()
        x_a = [x_f[i] for i, v in enumerate(ok2.tolist()) if v]
        y_a = df_forecast.loc[ok2, "actual_value"].astype(float).tolist()
        fig.add_trace(go.Scatter(
            x=x_a,
            y=y_a,
            mode="markers",
            marker=dict(size=7, symbol="circle"),
            name="Actual (from JSON, if available)",
            hovertemplate="Actual: %{y:.4f}<br>%{x}<extra></extra>",
        ))

    fig.update_layout(
        height=360,
        margin=dict(l=40, r=20, t=40, b=30),
        title=dict(text=f"{symbol} – Forecast History vs Realized (last 5 days)", font=dict(size=14), x=0.01),
        font=dict(size=11),
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
    )
    fig.update_yaxes(title_text="Volatility", tickfont=dict(size=10))
    fig.update_xaxes(title_text="Date", tickfont=dict(size=10))
    return fig


# -----------------------------
# Nonce state (forces re-fetch & re-render)
# -----------------------------
if "nonce" not in st.session_state:
    st.session_state["nonce"] = 0


# -----------------------------
# Sidebar (admin)
# -----------------------------
with st.sidebar:
    st.header("Admin")

    if st.button("Run Ingest"):
        try:
            res = api_post("/v1/admin/jobs/ingest")
            st.success(f"Return code: {res.get('returncode')}")
            st.code((res.get("stdout") or "")[-4000:] + "\n" + (res.get("stderr") or "")[-2000:])
            st.cache_data.clear()
            st.session_state["nonce"] += 1
        except Exception as e:
            st.error(str(e))

    if st.button("Run Forecast"):
        try:
            res = api_post("/v1/admin/jobs/forecast")
            st.success(f"Return code: {res.get('returncode')}")
            st.code((res.get("stdout") or "")[-4000:] + "\n" + (res.get("stderr") or "")[-2000:])
            st.cache_data.clear()
            st.session_state["nonce"] += 1
        except Exception as e:
            st.error(str(e))


# -----------------------------
# Load symbols
# -----------------------------
stocks = api_get("/v1/stocks", {"_nonce": st.session_state["nonce"]})
symbols = stocks.get("symbols", [])
last_updated = stocks.get("last_updated")

if not symbols:
    st.info("No symbols available yet. Run ingest + forecast jobs first.")
    st.stop()

tab1, tab2 = st.tabs(["📈 Dashboard", "🏁 Overview"])


# =============================
# Dashboard
# =============================
with tab1:
    symbol = st.selectbox("Stock", symbols, key="symbol_select")

    prev = st.session_state.get("_prev_symbol")
    if prev and prev != symbol:
        st.cache_data.clear()
        st.session_state["nonce"] += 1
    st.session_state["_prev_symbol"] = symbol

    st.caption(f"Last updated: {last_updated}")

    # --- Latest (nonce included so cache never “sticks” across symbol switches)
    latest = api_get("/v1/forecast/latest", {"symbol": symbol, "_nonce": st.session_state["nonce"]})
    record = latest.get("record", {}) or {}
    derived = latest.get("derived", {}) or {}

    # Metrics
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Δ Volatility", delta_badge(derived.get("delta_vol"), derived.get("delta_arrow")))

    band = derived.get("confidence_band", {}) or {}
    try:
        c2.metric("Confidence Band", f"{float(band.get('low')):.4f} – {float(band.get('high')):.4f}")
    except Exception:
        c2.metric("Confidence Band", "n/a")

    r5 = derived.get("range_5d", {}) or {}
    n = r5.get("n", 0)
    rng_label = f"Range (n={n})" if n else "Range"
    try:
        c3.metric(rng_label, f"{float(r5.get('min')):.4f} – {float(r5.get('max')):.4f}")
    except Exception:
        c3.metric(rng_label, "n/a")

    risk = (derived.get("risk", {}) or {}).get("label", "Medium")
    c4.metric("Risk", risk_badge(risk))

    pos = r5.get("pos", None)
    st.caption("Forecast position within realized range (0 = low end, 1 = high end)")
    st.progress(safe_progress_value(pos))

    # 5D chart
    st.subheader("5-Day Realized Range (from CSV) + Forecast (T+1)")
    points = r5.get("points", []) or []
    try:
        fval = float(record.get("forecast_value"))
    except Exception:
        fval = None

    plot5_slot = st.empty()
    fig5 = build_5d_window_figure(points, symbol, fval)

    target_date = (record or {}).get("target_date", "na")
    plot5_key = f"plot5::{symbol}::{target_date}::{hash(str(points))}::{st.session_state['nonce']}"

    if fig5:
        plot5_slot.plotly_chart(fig5, use_container_width=True, config=PLOTLY_CONFIG, key=plot5_key)
    else:
        plot5_slot.info("Not enough realized CSV data to render 5-day chart yet.")

    # Forecast history + realized
    st.subheader("Forecast History + Realized (last 5 days)")
    hist_payload = api_get("/v1/forecast/history", {"symbol": symbol, "_nonce": st.session_state["nonce"]})
    hist_rows = hist_payload.get("records", [])
    df_hist = sanitize_history_df(hist_rows)

    hist_slot = st.empty()
    if not df_hist.empty:
        # Keep history readable: last 30 points (adjust as you like)
        df_hist_tail = df_hist.tail(30).copy()

        fig_hist = build_history_with_realized_figure(df_hist_tail, points, symbol)

        # make key depend on symbol + content + nonce (forces remount)
        hist_key = f"hist::{symbol}::{hash(df_hist_tail.to_csv(index=False))}::{hash(str(points))}::{st.session_state['nonce']}"
        hist_slot.plotly_chart(fig_hist, use_container_width=True, config=PLOTLY_CONFIG, key=hist_key)
    else:
        hist_slot.info("No forecast history available yet.")


# =============================
# Overview
# =============================
with tab2:
    st.caption(f"Last updated: {last_updated}")

    rows = []
    for s in symbols:
        try:
            p = api_get("/v1/forecast/latest", {"symbol": s, "_nonce": st.session_state["nonce"]})
            d = p.get("derived", {}) or {}
            r = p.get("record", {}) or {}

            rows.append({
                "symbol": s,
                "risk": risk_badge((d.get("risk", {}) or {}).get("label", "Medium")),
                "delta": delta_badge(d.get("delta_vol"), d.get("delta_arrow")),
                "forecast": r.get("forecast_value", None),
            })
        except Exception:
            pass

    st.dataframe(pd.DataFrame(rows), use_container_width=True)
