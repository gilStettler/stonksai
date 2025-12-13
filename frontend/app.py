import os
import requests
import pandas as pd
import streamlit as st

DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="VolaTrade Insights", layout="wide")
st.title("VolaTrade Insights")
st.caption("Volatility Forecasts (EWMA + VIX + Chronos)")

with st.sidebar:
    st.subheader("Login")
    api_key = st.text_input("VolaTrade API Key", type="password")
    st.divider()
    st.subheader("Connection")
    api_url = st.text_input("API URL", value=DEFAULT_API_URL)

def _headers():
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}

def api_get(path: str, params=None):
    r = requests.get(f"{api_url}{path}", headers=_headers(), params=params, timeout=30)
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.text}")
        st.stop()
    return r.json()

def api_post(path: str):
    r = requests.post(f"{api_url}{path}", headers=_headers(), timeout=600)
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.text}")
        st.stop()
    return r.json()

if not api_key:
    st.info("Bitte API Key links eingeben.")
    st.stop()

# Fetch symbols
stocks = api_get("/v1/stocks")
symbols = stocks.get("symbols", [])
last_updated = stocks.get("last_updated")

top_left, top_right = st.columns([2, 1])

with top_left:
    st.subheader("Dashboard")
    st.caption(f"Forecast data last updated: {last_updated}")
    symbol = st.selectbox("Choose a stock symbol", symbols)

with top_right:
    st.subheader("Admin")
    st.caption("Nur Admin Keys können Jobs triggern. Sonst gibt’s 403.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Run Ingest"):
            res = api_post("/v1/admin/jobs/ingest")
            st.success(f"Return code: {res['returncode']}")
            if res.get("stderr"):
                st.warning(res["stderr"])
            st.code(res.get("stdout", ""), language="text")
    with c2:
        if st.button("Run Forecast"):
            res = api_post("/v1/admin/jobs/forecast")
            st.success(f"Return code: {res['returncode']}")
            if res.get("stderr"):
                st.warning(res["stderr"])
            st.code(res.get("stdout", ""), language="text")

st.divider()

# Latest + history
latest = api_get("/v1/forecast/latest", params={"symbol": symbol})
hist = api_get("/v1/forecast/history", params={"symbol": symbol})

rec = latest["record"]
records = hist["records"]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Forecast (median)", f"{rec['forecast_value']:.4f}")
k2.metric("q10 (low)", f"{rec['confidence_low']:.4f}")
k3.metric("q90 (high)", f"{rec['confidence_high']:.4f}")
k4.metric("Last known vol", f"{rec['last_known_vol']:.4f}")

df = pd.DataFrame(records)
df["target_date"] = pd.to_datetime(df["target_date"])
df = df.sort_values("target_date")

cA, cB = st.columns([2, 1])

with cA:
    st.subheader("Forecast history (median)")
    st.line_chart(df.set_index("target_date")[["forecast_value"]])

    st.subheader("Confidence band (q10 / median / q90)")
    band = df.set_index("target_date")[["confidence_low", "forecast_value", "confidence_high"]]
    st.line_chart(band)

with cB:
    st.subheader("Latest record")
    st.json(rec)

with st.expander("All records"):
    st.dataframe(df, use_container_width=True)
