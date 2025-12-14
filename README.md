# StonksAI - Daily Volatility Forecasting

This project implements an automated pipeline to forecast the Daily Volatility (EWMA) of SMI (Swiss Market Index) stocks using the **Amazon Chronos-2** foundation model.

## Frontend

- open Terminal
- cd frontend
- streamlit run app.py

## Backend
- open another Terminal
- cd backend
- python -m uvicorn api.main:app --reload --port 8000

## 🚀 Key Features

*   **Next-Day Forecast:** Predicts volatility for the next trading day.
*   **Context Window:** Provides 3-day lookback validation (Forecast vs. Actual) for UI visualization.
*   **VIX Integration:** Uses `yfinance` to fetch real-time VIX data (Systematic Risk) for improved accuracy during market stress.
*   **Automated Pipeline:** Two standalone scripts handle everything from data fetching to inference.

## 📂 Project Structure

*   `data_final.py`: **Data Fetcher**. Downloads daily OHLCV from Alpha Vantage and VIX from Yahoo Finance. Saves cleaned CSVs to `data/`.
*   `generate_daily_forecasts.py`: **Inference Engine**. Loads the Chronos-2 model, reads data from `data/`, generates forecasts, and updates `forecast_history.json`.
*   `forecast_history.json`: **Database**. A JSON file containing the full forecast history for each stock, ready to be consumed by a UI/Dashboard.
*   `data/`: Directory containing the latest daily stock data (CSV).
*   `archive/`: Contains legacy research notebooks, experiments, and old data.

## 🛠️ Installation

1.  python 3.10+ recommended.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `chronos` is installed directly from GitHub in `requirements.txt`.*

3.  Set API Keys in `data_final.py` (if not already set).

## ⚡ Usage (Daily Workflow)

Run the scripts in order. Ideal execution time: **After 22:30 CET** (when US markets close).

1.  **Update Data:**
    ```bash
    python data_final.py
    ```
2.  **Generate Forecasts:**
    ```bash
    python generate_daily_forecasts.py
    ```

## 📊 Output Format (`forecast_history.json`)

The output is optimized for UI consumption. Example structure:

```json
{
  "data": {
    "Nestle": [
        {
            "target_date": "2025-12-11",
            "forecast_value": 0.0125,
            "actual_value": 0.0130,  // For validation of past days
            "type": "historical_reforecast"
        },
        {
            "target_date": "2025-12-12",
            "forecast_value": 0.0128,
            "actual_value": null,     // Future forecast
            "type": "forecast_next_day"
        }
    ]
  },
  "last_updated": "..."
}
```

## 📜 Ticker Reference (SMI)

| Stock | Ticker (AV/Yahoo) |
|-------|-------------------|
| Nestlé | NSRGY |
| Novartis | 0QLR.LON |
| Roche | RHO6.FRK |
| UBS | UBS |
| ABB | ABBNY |
| Swisscom | 0QKI.LON |
| ... | (See code for full list) |
