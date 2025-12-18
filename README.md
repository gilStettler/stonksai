# StonksAI – Volatility Prediction for Swiss Market Index Stocks

This repository contains the complete **StonksAI MVP**, including data ingestion, daily volatility forecast generation, a backend API, and a web-based dashboard.  
The steps below allow lecturers to **clone the repository and execute the entire pipeline end-to-end** on a local machine.

---

## Prerequisites

Please ensure the following software is installed:

- **Python 3.10**
- **pip**
- **Docker Desktop** (includes Docker & Docker Compose)

Verify installation:

```bash
python3.10 --version
docker --version
docker compose version
```

---

## 1. Clone the Repository

```bash
git clone <REPOSITORY_URL>
cd stonksai
```

---

## 2. Create and Activate Virtual Environment (Python 3.10)

### macOS / Linux
```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Windows (PowerShell)
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

---

## 3. Install Python Dependencies

Install all required Python packages:

```bash
cd jobs
pip install -r requirements.txt
```

---

## 4. Environment Configuration

Create a `.env` file in the **repository root**:

```env
ALPHAVANTAGE_API_KEY=YOUR_ALPHA_VANTAGE_API_KEY
```

Notes:
- The Alpha Vantage API key is **required**.
- All output data will be written under `jobs/data_csv/`.

---

## 5. Run Data Ingestion Job

This job downloads daily OHLCV data from Alpha Vantage, fetches VIX data, merges both, and stores CSV files.

```bash
python jobs/data.py
```

After execution, the following structure is created:

```text
jobs/data_csv/
├── snapshots/
│   └── YYYY-MM-DD/
│       ├── data_<TICKER>.csv
│       └── ...
└── latest/
    ├── data_<TICKER>.csv
    └── ...
```

Snapshots are stored under **yesterday’s date**, as the data represents end-of-day market values.

---

## 6. Run Daily Forecast Generation

This job reads the ingested data and generates daily volatility forecasts.

```bash
python jobs/generate_daily_forecasts.py
```

---

## 7. Start Backend & Frontend via Docker Compose

After data ingestion and forecast generation, start the application:

```bash
docker compose up --build
```

To stop the application:

```bash
docker compose down
```

---

## 8. Access the Application

Once running, the services are available locally:

- **Streamlit Dashboard**: http://localhost:8501  
- **FastAPI Backend**: http://localhost:8000  
- **API Documentation (Swagger)**: http://localhost:8000/docs  

---

## 9. Recommended Execution Order

```bash
python jobs/data.py
python jobs/generate_daily_forecasts.py
docker compose up --build
```

---

## Project Structure

```text
stonksai/
├── jobs/
│   ├── data.py
│   ├── generate_daily_forecasts.py
│   └── data_csv/
├── backend/
│   └── FastAPI application
├── frontend/
│   └── Streamlit application
├── docker-compose.yml
└── requirements.txt
```

---

## Notes for Evaluation

- The system uses **CSV files as a single source of truth** for transparency and reproducibility.
- No external database is required.
- Individual ticker failures do not interrupt the entire pipeline.
- The project is designed as a **lightweight academic MVP** for volatility prediction on Swiss Market Index stocks.
