#!/usr/bin/env bash
set -euo pipefail

source scripts/env.sh

# start API
(uvicorn api.main:app --reload --port 8000) &
API_PID=$!

# start Streamlit
(API_URL="http://localhost:8000" streamlit run frontend/app.py) &
FE_PID=$!

echo "[dev] API PID=$API_PID | Frontend PID=$FE_PID"
echo "[dev] Ctrl+C to stop. If needed: kill $API_PID $FE_PID"
wait
