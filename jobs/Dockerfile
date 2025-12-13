FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY generate_daily_forecasts.py .

# Writes to /data via FORECAST_STORE_DIR
ENV FORECAST_STORE_DIR=/data

CMD ["python", "generate_daily_forecasts.py"]