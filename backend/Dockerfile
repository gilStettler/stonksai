FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

# requirements
COPY backend/requirements.txt /app/backend/requirements.txt
COPY jobs/requirements.txt /app/jobs/requirements.txt

RUN pip install --no-cache-dir -r /app/backend/requirements.txt \
 && pip install --no-cache-dir -r /app/jobs/requirements.txt

# Code kopieren
COPY backend /app/backend
COPY jobs /app/jobs

# Damit "app.main:app" importierbar ist:
# /app/backend wird als PYTHONPATH gesetzt -> "app" ist dann /app/backend/app
ENV PYTHONPATH=/app/backend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
