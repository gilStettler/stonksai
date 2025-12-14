FROM python:3.11-slim

WORKDIR /app

# System deps (falls pandas/numpy wheels ok; slim reicht meist)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

# Install deps: backend + jobs (damit Admin-Job-Trigger funktioniert)
COPY backend/requirements.txt /app/backend/requirements.txt
COPY jobs/requirements.txt /app/jobs/requirements.txt

RUN pip install --no-cache-dir -r /app/backend/requirements.txt \
 && pip install --no-cache-dir -r /app/jobs/requirements.txt

# Copy code
COPY backend /app/backend
COPY jobs /app/jobs

# Optional: if backend imports backend.api etc, ensure package root is /app
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
