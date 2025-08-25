# --------------------------
# 1. Base Image
# --------------------------
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# --------------------------
# 2. Working Directory
# --------------------------
WORKDIR /app

# --------------------------
# 3. Install Dependencies
# --------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --------------------------
# 4. Copy Application
# --------------------------
COPY . .

# --------------------------
# 5. Expose Render Port
# --------------------------
# Render sets PORT env var automatically (default: 10000)
EXPOSE $PORT

# --------------------------
# 6. Start Command (Gunicorn + Uvicorn Workers)
# --------------------------
CMD ["sh", "-c", \
     "gunicorn server:app \
      -k uvicorn.workers.UvicornWorker \
      --bind 0.0.0.0:${PORT:-10000} \
      --workers 4 \
      --timeout 60"]