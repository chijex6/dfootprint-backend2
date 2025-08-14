# --------------------------
# 1. Base Image
# --------------------------
FROM python:3.12-slim AS base

# Ensure Python output is sent straight to terminal (no buffering)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# --------------------------
# 2. Working Directory
# --------------------------
WORKDIR /app

# --------------------------
# 3. Install Dependencies
# --------------------------
# Upgrade pip and install system dependencies first
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --------------------------
# 4. Copy Application
# --------------------------
COPY . .

# --------------------------
# 5. Expose Port
# --------------------------
EXPOSE 8000

# --------------------------
# 6. Start Command (Gunicorn + Uvicorn Workers)
# --------------------------
# Adjust workers based on CPU cores: (2 x $cores) + 1
# Example: 4 cores => workers=9
CMD ["gunicorn", "server:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "4", \
     "--timeout", "60"]
