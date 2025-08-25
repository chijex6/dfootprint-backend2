# --------------------------
# 1. Base Image
# --------------------------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# --------------------------
# 2. Working Directory
# --------------------------
WORKDIR /app

# --------------------------
# 3. Create Virtual Environment
# --------------------------
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# --------------------------
# 4. Install System Dependencies
# --------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# --------------------------
# 5. Install Python Dependencies
# --------------------------
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --------------------------
# 6. Copy Application
# --------------------------
COPY . .

# --------------------------
# 7. Expose Render Port
# --------------------------
EXPOSE ${PORT:-10000}

# --------------------------
# 8. Start Command
# --------------------------
CMD gunicorn server:app \
    -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:${PORT:-10000} \
    --workers 4 \
    --timeout 60