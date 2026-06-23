# ─────────────────────────────────────────────
# InsightForge — Docker image
# Flask + scikit-learn + Gunicorn, production-ready
# ─────────────────────────────────────────────
FROM python:3.11-slim

# System deps needed by matplotlib / pandas / scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Folders the app writes to at runtime
RUN mkdir -p static/uploads static/plots data

# Non-root user (security best practice)
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 5050

ENV PORT=5050
ENV DATABASE_PATH=/app/data/insightforge.db

# Gunicorn as the production WSGI server
CMD gunicorn app:app --bind 0.0.0.0:${PORT} --workers 2 --timeout 120
