# Medical Prescription Validation Environment - Docker Container
# ==============================================================
# Trains AI to catch medication errors before they harm patients.

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure Python can find all modules
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV OPENENV_ENABLE_WEB_INTERFACE=true
ENV PORT=7860
ENV HOST=0.0.0.0
ENV WORKERS=4

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run uvicorn server
CMD ["sh", "-c", "uvicorn server.app:app --host ${HOST} --port ${PORT} --workers ${WORKERS}"]