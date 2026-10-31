FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Install system dependencies: ffmpeg for media processing, libgl for opencv, curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose default port
EXPOSE 8000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/status || exit 1

# Start server using PORT env variable
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
