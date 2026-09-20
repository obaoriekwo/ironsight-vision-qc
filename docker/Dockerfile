# ==========================================================================
# Dockerfile — standard x86_64 image for the QC dashboard + API.
# Use this for a factory-network server, a demo box, or CI. For the
# actual edge unit bolted to the inspection camera, use Dockerfile.jetson.
# ==========================================================================
FROM python:3.11-slim

# OpenCV needs a couple of system libs even in "headless" mode
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/

EXPOSE 8000

WORKDIR /app/app/backend
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
