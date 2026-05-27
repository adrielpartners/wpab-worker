FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (ffmpeg for audio chunking)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create work directory for temp audio files
RUN mkdir -p /work/jobs

EXPOSE 8080

# Default command: API server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]