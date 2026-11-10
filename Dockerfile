FROM python:3.12-slim

# libgomp is needed by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/
COPY models/ models/
ENV PYTHONPATH=/app/src PORT=8080

# Non-root user for the serving container
RUN useradd --create-home app && chown -R app /app
USER app

# Default: serve the API (Cloud Run). Vertex AI overrides the command to train.
CMD ["sh", "-c", "uvicorn macopt.api:app --host 0.0.0.0 --port ${PORT}"]
