# GridSentinel inference service (Phase 3).
# Build:  docker build -t gridsentinel .
# Run:    docker run -p 8000:8000 -v "$PWD/models:/models" gridsentinel
# The model artifact is mounted at /models (built offline via
# `python -m serving.build_artifact`), never baked into the image.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    MODEL_PATH=/models/anomaly_detector.joblib

# Install deps first for layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[serving]"

# App code.
COPY pipelines ./pipelines
COPY serving ./serving

EXPOSE 8000
# Non-root for safety.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
