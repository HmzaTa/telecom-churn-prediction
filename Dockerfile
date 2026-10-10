FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first so Docker can cache this layer
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Only the code the API needs (no plotting, training or Streamlit code)
COPY src/data_preprocessing.py src/feature_engineering.py src/inference.py src/
COPY api/ api/

# Run as a non-root user
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# The trained model is mounted at run time (see README), not baked into the image
ENV MODEL_PATH=/app/models/churn_model.pkl

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
