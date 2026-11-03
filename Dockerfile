# OLGA Trader — Cloud Run Streamlit image.
# GitHub Actions builds this and updates the existing service. The image name
# is us-central1-docker.pkg.dev/olga-trader/olga/olga-trader.
# Do not bake API keys into the image — Cloud Run already holds them.

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Binary wheels cover pandas/numpy/streamlit — no compiler toolchain needed.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

RUN mkdir -p /app/.cache \
    && useradd --create-home --uid 1001 olga \
    && chown -R olga:olga /app

USER olga

EXPOSE 8080

# Cloud Run injects $PORT; shell form expands it.
CMD ["sh", "-c", "exec streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
