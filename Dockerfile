FROM python:3.11-slim AS wheel

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TRANSCRIBEX_HOST=0.0.0.0 \
    TRANSCRIBEX_PORT=8000

WORKDIR /build

COPY pyproject.toml README.md ./
COPY transcribex ./transcribex

RUN python -m pip install --upgrade pip \
    && python -m pip wheel --no-deps --wheel-dir /dist .

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TRANSCRIBEX_HOST=0.0.0.0 \
    TRANSCRIBEX_PORT=8000 \
    TRANSCRIBEX_WORK_DIR=/tmp/transcribex \
    TRANSCRIBEX_MODEL_CACHE_DIR=/models \
    TRANSCRIBEX_CONFIG_PATH=/config/config.json \
    MODELSCOPE_CACHE=/models/modelscope \
    HF_HOME=/models/huggingface

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libgomp1 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/transcribex

COPY requirements.txt ./

RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url "${PYTORCH_INDEX_URL}" torch torchaudio \
    && python -m pip install -r requirements.txt

COPY --from=wheel /dist/transcribex-*.whl /tmp/

RUN python -m pip install --no-deps /tmp/transcribex-*.whl \
    && rm -f /tmp/transcribex-*.whl \
    && mkdir -p /models /config /tmp/transcribex

EXPOSE 8000
VOLUME ["/models", "/config"]

CMD ["python", "-m", "uvicorn", "transcribex.main:app", "--host", "0.0.0.0", "--port", "8000"]
