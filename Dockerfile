# ===========================================================================
# QuantBot container image
#
# Multi-stage: dependencies are built once in a builder layer and copied into
# a slim runtime, so the final image carries no compiler toolchain.
#
#   docker build -t quantbot .
#   docker run --rm --env-file .env quantbot config-check
# ===========================================================================

# --- Stage 1: build the virtualenv -----------------------------------------
FROM python:3.11-slim-bookworm AS builder

# Build tools are needed for the numpy/scipy wheels on architectures without
# prebuilt binaries (arm64 VPS, for instance). They stay in this stage only.
RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential gcc g++ gfortran \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the requirements first: this layer is then cached across every
# source change, which is the difference between a 5-second and a 5-minute
# rebuild.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt


# --- Stage 2: runtime -------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="quantbot" \
      org.opencontainers.image.description="Risk-first trading-signal bot (signal/paper only)" \
      org.opencontainers.image.licenses="MIT"

# tzdata: the bot reasons in UTC everywhere, but a correct zone database keeps
# log timestamps and any local-time reporting honest.
# curl: used by the healthcheck and for debugging from inside the container.
RUN apt-get update && apt-get install --no-install-recommends -y \
        tzdata curl \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    TZ=UTC

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Run as an unprivileged user. A bot holding exchange credentials has no
# business running as root, even in a container.
RUN useradd --create-home --shell /bin/bash --uid 10001 quantbot

COPY --chown=quantbot:quantbot config/ ./config/
COPY --chown=quantbot:quantbot data/ ./data/
COPY --chown=quantbot:quantbot indicators/ ./indicators/
COPY --chown=quantbot:quantbot analysis/ ./analysis/
COPY --chown=quantbot:quantbot strategies/ ./strategies/
COPY --chown=quantbot:quantbot risk/ ./risk/
COPY --chown=quantbot:quantbot backtest/ ./backtest/
COPY --chown=quantbot:quantbot notify/ ./notify/
COPY --chown=quantbot:quantbot live/ ./live/
COPY --chown=quantbot:quantbot utils/ ./utils/
COPY --chown=quantbot:quantbot main.py pyproject.toml ./

# Mount points for the things that must outlive the container.
RUN mkdir -p /app/logs /app/data/cache /app/reports \
    && chown -R quantbot:quantbot /app/logs /app/data /app/reports

USER quantbot

# Fails the container if the configuration stops being valid (a bad mount, a
# missing env var). Cheap, and catches the most common deploy mistake.
HEALTHCHECK --interval=5m --timeout=30s --start-period=40s --retries=3 \
    CMD python main.py config-check > /dev/null || exit 1

ENTRYPOINT ["python", "main.py"]
CMD ["run"]
