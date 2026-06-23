# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — TCR Autoreactivity Pipeline
# Base: Python 3.11 slim + R 4.x + all ML/bioinformatics dependencies
#
# Build:  docker build -t YOUR_USERNAME/tcr-autoreactivity:latest .
# Push:   docker push YOUR_USERNAME/tcr-autoreactivity:latest
# Test:   docker run --rm YOUR_USERNAME/tcr-autoreactivity:latest python --version
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

LABEL maintainer="Luis Chan <luis@lifebit.ai>"
LABEL description="TCR autoreactivity ML pipeline — thesis-based synthetic dataset + Elastic Net + R Shiny"
LABEL version="1.0.0"

# ─── System dependencies ─────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base \
    r-base-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    procps \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ─── Python dependencies ─────────────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ─── R dependencies ──────────────────────────────────────────────────────────
RUN Rscript -e "\
    install.packages(c( \
        'shiny', \
        'ggplot2', \
        'dplyr', \
        'readr', \
        'DT', \
        'plotly', \
        'shinydashboard', \
        'optparse' \
    ), repos='https://cloud.r-project.org', quiet=TRUE)"

# ─── Copy pipeline scripts ───────────────────────────────────────────────────
COPY bin/ /app/bin/
RUN chmod +x /app/bin/*.py /app/bin/*.R 2>/dev/null || true

# ─── Working directory ───────────────────────────────────────────────────────
WORKDIR /data

# ─── Default entrypoint ──────────────────────────────────────────────────────
CMD ["python", "--version"]
