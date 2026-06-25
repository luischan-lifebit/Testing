# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile.r — optional, kept for reference
# Not required if using public image in nextflow.config
# Build:  podman build -t quay.io/luischanlifebit/tcr-autoreactivity-r:latest -f Dockerfile.r .
# Push:   podman push quay.io/luischanlifebit/tcr-autoreactivity-r:latest
# ─────────────────────────────────────────────────────────────────────────────

FROM rocker/shiny-verse:latest

LABEL maintainer="Luis Chan <luis@lifebit.ai>"

RUN Rscript -e "install.packages(c( \
    'DT', \
    'plotly', \
    'shinydashboard', \
    'optparse' \
    ), repos='https://cloud.r-project.org', quiet=TRUE)"

COPY bin/generate_shiny_report.R /usr/local/bin/generate_shiny_report.R
RUN chmod +x /usr/local/bin/generate_shiny_report.R

WORKDIR /data

ENTRYPOINT []
