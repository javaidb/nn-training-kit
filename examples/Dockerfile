FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install curl and jq for fetching latest release
RUN apt-get update && apt-get install -y curl jq && rm -rf /var/lib/apt/lists/*

# Install your package from latest release with all dependencies
RUN LATEST_RELEASE=$(curl -s https://api.github.com/repos/javaidb/nn-training-kit/releases/latest | jq -r '.assets[0].browser_download_url') && \
    pip install --no-cache-dir $LATEST_RELEASE

# Set working directory
WORKDIR /app

# Create directory for MLflow
RUN mkdir -p /app/mlruns