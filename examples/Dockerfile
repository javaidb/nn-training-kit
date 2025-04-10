FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies - this layer will be cached unless requirements.txt changes
RUN pip install -r requirements.txt

COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV NVIDIA_VISIBLE_DEVICES=all

# Default command
CMD ["python", "examples/train_regression.py"] 