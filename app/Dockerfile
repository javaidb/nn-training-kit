FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

RUN apt-get update

WORKDIR /app

ADD https://github.com/javaidb/nn-training-kit/releases/download/v1.0.0/nn-training-kit-1.0.0.tar.gz dist/nn-training-kit-1.0.0.tar.gz
RUN pip install dist/*.tar.gz

# Copy the package
COPY . /app/

# Install the package in development mode
RUN pip install -e ".[dev]"

ENV NVIDIA_VISIBLE_DEVICES=all