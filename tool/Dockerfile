# Unified tool image: dataset downloader (gsutil) + video exporter (TensorFlow + ffmpeg)
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# System dependencies:
# - ffmpeg: video encoding for the exporter
# - curl, gnupg, ca-certificates: add & trust Google Cloud SDK apt repo
# - build-essential, python3-dev: compile crcmod (for fast gsutil sliced downloads)
# - google-cloud-cli: provides gsutil used by the downloader
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    gnupg \
    ca-certificates \
    build-essential \
    python3-dev \
  && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" > /etc/apt/sources.list.d/google-cloud-sdk.list \
  && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
  && apt-get update && apt-get install -y --no-install-recommends \
    google-cloud-cli \
  && rm -rf /var/lib/apt/lists/*

## Ensure gsutil runs with Python 3 inside the container
ENV CLOUDSDK_PYTHON=python3

# Python dependencies:
# - crcmod (compiled from source) enables fast sliced object downloads in gsutil
# - tensorflow/tensorflow-datasets/pillow/imageio[ffmpeg]/tqdm: video exporter stack
RUN pip install --no-cache-dir --no-binary=crcmod crcmod \
 && pip install --no-cache-dir \
    tensorflow==2.17.1 \
    tensorflow-datasets \
    pillow \
    imageio[ffmpeg] \
    tqdm \
    'replicate>=1.0.7'

## Application workspace
WORKDIR /app

## Bring in self-contained workflow components (Python-only)
COPY tool/export_videos.py /app/export_videos.py
COPY tool/download_dataset.py /app/download_dataset.py
COPY tool/generate_video.py /app/generate_video.py

## Unified CLI wrapper
COPY tool/run.py /app/run.py

RUN true

ENTRYPOINT ["python", "/app/run.py"]
CMD ["--help"]


