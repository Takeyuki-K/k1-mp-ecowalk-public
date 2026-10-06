# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
# Reproducible CPU environment for K1 MP Eco-Walk (training, evaluation, video rendering).
#   docker build -t k1-mp-ecowalk .
#   docker run --rm -it -v "$PWD":/work k1-mp-ecowalk scripts/smoke_test.sh
FROM python:3.13-slim-bookworm

# rendering (OSMesa / OpenGL), video encoding, caption font, git for the pinned upstream fetch
RUN apt-get update && apt-get install -y --no-install-recommends \
        libosmesa6 libgl1 libglib2.0-0 ffmpeg fonts-noto-cjk git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /tmp/requirements.txt

ENV PYTHONUNBUFFERED=1
CMD ["bash"]
