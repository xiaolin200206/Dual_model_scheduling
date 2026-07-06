# =========================================================
# Dockerfile — Raspberry Pi 5
# =========================================================
#
# Much simpler than the Jetson build (see ../../jetson/docker/
# DOCKER_NOTES.md for the full story of what goes wrong on Jetson).
# RPi5 has no GPU/CUDA path to fight with — CPUExecutionProvider only,
# and all Python packages install from standard PyPI without custom
# Jetson-specific wheels.
# =========================================================
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

# opencv-python-headless still needs a handful of system libs even
# without a GUI backend (libgl1 for headless rendering internals,
# libglib for gobject introspection used by some cv2 codepaths)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model files (yolov11s.onnx, yolov11n.onnx) must be placed next to
# this Dockerfile before building — same convention as jetson/docker/
COPY . .

CMD ["python3", "main.py"]
