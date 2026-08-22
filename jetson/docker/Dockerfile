# =========================================================
# Dockerfile — Jetson Orin Nano Super (JetPack 6.2 / L4T R36.4.4)
# =========================================================
#
# WHY THIS BASE IMAGE:
# We tried building from nvcr.io/nvidia/l4t-jetpack directly and manually
# pip-installing torch / torchvision / onnxruntime-gpu. That path hit three
# separate, real problems documented in docker/DOCKER_NOTES.md:
#   1. Ultralytics' AutoUpdate silently replaced onnxruntime-gpu with a
#      CPU-only build from PyPI.
#   2. pip's dependency resolver rejected NVIDIA's custom-versioned torch
#      wheel (e.g. "2.5.0a0+872d972e41.nv24.8") as incompatible with
#      torchvision's exact pin ("torch==2.5.0"), even though they were the
#      correct matching pair.
#   3. Even after fixing 1 and 2, torch.cuda.is_available() still returned
#      False due to an ARM SVE cpuinfo detection error inside the l4t-jetpack
#      base image's userspace.
#
# Ultralytics maintains a pre-built image with all of this already solved
# and version-matched for each JetPack release. Use that instead of
# re-solving dependency hell yourself:
#   https://docs.ultralytics.com/guides/nvidia-jetson/
#
# If you are on a different JetPack major version, swap the tag below:
#   JetPack 4 -> ultralytics/ultralytics:latest-jetson-jetpack4
#   JetPack 5 -> ultralytics/ultralytics:latest-jetson-jetpack5
#   JetPack 6 -> ultralytics/ultralytics:latest-jetson-jetpack6
# =========================================================
FROM ultralytics/ultralytics:latest-jetson-jetpack6

WORKDIR /app

# Copies main.py, the two .onnx models, and anything else in this folder.
# Model files (yolov11s.onnx, yolov11n.onnx) must be placed next to the
# Dockerfile before building — see README.md "Build" section.
COPY . .

CMD ["python3", "main.py"]
