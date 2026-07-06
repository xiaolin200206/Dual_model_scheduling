# Docker on Jetson — Notes From Getting This Working

This is a debugging diary, written so it can double as a learning
resource. Every problem below actually happened while containerizing
the dual-model inference script on a Jetson Orin Nano Super
(JetPack 6.2 / L4T R36.4.4). If you're new to Docker, read this
top to bottom — it covers the core concepts (images, containers,
volumes, GPU passthrough) through real failures, which tends to
stick better than a pure tutorial.

---

## 1. The absolute basics

- **Image** = a frozen snapshot of a filesystem + instructions for how
  to run something inside it (built from a `Dockerfile`).
- **Container** = a running (or stopped) instance of an image. You can
  run many containers from the same image.
- **`docker build`** = read a `Dockerfile`, produce an image.
- **`docker run`** = start a container from an image.
- **`--rm`** = delete the container automatically when it stops
  (doesn't touch the image, just the temporary container instance).
- **Volume mount (`-v host_path:container_path`)** = make a folder on
  your real machine visible inside the container at some path. Without
  this, anything the container writes disappears when the container is
  removed (`--rm`) — this bit us directly, see Problem 3 below.

---

## 2. GPU passthrough: `--runtime nvidia`

By default, a container CANNOT see your GPU. On Jetson, GPU access
goes through `nvidia-container-toolkit`, and you enable it per-run
with `--runtime nvidia`:

```bash
sudo docker run --rm --runtime nvidia nvidia/cuda-image-name ...
```

**How to verify it's actually working**, before touching your real
code — check the GPU device nodes are visible inside the container:

```bash
sudo docker run --rm --runtime nvidia \
  nvcr.io/nvidia/l4t-base:r36.2.0 \
  ls -la /dev/nvhost-gpu /dev/nvmap
```

If both device files show up (not "No such file or directory"),
passthrough is working at the OS level. This does NOT yet mean your
Python libraries (PyTorch, ONNX Runtime) can use it — that's a
separate, higher-level problem, covered below.

Jetson does not have `nvidia-smi` (that's a desktop-GPU tool). Use
`tegrastats` on the **host**, not inside a minimal container image
(it's not bundled in slim base images).

---

## 3. Problem: files disappear after the container exits

**Symptom:** ran a 3-hour experiment inside a container with `--rm`,
came back, the CSV log was gone.

**Why:** a container's filesystem is layered on top of the image and
is thrown away when the container is removed. Anything written
*inside* the container that isn't explicitly mapped to the host disk
via `-v` vanishes with it.

**Fix:** mount a host directory to wherever your script writes output:

```bash
sudo docker run --rm --runtime nvidia \
  -v ~/docker_build/output:/app/output \
  my-image python3 main.py
```

**Gotcha we hit:** mounting to the *root* of the app directory
(`-v ~/docker_build/output:/app`) instead of a *subdirectory*
(`-v ~/docker_build/output:/app/output`) completely overwrites
`/app` with whatever's on the host side — including deleting
`main.py` and the model files that were baked into the image at
build time. Mount to a subdirectory unless you deliberately want to
replace the whole folder.

---

## 4. Problem: Ultralytics silently swapped our GPU-enabled ONNX Runtime for a CPU-only one

**Symptom:** container built fine, ran fine, but logs showed
`Using ONNX Runtime 1.23.2 with CPUExecutionProvider` — even though
we had explicitly installed `onnxruntime-gpu==1.23.0` and confirmed
`CUDAExecutionProvider` was available when testing standalone.

**Why:** Ultralytics has an "AutoUpdate" feature that checks whether
its dependencies are satisfied. It looked for a package literally
named `onnxruntime` (not `onnxruntime-gpu` — pip treats these as
different package names even though they conflict at the file level),
decided it was "missing", and auto-installed the standard PyPI
`onnxruntime` — which has no aarch64+CUDA build and silently falls
back to CPU-only.

**Fix:** disable Ultralytics' autoupdate behavior with an environment
variable in the Dockerfile:

```dockerfile
ENV YOLO_AUTOINSTALL=False
```

**Lesson:** if a version number in your logs looks like it "upgraded
itself" without you asking, some tool in your dependency chain
probably has an autoupdate/autoinstall feature you didn't know about.
Always print `import mylib; print(mylib.__version__)` right after
installing to catch this kind of silent substitution early.

---

## 5. Problem: pip refuses to install NVIDIA's custom torch build

**Symptom:**

```
ERROR: Cannot install torch 2.5.0a0+872d972e41.nv24.8 and torchvision
because these package versions have conflicting dependencies.
The conflict is caused by: torchvision 0.20.0 depends on torch==2.5.0
```

**Why:** NVIDIA ships Jetson-specific PyTorch builds with a custom
version suffix (`2.5.0a0+872d972e41.nv24.8`) baked in — this exact
build is compiled against Jetson's CUDA/cuDNN stack and is NOT
interchangeable with the standard PyPI `torch==2.5.0`. But pip's
resolver does a literal string comparison and decides
`2.5.0a0+872d972e41.nv24.8 != 2.5.0`, so it thinks the two packages
are incompatible — even though they're the exact matching pair NVIDIA
intends you to use together.

**Fix (works, but fragile):** install each package in its own
`pip install --no-deps` call, so pip never evaluates them against
each other in the same dependency resolution pass:

```dockerfile
RUN pip3 install --no-deps torch@<nvidia_url>
RUN pip3 install --no-deps torchvision==0.20.0
```

**Better fix (what we ended up using):** don't fight this at all —
see Section 7.

---

## 6. Problem: `torch.cuda.is_available()` returns `False` even after fixing everything above

**Symptom:**

```
Error in cpuinfo: prctl(PR_SVE_GET_VL) failed
False
```

**Why:** this is a known compatibility gap between certain PyTorch
builds and ARM's SVE (Scalable Vector Extension) instruction set
detection inside some `l4t-jetpack` base image variants. It's a CPU
feature-detection bug inside PyTorch's `cpuinfo` submodule, unrelated
to CUDA itself — but it can crash PyTorch's startup sequence before
it even gets to checking for a GPU, causing CUDA detection to report
`False` even though the GPU and drivers are completely fine.

This is the point where we stopped trying to hand-solve every layer
of the dependency stack ourselves.

---

## 7. The actual fix: use Ultralytics' pre-built Jetson image

After the three problems above, we switched the base image entirely:

```dockerfile
FROM ultralytics/ultralytics:latest-jetson-jetpack6
```

instead of building up from `nvcr.io/nvidia/l4t-jetpack` by hand.
Ultralytics maintains and tests this image specifically so that
torch, torchvision, onnxruntime-gpu, and TensorRT are all pre-installed,
version-matched, and confirmed working for each JetPack release. Full
matrix of tags: <https://docs.ultralytics.com/guides/nvidia-jetson/>.

**Verify it before building anything on top of it:**

```bash
sudo docker run --rm --runtime nvidia -it \
  ultralytics/ultralytics:latest-jetson-jetpack6 \
  python3 -c "import torch; print(torch.cuda.is_available()); \
              import onnxruntime; print(onnxruntime.get_available_providers())"
```

Expected output:

```
True
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

**Lesson, generalized:** if you're hand-assembling a GPU + deep
learning + embedded-ARM environment from scratch and keep hitting
version conflicts, check whether the hardware vendor (NVIDIA) or a
major library maintainer (Ultralytics, PyTorch) already publishes a
pre-built image for your exact hardware/OS combination *before*
spending hours resolving dependency conflicts by hand. This is true
well beyond Jetson — it's a general Docker/ML-ops pattern.

---

## 8. Problem: model paths were wrong inside the container

**Symptom:** `[FAIL] Leaf ONNX NOT FOUND: /home/lin/yolov11s.onnx`

**Why:** the script had the model path hardcoded to a path on the
*host* machine (`/home/lin/...`). A container has its own, separate
filesystem — `COPY . .` in the Dockerfile only copies files into
`/app` (or wherever `WORKDIR` points), not into `/home/lin/` inside
the container.

**Fix:** never hardcode host-specific absolute paths in a script meant
to run in a container. Use paths relative to the script's own location
(`os.path.dirname(os.path.abspath(__file__))`), or read from an
environment variable with a sensible default — both patterns are used
in `main.py` in this repo.

---

## 9. Security note: don't bake secrets into the image

Docker images are frequently pushed to registries, shared, or — if
you ever push this repo's Dockerfile publicly with a hardcoded
`TELEGRAM_BOT_TOKEN` baked in — permanently embedded in a layer's
history (removing it from the Dockerfile later does NOT remove it
from already-built image layers or git commit history).

Instead, pass secrets in at `docker run` time via `-e`:

```bash
sudo docker run --rm --runtime nvidia \
  -e TELEGRAM_BOT_TOKEN="your_token" \
  -e TELEGRAM_CHAT_ID="your_chat_id" \
  my-image python3 main.py
```

and read them in Python via `os.environ.get(...)` (see `main.py`).
If a token was ever committed in plaintext anywhere, treat it as
compromised — revoke and regenerate it, don't just delete the line.

---

## 10. Full command reference used in this project

```bash
# 1. Confirm GPU device nodes are visible in a container
sudo docker run --rm --runtime nvidia \
  nvcr.io/nvidia/l4t-base:r36.2.0 ls -la /dev/nvhost-gpu /dev/nvmap

# 2. Pull the pre-built Ultralytics Jetson image
sudo docker pull ultralytics/ultralytics:latest-jetson-jetpack6

# 3. Verify torch + onnxruntime see the GPU inside it
sudo docker run --rm --runtime nvidia -it \
  ultralytics/ultralytics:latest-jetson-jetpack6 \
  python3 -c "import torch; print(torch.cuda.is_available())"

# 4. Build our image on top of it (run from repo root, models must be
#    copied next to the Dockerfile first — see README.md)
cd docker && sudo docker build -t durian-jetson:test .

# 5. Run a full experiment, with output volume-mounted and secrets
#    passed as env vars, in a detached tmux session so SSH disconnects
#    don't kill it (see README.md "Running a full 3-hour experiment")
tmux new -s docker_run
timeout 10800 sudo docker run --rm --runtime nvidia --device=/dev/video0 \
  -v ~/docker_build/output:/app/output \
  -e TELEGRAM_BOT_TOKEN="..." -e TELEGRAM_CHAT_ID="..." \
  -e SCHEDULE_MODE="sequential" -e DUTY_CYCLE_ENABLED="True" \
  durian-jetson:test python3 main.py 2>&1 | tee ~/docker_build/run_log.txt
# Ctrl+B then D to detach
```
