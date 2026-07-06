# Durian Edge AI — Cross-Platform Dual-Model Inference Study

Benchmarking of dual-model (leaf disease + pest detection) real-time edge inference on **Jetson Orin Nano Super** and **Raspberry Pi 5**, comparing three scheduling strategies (staggered / parallel / sequential) under duty-cycled and continuous operation.

**Status: ✅ COMPLETE** — Both platforms have full datasets (6 configurations × 3 hours each).

## Repository Structure

```
.
├── README.md / README_zh.md          # this file
├── compare_platforms.py               # cross-platform comparison tool
├── jetson/                            # Jetson Orin Nano Super (complete)
│   ├── main.py                        # unified cross-platform script
│   ├── requirements.txt
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── DOCKER_NOTES.md
│   ├── data/                          # 7 CSV files (6 configs + battery)
│   └── analysis/
├── raspberry-pi/                      # Raspberry Pi 5 (complete)
│   ├── main.py                        # same script as jetson/
│   ├── requirements.txt
│   ├── docker/
│   ├── data/                          # 7 CSV files (6 configs + battery)
│   └── analysis/
└── .gitignore
```

Both `jetson/main.py` and `raspberry-pi/main.py` are **identical** — platform auto-detection at startup picks the right code paths (CPU thermal zones, camera interfaces, GPU reads). No manual script swapping needed.

## Why Cross-Platform

Previous single-platform work (Jetson only) observed that **sequential scheduling reduces pest-model latency by ~35% vs. concurrent modes**, attributed to CPU/GPU cache exclusivity — the two models share cache when run concurrently, causing mutual eviction.

**The research question:** does this effect hold on a platform with **different cache geometry** (RPi5's smaller L2 on Cortex-A76, vs. Jetson's larger hierarchy)? If the sequential advantage is comparable on both, it's a general principle of co-located inference rather than a platform artifact.

**Finding:** ✅ The effect **does generalize**. Sequential scheduling achieves ~39–40% latency reduction on RPi5, and ~42–44% on Jetson — consistent relative magnitude despite the cache size disparity.

## Datasets

| Platform | Hardware | Models | Scheduling Modes | Duty-Cycle States | Duration per Config | Total Rows |
|---|---|---|---|---|---|---|
| **Jetson** | Orin Nano Super (CUDA) | YOLOv11s + YOLOv11n (ONNX) | 3 (seq/par/stag) | 2 (on/off) | 3 hours | ~127K |
| **RPi5** | Cortex-A76 CPU | YOLOv11s + YOLOv11n (ONNX) | 3 (seq/par/stag) | 2 (on/off) | 3 hours | ~127K |

**CSV Format:** 17 columns (unified across both platforms)
```
Timestamp, Schedule_Mode, FPS, Leaf_Lat_ms, Pest_Lat_ms,
CPU_%, RAM_MB, Temp_C, Freq_MHz, GPU_%, GPU_MHz,
Batt_Voltage_mV, Batt_Current_mA, Batt_Percent, Batt_State,
Leaf_Detections, Pest_Detections
```

**Note on GPU/Battery columns:**
- **Jetson:** GPU_% and GPU_MHz populated (NVIDIA Jetson-specific via `/sys/devices/platform/17000000.gpu`); Battery columns from I2C INA219 module.
- **RPi5:** GPU_% and GPU_MHz are 0 (CPU-only); Battery columns are 0 / "N/A" (no battery module).

## Running Experiments

### Native (no Docker)

```bash
# Install dependencies
pip3 install -r jetson/requirements.txt  # or raspberry-pi/requirements.txt

# Export configuration
export SCHEDULE_MODE=sequential                 # staggered | parallel | sequential
export DUTY_CYCLE_ENABLED=True                 # True | False
export OUTPUT_DIR=./output

# Run
python3 jetson/main.py  # or raspberry-pi/main.py — script auto-detects platform
```

### Docker (recommended on Jetson for GPU passthrough)

```bash
# Prepare
cp yolov11s.onnx yolov11n.onnx jetson/docker/

# Build & run
cd jetson/docker
docker build -t durian-jetson:v1 .
docker run --rm --runtime nvidia --device=/dev/video0 \
  -v ~/output:/app/output \
  -e SCHEDULE_MODE=sequential -e DUTY_CYCLE_ENABLED=True \
  durian-jetson:v1
```

**Note:** RPi5 users can use Docker without GPU complications (no CUDA). Both platforms use the **same script with unified USB camera interface** (V4L2 + MJPG format) as the default.

## Cross-Platform Comparison

Once both datasets are collected, run:

```bash
python3 compare_platforms.py
```

Output shows side-by-side metrics and % deltas (Jetson vs RPi5) per configuration.

Example:
```
Config           Platform  Rows      Duration     AvgTemp  AvgCPU  AvgFPS  LeafLat  PestLat
─────────────────────────────────────────────────────────────────────────────────────────
sequential_duty  Jetson    21340  3h 0m 38s       48.2     6.7     8.21    85.3     34.7
                 RPi5      21457  3h 0m 12s       60.2    18.1     8.27   489.1    186.0
                 Δ(J/R)                           -19.9%  -63.0%   -0.7%   -82.5%   -81.4%
```

## Known Limitations & Data Notes

1. **Single trial per configuration:** Each run is one continuous 3-hour session. Uncertainty comes from block bootstrap (40-sample windows), not repeated trials. See paper for confidence intervals.

2. **Jetson ambient field test (outdoor thermal/battery validation):** The first attempt was discarded (data-logger outage left a 3.09-hour telemetry gap, and battery I2C reads were non-physical). `jetson_battery.csv` is the complete re-run after fixing the I2C interface: 3h02m continuous (21,500 rows, no gaps), with a full physical discharge curve (82%→16% SoC, 11.99V→9.52V) and pulsed GPU telemetry (306/714 MHz).

3. **FPS artifacts:** Duty-cycle ON configurations occasionally show FPS spikes >100 (vs. stable 8–9 baseline) at active/sleep transition boundaries. These are filtering artifacts from the FPS calculation denominator, not real variations; `compare_platforms.py` excludes FPS >20 before averaging.

4. **USB camera as unified interface:** Both platforms default to the same USB camera (1280×720, MJPG format) for direct comparability. RPi5's native Picamera2/CSI capability exists but is set aside to maintain hardware-level parity where possible.

## Citation

If you use this repository or its datasets in research:

```bibtex
@misc{durian-cross-platform-2026,
  title     = {Cross-Platform Validation of Scheduling-Aware Dual-Model Edge Inference},
  author    = {Anonymous},
  year      = {2026},
  url       = {https://github.com/your-user/durian-edge-scheduling},
  note      = {Jetson Orin Nano Super \& Raspberry Pi 5 dual-model inference benchmarks}
}
```

---

> 中文説明は [README_zh.md](README_zh.md) を参照してください。
