# Raspberry Pi 5

- **Device:** Raspberry Pi 5, 8 GB, Cortex-A76 ×4 @ 2.4 GHz
- **OS:** Raspberry Pi OS (Bookworm), 64-bit
- **Camera:** USB 2K autofocus UVC webcam, MJPG @ 1280×720 (identical on both platforms)
- **Inference:** ONNX Runtime, `CPUExecutionProvider`, via the Ultralytics ONNX backend
- **UPS:** Waveshare UPS HAT (E), 4 × 21700 5,000 mAh; BQ4050 fuel gauge via MCU @0x2D
- **Cooling:** official Active Cooler, default fan curve
- **Software thermal cut-off:** 82 °C (`MAX_TEMP_LIMIT` in `main.py`)

## Configuration

Identical to the Jetson (see `jetson/README.md`): same models, resolution,
thresholds, timers, duty gate, trial length and telemetry interval. `main.py`
is byte-identical on both platforms.

## Data

Seven CSVs in `data/`: six mains-powered configurations (~21,200–21,500 rows
each) and `raspberry_battery.csv` (sequential + duty, on pack power, outside
climate control). Note: the battery trial ran at ~8.3 FPS camera capture against
~16.5 FPS in the mains-powered trials, so its latencies are not directly
comparable with the mains-powered ones (paper Section 4.5).

## Result summary (per-inference estimator)

| | parallel + duty | sequential + duty |
|---|---|---|
| M_S latency (ms) | 301.3 | 185.8 |
| M_S P95 (ms) | 586.5 | 210.5 |
| M_S inferences > 400 ms | 27.2% | 0.02% |
| M_S throughput (h⁻¹) | 2384 | 1935 |
| M_L latency (ms) | 551.8 | 462.3 |
| M_L throughput (h⁻¹) | 2651 | 1943 |
| Mean CPU (%) | 51.8 | 38.1 |
| Mean die temperature (°C) | 64.8 | 60.2 |

Sequential scheduling reduces M_S latency by 38% at a 19% throughput cost and
removes the tail almost entirely; for M_L it saves 16% latency for 27%
throughput. Concurrent-mode stalls recur with a ~15 s period, which is why the
0.4 s stagger offset does not help.

## Install

```bash
pip install -r requirements.txt
SCHEDULE_MODE=sequential DUTY_CYCLE_ENABLED=true python3 main.py
```

Camera note: on this platform do **not** pass `cv2.CAP_V4L2` explicitly (the
device fails to open); on the Jetson it is required. `main.py` handles this by
platform detection.

## Docker

```bash
cd docker && docker build -t dualedge-rpi5:v1 .
docker run --rm --device /dev/video0 -v $PWD/../output:/app/output \
  -e SCHEDULE_MODE=sequential dualedge-rpi5:v1
```
