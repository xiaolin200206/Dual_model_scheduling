# Jetson Orin Nano Super

- **Device:** Jetson Orin Nano Super Developer Kit
- **JetPack:** 6.2 (L4T R36.4.4)
- **Power mode (`nvpmodel`):** _to be documented_
- **Camera:** USB 2K autofocus UVC webcam, MJPG @ 1280×720 (identical on both platforms)
- **Inference:** ONNX Runtime 1.23.0, `CUDAExecutionProvider`, via the Ultralytics ONNX backend
- **UPS:** Waveshare UPS Power Module (C), 3 × 21700 5,000 mAh; INA219 @0x41, SoC voltage-derived
- **Cooling:** stock active cooler
- **Software thermal cut-off:** 82 °C (`MAX_TEMP_LIMIT` in `main.py`)

## Configuration

| Parameter | Value |
|---|---|
| M_L (`Leaf` in code) | YOLO11s, 36.2 MB, 5 classes |
| M_S (`Pest` in code) | YOLO11n, 10.1 MB, 7 classes |
| Input resolution | 640 × 640 |
| Confidence / NMS IoU | 0.35 / 0.45 |
| M_L / M_S post-inference sleep | 0.8 s / 1.2 s |
| Stagger delay | 0.4 s |
| Duty cycle (capture gate) | 180 s on / 45 s off |
| Trial length | 3 h |
| Telemetry interval | 0.5 s |

## Data

Seven CSVs in `data/`: six mains-powered configurations (3 policies × 2 duty
states, ~21,100–21,600 rows each) and `jetson_battery.csv` (sequential + duty,
on pack power, outside climate control). Columns are listed in the top-level
README.

## Result summary (per-inference estimator)

| | parallel + duty | sequential + duty |
|---|---|---|
| M_S latency (ms) | 62.4 | 36.1 |
| M_S throughput (h⁻¹) | 2817 | 2629 |
| M_L latency (ms) | 84.6 | 86.3 |
| M_L throughput (h⁻¹) | 4040 | 2699 |
| Mean CPU (%) | 8.0 | 6.7 |
| Mean die temperature (°C) | 49.7 | 48.2 |

Sequential scheduling reduces M_S latency by 42% at a 7% throughput cost; for M_L
it is a net loss (latency +2%, throughput −33%). The mechanism on the GPU is not
instrumented; the paper describes it as shared-resource contention and does not
extend a cache-exclusivity account to this platform.

## Native install

```bash
# Jetson-specific wheels first (see requirements.txt header), then:
pip3 install -r requirements.txt
SCHEDULE_MODE=sequential DUTY_CYCLE_ENABLED=true python3 main.py
```

## Docker

Use the vendor-built image; hand-assembly from `l4t-jetpack` fails for the
reasons recorded in `docker/DOCKER_NOTES.md`.

```bash
cd docker && sudo docker build -t dualedge-jetson:v1 .
sudo docker run --rm --runtime nvidia --device /dev/video0 \
  -v $PWD/../output:/app/output -e SCHEDULE_MODE=sequential dualedge-jetson:v1
```
