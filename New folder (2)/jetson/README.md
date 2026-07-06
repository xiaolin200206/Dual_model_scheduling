# Jetson Orin Nano Super — Durian Dual-Model Benchmarking

**Status: ✅ COMPLETE** — All 6 configurations (3 scheduling modes × 2 duty-cycle states) are fully benchmarked with 3-hour runs each.

## Hardware & Environment

- **Device:** Jetson Orin Nano Super Developer Kit
- **JetPack:** 6.2 (L4T R36.4.4)
- **Camera:** USB 2K autofocus webcam (MJPG @ 1280×720)
- **Inference Engine:** ONNX Runtime 1.23.0, CUDAExecutionProvider
- **Thermal Management:** Hard cutoff at 82°C; duty-cycle throttling via 180s active / 45s sleep

## Experiment Configuration

| Parameter | Value |
|---|---|
| Leaf Model | YOLOv11s (36.2 MB, 5 classes) |
| Pest Model | YOLOv11n (10.1 MB, 7 classes) |
| Inference Resolution | 640×640 |
| Confidence Threshold | 0.35 |
| Leaf Inference Interval | 0.8s |
| Pest Inference Interval | 1.2s |
| Duty Cycle (when ON) | 180s active, 45s sleep |
| Run Duration per Config | 3 hours |
| Telemetry Interval | 0.5s |

## Datasets

7 CSV files in `data/`:
- `jetson_sequential_duty.csv` — Sequential mode, duty-cycle ON (~21,340 rows)
- `jetson_sequential_nonduty.csv` — Sequential mode, continuous (~21,145 rows)
- `jetson_parallel_duty.csv` — Parallel mode, duty-cycle ON
- `jetson_parallel_nonduty.csv` — Parallel mode, continuous
- `jetson_staggered_duty.csv` — Staggered mode, duty-cycle ON
- `jetson_staggered_nonduty.csv` — Staggered mode, continuous
- `jetson_battery.csv` — Ambient/outdoor thermal + battery validation (field test)

**CSV Format:** 17 columns (GPU and Battery columns populated on Jetson)
```
Timestamp, Schedule_Mode, FPS, Leaf_Lat_ms, Pest_Lat_ms,
CPU_%, RAM_MB, Temp_C, Freq_MHz, GPU_%, GPU_MHz,
Batt_Voltage_mV, Batt_Current_mA, Batt_Percent, Batt_State,
Leaf_Detections, Pest_Detections
```

## Key Findings

Sequential scheduling achieves **~42–44% reduction in pest-model latency** compared to concurrent (parallel/staggered) modes, with corresponding CPU/GPU cache efficiency gains:

| Metric | Parallel | Sequential | Reduction |
|---|---|---|---|
| Avg Pest Latency | 62.3 ms | 34.7 ms | 44.3% |
| Avg Temperature | 49.6 °C | 48.2 °C | 2.8% |
| Avg CPU % | 8.5% | 6.7% | 21.2% |

This pattern is consistent across duty-cycle ON/OFF states and validates the cache-exclusivity hypothesis.

## Running the Script

Both native and Docker workflows are supported. See [top-level README](../README.md) for full instructions.

### Quick Start (Native)

```bash
pip3 install -r requirements.txt
export SCHEDULE_MODE=sequential
export DUTY_CYCLE_ENABLED=True
python3 main.py
```

### Quick Start (Docker with GPU)

```bash
cp yolov11s.onnx yolov11n.onnx docker/
cd docker && docker build -t durian-jetson:v1 .
docker run --rm --runtime nvidia --device=/dev/video0 \
  -v $(pwd)/../output:/app/output \
  -e SCHEDULE_MODE=sequential \
  durian-jetson:v1
```

## Platform-Specific Notes

- **GPU Load Reading:** `/sys/devices/platform/17000000.gpu/load` (0–1000, divide by 10 for %)
- **GPU Frequency:** `/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/cur_freq` (Hz)
- **Battery (I2C INA219):** Reads from SMBus at address 0x40; requires Python `smbus2` library
- **Camera Interface:** Defaults to USB with V4L2 backend + MJPG format (see `main.py` for fallback CSI/Argus path)

## Cross-Platform Validation

See [top-level README](../README.md) and run `python3 compare_platforms.py` from repo root to compare Jetson results against Raspberry Pi 5.

---

**For Docker troubleshooting and GPU/container setup details, see [`docker/DOCKER_NOTES.md`](docker/DOCKER_NOTES.md).**
