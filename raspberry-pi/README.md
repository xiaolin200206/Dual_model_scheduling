# Raspberry Pi 5 — Durian Dual-Model Benchmarking

**Status: ✅ COMPLETE** — All 6 configurations (3 scheduling modes × 2 duty-cycle states) plus outdoor battery validation.

## Hardware & Environment

- **Device:** Raspberry Pi 5
- **Processor:** ARM Cortex-A76 (8-core)
- **RAM:** 8 GB
- **Camera:** USB 2K autofocus webcam (MJPG @ 1280×720) — same as Jetson for cross-platform parity
- **Inference Engine:** ONNX Runtime (CPUExecutionProvider, no GPU acceleration)
- **Thermal Management:** Hard cutoff at 82°C; duty-cycle via 180s active / 45s sleep

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
- `raspberry_sequential_duty.csv` — Sequential mode, duty-cycle ON (~21,457 rows)
- `raspberry_sequential_nonduty.csv` — Sequential mode, continuous (~21,298 rows)
- `raspberry_parallel_duty.csv` — Parallel mode, duty-cycle ON
- `raspberry_parallel_nonduty.csv` — Parallel mode, continuous
- `raspberry_staggered_duty.csv` — Staggered mode, duty-cycle ON
- `raspberry_staggered_nonduty.csv` — Staggered mode, continuous
- `raspberry_battery.csv` — Outdoor/ambient thermal + battery discharge validation

**CSV Format:** 17 columns (same as Jetson; GPU_% / GPU_MHz / Batt columns are 0 / "N/A" on RPi5)
```
Timestamp, Schedule_Mode, FPS, Leaf_Lat_ms, Pest_Lat_ms,
CPU_%, RAM_MB, Temp_C, Freq_MHz, GPU_%, GPU_MHz,
Batt_Voltage_mV, Batt_Current_mA, Batt_Percent, Batt_State,
Leaf_Detections, Pest_Detections
```

## Key Findings

Sequential scheduling achieves **~39–40% reduction in pest-model latency** vs. concurrent modes, closely mirroring the Jetson result (~42–44%) despite the RPi5's smaller cache and CPU-only execution:

| Metric | Parallel | Sequential | Reduction |
|---|---|---|---|
| Avg Pest Latency | 322.2 ms | 186.0 ms | 42.3% |
| Avg Temperature | 62.5 °C | 60.2 °C | 3.7% |
| Avg CPU % | 39.1% | 18.1% | 53.7% |

**Implication:** The cache-exclusivity effect is **not GPU-specific** and **generalizes across ARM CPU architectures**, validating the cross-platform hypothesis.

## Running the Script

The `main.py` in this directory is **identical** to `../jetson/main.py` — platform auto-detection handles all differences (thermal zones, camera setup, GPU reads). No script swapping needed.

### Native (Recommended)

```bash
pip3 install -r requirements.txt
export SCHEDULE_MODE=sequential
export DUTY_CYCLE_ENABLED=True
python3 main.py
```

### Docker

```bash
cp yolov11s.onnx yolov11n.onnx docker/
cd docker && docker build -t durian-rpi5:v1 .
docker run --rm --device=/dev/video0 \
  -v $(pwd)/../output:/app/output \
  -e SCHEDULE_MODE=sequential \
  durian-rpi5:v1
```

## Platform-Specific Notes

- **Thermal Zone:** RPi5 exposes `/sys/class/thermal/thermal_zone0/temp` (single zone, reliable CPU reading)
- **GPU/Battery:** Not applicable; script outputs 0 / "N/A" for GPU and battery columns
- **Camera Interface:** Defaults to USB with V4L2 backend + MJPG (unified across platforms)
- **CPU Frequency:** Read from `/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq`

## Cross-Platform Analysis

Run `python3 compare_platforms.py` from the repo root to see Jetson vs. RPi5 side-by-side results:
- Temperature differences (Jetson 48–49°C vs. RPi5 60–62°C under same load)
- CPU utilization gaps (Jetson ~7% vs. RPi5 ~18% for sequential mode)
- Latency multiplier (RPi5 is ~5–6× slower in absolute terms, but maintains the cache-exclusivity pattern)

---

**See [top-level README](../README.md) for full context and cross-platform comparison instructions.**
