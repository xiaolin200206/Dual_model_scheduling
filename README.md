# DualEdge: characterising battery-powered dual-model edge inference

Code, raw telemetry and analysis for the study *"Characterising battery-powered
dual-model edge inference: an open framework and a two-platform ARM study of
scheduling, energy and cost."*

**DualEdge** is a small, platform-agnostic framework for measuring what a
multi-model inference node actually costs at the battery rail. One runtime runs two
co-located ONNX detectors under three scheduling policies and an optional duty
cycle, and logs per-model latency, CPU/GPU utilisation, die temperature, clock and
pack-side voltage / current / state of charge in the same row at 0.5 s. The study
applies it to a Raspberry Pi 5 and a Jetson Orin Nano Super: twelve three-hour
trials (3 policies × 2 duty states × 2 platforms), two containerised trials, and two
battery-powered trials outside climate control.

Headline results, all regenerable from the CSVs in this repository:

- Sequential scheduling cuts the smaller model's latency by 38% (Pi 5) and 42%
  (Jetson) and near-eliminates the tail on the Pi 5 — at a cost of 19% / 7% of that
  model's throughput, and at a net loss for the larger model on the Jetson.
- At the pack rail the Jetson draws 11.7 W vs 8.2 W. Capacity-normalised to a common
  60 W h pack, endurance is 5.1 h vs 7.3 h, and energy per cycle is 4.44 vs
  4.20 mW h — a 5× latency advantage that buys no energy advantage per unit of work.
- A two-state decomposition of the pack trace shows why: the Jetson idles at
  11.4 W for 91% of wall time, so all inference-runtime optimisation together is
  worth at most ~12% of energy per cycle on this node.
- Docker costs nothing measurable at run time on either platform.

## Repository structure

```
.
├── raspberry-pi/ , jetson/         identical main.py (platform auto-detected)
│   ├── main.py                     capture, dual-model scheduling, telemetry
│   ├── battery_backends.py         UPS monitor accessors (identical on both)
│   ├── docker/                     Dockerfile (+ DOCKER_NOTES.md on Jetson)
│   ├── data/                       8 CSV each: 6 configurations + docker + battery
│   └── analysis/summarize_results.py
├── compare_platforms.py            side-by-side summary of all configurations
├── reproduce/
│   └── make_figures.py             regenerate Figs. 1–4 from the telemetry
├── analysis/
│   └── two_state_decomposition.py  idle / inferring split, energy floor, crossover
├── ondemand/
│   └── ondemand_session.py         instrumented operator-triggered session (future work)
├── figures/                        Figs. 1–4 at 300 dpi (PNG + PDF)
├── legacy/                         dataset-audit and mAP tooling for the deployed
│   ├── audit/ , accuracy/          weights; not used by any result in the paper
├── all_csv.zip                     the same 14 CSVs, zipped
├── CITATION.cff
└── LICENSE                         MIT
```

## The two models

The workload is a pair of independently trained YOLO11 detectors exported to ONNX
with Ultralytics 8.4.63. The paper refers to them by size; the code and CSV columns
use the names they were trained under.

| Paper | Code / CSV name | Architecture | Classes | Size | SHA256 |
|---|---|---|---|---|---|
| **M_L** (large) | `Leaf` / `yolov11s_leaf.onnx` | YOLO11s | 5 | 36.2 MB | `1934813e2319fa1f9ab60fb681ae5838324bc3a53dc3be18a23f23b9e5a1636f` |
| **M_S** (small) | `Pest` / `yolov11n_pest.onnx` | YOLO11n | 7 | 10.1 MB | `45f18a01d6e0936a2500b15ef6e745200433ba27d8769a091a2040a016c9ab46` |

The weights are not released (they are assets of a commercial effort). No result in
the paper depends on what they have learned — latency, throughput, utilisation,
temperature and pack power are properties of the architecture, input resolution
and schedule — so **any two ONNX detectors of the same architectures at 640 × 640
can be dropped in** to re-run the framework. Set `LEAF_MODEL_PATH` and
`PEST_MODEL_PATH` to your own files.

## Running the framework

```bash
pip install -r raspberry-pi/requirements.txt      # or jetson/requirements.txt
SCHEDULE_MODE=sequential DUTY_CYCLE_ENABLED=true python3 raspberry-pi/main.py
```

Environment variables:

| Variable | Values | Meaning |
|---|---|---|
| `SCHEDULE_MODE` | `parallel` / `staggered` / `sequential` | co-location policy (Section 3.2 of the paper) |
| `DUTY_CYCLE_ENABLED` | `true` / `false` | gate **frame capture** for 45 s in every 225 s; inference workers keep running |
| `DUALEDGE_PLATFORM` (or legacy `DURIAN_PLATFORM`) | `rpi5` / `jetson` | override platform auto-detection |
| `LEAF_MODEL_PATH`, `PEST_MODEL_PATH` | path | ONNX files for M_L and M_S |
| `BATT_BACKEND`, `BATT_I2C_BUS` | see Battery instrumentation | pack-rail monitor |
| `USE_USB_CAMERA`, `USB_CAMERA_INDEX` | | camera selection |
| `OUTPUT_DIR` | path | where telemetry CSV and detections are written |

Timers are `LEAF_INTERVAL = 0.8 s`, `PEST_INTERVAL = 1.2 s`, `STAGGER_DELAY = 0.4 s`,
`LOG_INTERVAL = 0.5 s`, software thermal cut-off `MAX_TEMP_LIMIT = 82 °C`.

**Adding a platform** means implementing four accessors in `main.py`:
`get_cpu_temp`, `get_cpu_freq`, `get_gpu_load`, `get_battery_data`.

Containerised runs: see `raspberry-pi/docker/` and `jetson/docker/`. The Jetson
target needs the vendor-built `ultralytics/ultralytics:latest-jetson-jetpack6`
image; `jetson/docker/DOCKER_NOTES.md` records why hand-assembly from `l4t-jetpack`
fails.

## Reproducing the paper

```bash
python compare_platforms.py                          # Table 1 summary
python reproduce/make_figures.py --repo . --out figures/
python analysis/two_state_decomposition.py           # Tables 5–6, Fig. 4 numbers
```

Every number in the paper's Sections 5–6 regenerates from `*/data/*.csv`. Note the
telemetry schema carries the last inference latency forward between inferences;
the scripts de-duplicate consecutive values to obtain per-inference statistics
(paper Section 3.3). A row-weighted mean is length-biased and gives slightly
different numbers.

## Telemetry format

0.5 s sampling. Raspberry Pi 5 logs 15 columns, Jetson 17 (adding `GPU_%`,
`GPU_MHz`); the battery trials populate `Batt_Voltage_mV`, `Batt_Current_mA`,
`Batt_Percent`, `Batt_State`. Roughly 21,000 rows per three-hour trial.

| File | Configuration |
|---|---|
| `*_parallel_duty.csv` / `_nonduty` | two worker threads, zero start offset |
| `*_staggered_duty.csv` / `_nonduty` | two threads, M_S delayed 0.4 s at start |
| `*_sequential_duty.csv` / `_nonduty` | one thread, M_L then M_S per frame |
| `*_docker_sequential_duty.csv` | sequential + duty, inside a container |
| `*_battery.csv` | sequential + duty, on pack power, outside climate control |

`_duty` = capture gated 180 s on / 45 s off; `_nonduty` = continuous capture.

## Battery instrumentation

| Platform | UPS module | Monitor | State of charge |
|---|---|---|---|
| Raspberry Pi 5 | Waveshare UPS HAT (E), 4 × 21700 (4S) | BQ4050 fuel gauge via on-board MCU, I2C 0x2D | gauge reading |
| Jetson Orin Nano Super | Waveshare UPS Power Module (C), 3 × 21700 (3S) | INA219, I2C 0x41 | vendor linear map of pack voltage (9.0–12.6 V → 0–100 %) — **not** a fuel gauge |

Both use 5,000 mAh cells. Voltage and current are direct measurements on both
platforms; only the `Batt_Percent` column differs in provenance. Select the
backend with `BATT_BACKEND=waveshare_ups_hat_e` (Pi) or
`BATT_BACKEND=waveshare_ina219` (Jetson); see `battery_backends.py`.

```bash
BATT_BACKEND=waveshare_ups_hat_e python3 battery_backends.py       # Raspberry Pi 5
BATT_BACKEND=waveshare_ina219 BATT_I2C_BUS=7 python3 battery_backends.py   # Jetson
```

The Raspberry Pi accessor is verified against the published logs (16.6 V,
-257 mA, 93 %, discharging). The Jetson accessor follows the vendor reference
implementation and reproduces the formula behind `jetson_battery.csv`, but has
not yet been re-run on the device; the I2C bus number and address
(`BATT_I2C_BUS`, `BATT_INA_ADDR`) may need adjusting.

## Known gaps

- The Jetson `nvpmodel` power mode in force during the trials was not recorded.
  Logged GPU clocks alternate between the 306 MHz idle and 714 MHz active
  operating points, consistent with a mode below MAXN SUPER.

## Limitations

Each platform–configuration cell is a single three-hour trial; within-trial
uncertainty is quantified by block bootstrap, which is not a substitute for
replication. The duty-cycle condition gates capture, not computation. The
battery-powered trials were not matched for ambient temperature. Offered load is not
held constant across scheduling policies — the sequential policy issues fewer
inferences by construction, and the paper reports throughput alongside latency for
that reason.

## Licence

See `LICENSE`. Please cite the paper if you use the framework or the telemetry.
