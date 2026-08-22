# Cross-Platform Selection for On-Farm Dual-Model Edge Inference

Code, telemetry and audit tooling for the study *"What does a farm gain from a 3×
more expensive edge platform? Cost, endurance and thermal trade-offs between
Raspberry Pi 5 and Jetson Orin Nano Super for on-farm disease and pest inspection."*

Two co-located YOLO detectors — one for durian leaf disease, one for pests — are run
on both platforms under one protocol, across three scheduling strategies × two
duty-cycle states × two platforms (twelve three-hour trials), plus containerisation
trials and outdoor battery trials. The output is a **platform-selection rule**, not a
single recommendation: the ranking reverses with deployment condition.

## Companion studies

This repository belongs to a group of three studies on the same deployment. Each
answers a different question, and they should not be read as one result split three
ways:

| Study | Question | Repository |
|---|---|---|
| Lin (2026a), *Smart Agric. Technol.*, under review | Is the dataset behind the accuracy figure sound, and what actually ends a session thermally? | [Edge-Disease-Inference-Engine](https://github.com/xiaolin200206/Edge-Disease-Inference-Engine) |
| Lin (2026b), *IEEE TICPS*, under review | How do two co-located models interact on one platform? | — |
| **This study** | **Which platform should a given holding buy?** | this repository |

The dataset-integrity audit procedure applied in `audit/` originates in Lin (2026a)
and its released script; the version here is adapted to the five-class leaf and
seven-class pest taxonomy of the deployed weights.

## Repository structure

```
.
├── compare_platforms.py            cross-platform comparison tool
├── jetson/ , raspberry-pi/         identical main.py (platform auto-detected)
│   ├── main.py                     capture, dual-model scheduling, telemetry
│   ├── docker/                     containerisation trial
│   ├── data/                       7 CSV each (6 configs + battery trial)
│   └── analysis/
├── audit/                          dataset-integrity audit
│   ├── find_dataset.py             locate partitions matching the deployed classes
│   ├── check_leakage_phash.py      MD5 + perceptual-hash cross-partition check
│   └── build_clean_eval.py         emit a source-level leak-free evaluation set
├── accuracy/
│   ├── run_model_eval.py           mAP and per-class P·R on the audited partitions
│   └── sensitivity_pest.py         settings sweep for cross-study comparison
├── ondemand/
│   └── ondemand_session.py         instrumented on-demand inspection session
├── reproduce/
│   └── make_figures.py             regenerate every figure from the telemetry
└── figures/                        Fig. 1–3 at 300 dpi (PNG + PDF)
```

`jetson/main.py` and `raspberry-pi/main.py` are identical: platform detection at
startup selects the thermal-zone paths, camera backend and GPU reads.

## Deployed models

Detection accuracy reported in the paper is measured on the **exported ONNX
artefacts**, not on the PyTorch checkpoints. The weights are not released (they are
assets of an ongoing commercialisation effort), but the artefacts evaluated are
identified by digest so that the claim is checkable:

| File | Classes | Export | SHA256 |
|---|---|---|---|
| `yolov11s_leaf.onnx` | 5 | Ultralytics 8.4.63, 2026-06-17 | `1934813e2319fa1f9ab60fb681ae5838324bc3a53dc3be18a23f23b9e5a1636f` |
| `yolov11n_pest.onnx` | 7 | Ultralytics 8.4.63, 2026-06-19 | `45f18a01d6e0936a2500b15ef6e745200433ba27d8769a091a2040a016c9ab46` |

Leaf classes: `algal`, `leaf_rot`, `Phomopsis`, `pink`, `root`.
Pest classes: `leafhopper damage`, `Psyllid`, `Psyllid_damage`, `Scale_insect`,
`Stem-borer`, `weevil`, `weevil_damage`.

The display maps in `main.py` were previously written for an earlier taxonomy
(`algal_leave`, `anthracnose`, `early_blight`, `Mealybugs`, `Red_Spider`) and have
been corrected to the classes above.

## Reproducing the results

**Telemetry results (paper Sections 4.1–4.4).** Fully reproducible from this
repository:

```bash
python compare_platforms.py
python reproduce/make_figures.py --repo . --out figures/
```

**Accuracy (paper Section 3.2).** Not reproducible by a third party, because the
images and weights are not released. The procedure is released so that it can be run
against any detection dataset:

```bash
python audit/find_dataset.py                 # locate candidate partitions
python audit/check_leakage_phash.py          # cross-partition duplicate check
python audit/build_clean_eval.py             # emit the leak-free evaluation set
python accuracy/run_model_eval.py            # mAP and per-class figures
```

On the partitions used in the paper the audit removed 37 of 184 leaf images (147
retained, 471 instances). Exclusion was not uniform across classes: Phomopsis lost
48% of its instances and leaf_rot 35%, while pink and root lost none.

**On-demand session.** The intermittent duty the paper argues for is identified as
future work; the instrumentation is released so that the measurement can be made:

```bash
DURIAN_PLATFORM=rpi5   SESSION_MIN=60 python3 ondemand/ondemand_session.py
DURIAN_PLATFORM=jetson SESSION_MIN=60 python3 ondemand/ondemand_session.py
python3 ondemand/ondemand_session.py analyze <telemetry.csv> <events.csv>
```

## Telemetry format

Per-inference sampling at 0.5 s. Raspberry Pi 5 logs 15 columns, Jetson 17 (adding
`GPU_%` and `GPU_MHz`); the battery trials add `Batt_Voltage_mV`,
`Batt_Current_mA`, `Batt_Percent` and `Batt_State`. Roughly 21,000 samples per
three-hour configuration.

| File | Configuration |
|---|---|
| `*_sequential_duty.csv` | 180 s active / 45 s sleep, single worker thread |
| `*_sequential_nonduty.csv` | continuous, single worker thread |
| `*_parallel_duty.csv` / `_nonduty` | two threads, zero startup offset |
| `*_staggered_duty.csv` / `_nonduty` | two threads, pest delayed 0.4 s |
| `*_battery.csv` | outdoor trial, sequential + duty-cycle, on pack power |

## What is not in this repository

The durian disease and pest image datasets and the trained weights are assets of an
ongoing commercialisation effort and are not released. This restriction applies to
those two artefacts only: all code, all telemetry, and the audit manifest (released
as perceptual hashes rather than as images) are available here.

## Limitations carried in the paper

Each platform-configuration combination is a single three-hour trial; within-trial
uncertainty is quantified by block bootstrap, which is not a substitute for
replication across trials. The outdoor trials were not matched for ambient
conditions and are reported as preliminary, not as agricultural field validation.

## Licence

See `LICENSE`. Please cite the paper if you use the telemetry or the audit tooling.
