# Per-policy battery trials (Raspberry Pi 5, 3 Sep 2026)

Three one-hour runs on pack power, consecutively on a single discharge (93% -> 49% SoC,
17:16-21:20 local time, one room), one per scheduling policy, duty cycle enabled.
Each run begins with a 60 s ready-state baseline (`IDLE_BASELINE_SEC=60`): both models
loaded, camera open, telemetry written, no inference issued. Rows in the baseline have
`Leaf_Lat_ms = Pest_Lat_ms = 0`.

| File | Policy | Start SoC | Notes |
|---|---|---|---|
| raspberry_parallel_duty_battery_1h.csv | parallel | 93% | |
| raspberry_staggered_duty_battery_1h.csv | staggered | 62% | re-run; a first attempt (81% → 75%) ended at 29 min when a failed USB frame read on duty resume exited the runtime (fixed in main.py) |
| raspberry_staggered_duty_battery_aborted_29min.csv | staggered | 81% | the aborted first attempt (18:17–18:46, 81% → 75%); its 28 min of data agree with the full re-run and are not used in the paper |
| raspberry_sequential_duty_battery_1h.csv | sequential | 75% | replicate of `../raspberry_battery.csv` (3 h, nine weeks earlier) |

Software (see `versions_rpi5_20260903.txt`): ONNX Runtime 1.29.0, Ultralytics 8.4.138, OpenCV 5.0.0, NumPy 2.2.4,
torch 2.14.0, Raspberry Pi OS kernel 6.18.39. The models themselves are the 8.4.63 exports used in every other trial.
`notes.txt` is the run log written by the launcher script: start/end times and a single pack reading per
line (single samples, before `main.py` starts and after it is killed; use the 60 s baseline inside each CSV,
not these one-shot readings, for idle power).

Because the runtime is newer than the one used for the mains ablation, latencies here are 6–12% lower than in
`../raspberry_*_duty.csv` and are not compared with them in the paper; power and energy are compared only among
these three runs and with `../raspberry_battery.csv`.

Reproduce Table 4 and the baseline of Section 5.2 with `python analysis/rpi5_policy_energy.py`
from the repository root.
