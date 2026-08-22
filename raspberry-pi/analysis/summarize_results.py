#!/usr/bin/env python3
"""
Summarize and compare the 6 schedule x duty-cycle CSV logs in ../data/.

Usage:
    python3 summarize_results.py

Notes:
    - FPS readings above 20 are excluded as timing artifacts (transient
      spikes observed at duty-cycle active/sleep transition boundaries,
      most likely caused by a near-zero denominator in the FPS calculation
      during a single anomalous frame interval). See README.md "Known data
      quality notes" for details.
    - This script only depends on the Python standard library (csv,
      collections) so it runs anywhere without extra installs.
"""
import csv
import glob
import os
from collections import namedtuple

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
FPS_ARTIFACT_THRESHOLD = 20.0  # readings above this are excluded, see docstring

Summary = namedtuple("Summary", [
    "name", "n_rows", "duration",
    "avg_temp", "max_temp", "avg_cpu", "max_cpu",
    "avg_fps", "avg_leaf_lat", "avg_pest_lat",
])


def parse_ts(ts):
    import datetime
    for fmt in ("%H:%M:%S.%f", "%H:%M:%S"):
        try:
            return datetime.datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None


def summarize_file(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    name = os.path.basename(path).replace("dual_", "").rsplit("_", 2)[0]

    temps = [float(r["Temp_C"]) for r in rows if r["Temp_C"]]
    cpus = [float(r["CPU_%"]) for r in rows if r["CPU_%"]]
    fps_vals = [float(r["FPS"]) for r in rows if r["FPS"] and float(r["FPS"]) < FPS_ARTIFACT_THRESHOLD]
    leaf_lat = [float(r["Leaf_Lat_ms"]) for r in rows if r["Leaf_Lat_ms"] and r["Leaf_Lat_ms"] != "0.0"]
    pest_lat = [float(r["Pest_Lat_ms"]) for r in rows if r["Pest_Lat_ms"] and r["Pest_Lat_ms"] != "0.0"]

    t0, t1 = parse_ts(rows[0]["Timestamp"]), parse_ts(rows[-1]["Timestamp"])
    if t0 and t1:
        delta = t1 - t0
        if delta.total_seconds() < 0:
            # run crossed midnight; both timestamps are time-of-day only
            # (no date), so add 24h to correct the negative wraparound
            import datetime
            delta += datetime.timedelta(days=1)
        duration = str(delta)
    else:
        duration = "unknown"

    return Summary(
        name=name,
        n_rows=len(rows),
        duration=duration,
        avg_temp=sum(temps) / len(temps) if temps else 0,
        max_temp=max(temps) if temps else 0,
        avg_cpu=sum(cpus) / len(cpus) if cpus else 0,
        max_cpu=max(cpus) if cpus else 0,
        avg_fps=sum(fps_vals) / len(fps_vals) if fps_vals else 0,
        avg_leaf_lat=sum(leaf_lat) / len(leaf_lat) if leaf_lat else 0,
        avg_pest_lat=sum(pest_lat) / len(pest_lat) if pest_lat else 0,
    )


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "dual_*.csv")))
    if not files:
        print(f"No CSV files found in {DATA_DIR}")
        return

    summaries = [s for s in (summarize_file(f) for f in files) if s]

    header = f"{'Config':<28}{'Rows':>7}  {'Duration':>15}{'AvgTemp':>9}{'MaxTemp':>9}{'AvgCPU':>8}{'MaxCPU':>8}{'AvgFPS':>8}{'LeafLat':>9}{'PestLat':>9}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(f"{s.name:<28}{s.n_rows:>7}  {s.duration:>15}{s.avg_temp:>9.2f}{s.max_temp:>9.2f}"
              f"{s.avg_cpu:>8.2f}{s.max_cpu:>8.2f}{s.avg_fps:>8.2f}{s.avg_leaf_lat:>9.2f}{s.avg_pest_lat:>9.2f}")


if __name__ == "__main__":
    main()
