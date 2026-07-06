#!/usr/bin/env python3
"""
Cross-platform comparison: Jetson Orin Nano Super vs. Raspberry Pi 5.

CSV naming convention (both platforms):
  - jetson_<scheduling>_<dutycycle>.csv  →  config name: "<scheduling>_<dutycycle>"
  - raspberry_<scheduling>_<dutycycle>.csv

17-column format: Timestamp, Schedule_Mode, FPS, Leaf_Lat_ms, Pest_Lat_ms,
  CPU_%, RAM_MB, Temp_C, Freq_MHz, GPU_%, GPU_MHz,
  Batt_Voltage_mV, Batt_Current_mA, Batt_Percent, Batt_State,
  Leaf_Detections, Pest_Detections
  (Jetson: GPU/Batt populated; RPi5: 0 / N/A)
"""

import csv, glob, os, datetime
from collections import namedtuple

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
FPS_ARTIFACT_THRESHOLD = 20.0

Summary = namedtuple("Summary", [
    "config", "n_rows", "duration",
    "avg_temp", "max_temp", "avg_cpu", "max_cpu",
    "avg_fps", "avg_leaf_lat", "avg_pest_lat",
])

def parse_ts(ts):
    for fmt in ("%H:%M:%S.%f", "%H:%M:%S"):
        try:
            return datetime.datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None

def config_name_from_filename(path):
    base = os.path.basename(path).replace(".csv", "")
    if base.startswith("jetson_"):
        return base.replace("jetson_", "")
    elif base.startswith("raspberry_"):
        return base.replace("raspberry_", "")
    return base

def summarize_file(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    ts_list = []
    for r in rows:
        t = parse_ts(r.get("Timestamp", ""))
        if t:
            ts_list.append(t)
    
    duration = ts_list[-1] - ts_list[0] if len(ts_list) > 1 else datetime.timedelta(0)

    temps = [float(r["Temp_C"]) for r in rows if r["Temp_C"]]
    cpus = [float(r["CPU_%"]) for r in rows if r["CPU_%"]]
    fps_vals = [float(r["FPS"]) for r in rows if r["FPS"] and float(r["FPS"]) < FPS_ARTIFACT_THRESHOLD]
    leaf_lat = [float(r["Leaf_Lat_ms"]) for r in rows if r["Leaf_Lat_ms"] and float(r["Leaf_Lat_ms"]) > 0]
    pest_lat = [float(r["Pest_Lat_ms"]) for r in rows if r["Pest_Lat_ms"] and float(r["Pest_Lat_ms"]) > 0]

    return Summary(
        config=config_name_from_filename(path),
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

def load_platform(platform_dir):
    files = sorted(glob.glob(os.path.join(REPO_ROOT, platform_dir, "data", "*.csv")))
    files = [f for f in files if "battery" not in os.path.basename(f)]
    return {(s := summarize_file(f)).config: s for f in files if (s := summarize_file(f))}

def pct_delta(j_val, r_val):
    return "N/A" if r_val == 0 else f"{(j_val - r_val) / r_val * 100:+.1f}%"

def main():
    jetson = load_platform("jetson")
    rpi5 = load_platform("raspberry-pi")

    if not jetson:
        print("❌ No Jetson data found in jetson/data/")
        return
    
    print("\n" + "="*100)
    print("  Durian Dual-Model Edge Inference – Cross-Platform Comparison")
    print("  Jetson Orin Nano Super (CUDA) vs. Raspberry Pi 5 (CPU)")
    print("="*100 + "\n")

    if not rpi5:
        print("⚠️  Raspberry Pi 5 data incomplete. Showing Jetson only.\n")

    all_configs = sorted(set(jetson.keys()) | set(rpi5.keys()))

    header = f"{'Config':<20} {'Platform':<8} {'Rows':>7} {'Duration':>12} {'AvgTemp':>8} {'AvgCPU':>7} {'AvgFPS':>7} {'LeafLat':>8} {'PestLat':>8}"
    print(header)
    print("-" * len(header))

    for cfg in all_configs:
        j = jetson.get(cfg)
        r = rpi5.get(cfg)

        if j:
            dur = f"{int(j.duration.total_seconds()//3600)}h{int((j.duration.total_seconds()%3600)//60)}m"
            print(f"{cfg:<20} {'Jetson':<8} {j.n_rows:>7} {dur:>12} {j.avg_temp:>8.1f} {j.avg_cpu:>7.1f} {j.avg_fps:>7.2f} {j.avg_leaf_lat:>8.1f} {j.avg_pest_lat:>8.1f}")

        if r:
            dur = f"{int(r.duration.total_seconds()//3600)}h{int((r.duration.total_seconds()%3600)//60)}m"
            print(f"{'':<20} {'RPi5':<8} {r.n_rows:>7} {dur:>12} {r.avg_temp:>8.1f} {r.avg_cpu:>7.1f} {r.avg_fps:>7.2f} {r.avg_leaf_lat:>8.1f} {r.avg_pest_lat:>8.1f}")

        if j and r:
            print(f"{'':<20} {'Δ(J/R)':<8} {'':<7} {'':<12} {pct_delta(j.avg_temp, r.avg_temp):>8} {pct_delta(j.avg_cpu, r.avg_cpu):>7} {pct_delta(j.avg_fps, r.avg_fps):>7} {pct_delta(j.avg_leaf_lat, r.avg_leaf_lat):>8} {pct_delta(j.avg_pest_lat, r.avg_pest_lat):>8}")
        print()

    print("="*100 + "\n")

if __name__ == "__main__":
    main()
