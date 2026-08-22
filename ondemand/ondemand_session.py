#!/usr/bin/env python3
# =========================================================
# Durian AI - ON-DEMAND INSPECTION SESSION (complete)
# Jetson Orin Nano Super & Raspberry Pi 5
#
# Simulates the deployment mode the paper argues for: an operator
# triggers an inspection on a suspect leaf, waits for the verdict,
# then walks on (idle). Measures per-event end-to-end verdict
# latency and continuous telemetry, then computes per-session
# energy under the REAL intermittent duty pattern.
#
# Class maps below are written for the DEPLOYED weights
# (verified against model_report.txt, exports of 2026-06-17/19):
#   leaf : algal, leaf_rot, Phomopsis, pink, root
#   pest : leafhopper damage, Psyllid, Psyllid_damage,
#          Scale_insect, Stem-borer, weevil, weevil_damage
#
# Run a session:
#   DURIAN_PLATFORM=rpi5   SESSION_MIN=60 python3 ondemand_session.py
#   DURIAN_PLATFORM=jetson SESSION_MIN=60 python3 ondemand_session.py
# Manual triggering (press Enter per inspection):
#   TRIGGER_MODE=manual python3 ondemand_session.py
# Re-analyze existing CSVs without running a session:
#   python3 ondemand_session.py analyze <telemetry.csv> <events.csv>
# =========================================================

import cv2
import time
import sys
import os
import csv
import random
import threading
import statistics
import traceback
import psutil
from datetime import datetime

# ================= PLATFORM DETECTION (same logic as main.py) =================
PLATFORM = os.environ.get("DURIAN_PLATFORM", "").lower()
if not PLATFORM:
    PLATFORM = "jetson" if os.path.exists("/sys/devices/platform/17000000.gpu") else "rpi5"
IS_JETSON = PLATFORM == "jetson"

# ================= CONFIG =================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LEAF_MODEL_PATH = os.environ.get("LEAF_MODEL_PATH", os.path.join(_SCRIPT_DIR, "yolov11s.onnx"))
PEST_MODEL_PATH = os.environ.get("PEST_MODEL_PATH", os.path.join(_SCRIPT_DIR, "yolov11n.onnx"))

SESSION_MIN  = float(os.environ.get("SESSION_MIN", "60"))     # session length, minutes
TRIGGER_MODE = os.environ.get("TRIGGER_MODE", "auto").lower() # auto | manual
TRIG_MIN     = float(os.environ.get("TRIG_MIN", "20"))        # auto: min idle gap, s
TRIG_MAX     = float(os.environ.get("TRIG_MAX", "40"))        # auto: max idle gap, s

CONF_THRESH        = 0.35
INFERENCE_SIZE     = 640
MAX_BOX_AREA_RATIO = 0.5
LOG_INTERVAL       = 0.5

USB_CAMERA_INDEX      = int(os.environ.get("USB_CAMERA_INDEX", "0"))
USB_CAMERA_RESOLUTION = (1280, 720)
KEEP_CAMERA_OPEN = os.environ.get("KEEP_CAMERA_OPEN", "true").lower() == "true"

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(_SCRIPT_DIR, "output"))

# ================= CLASS MAPS - CURRENT WEIGHTS =================
# Display names for the five leaf classes actually in yolov11s_leaf.onnx.
# (The LEAF_MERGE_MAP in the repo's main.py targets an earlier set of
#  weights - 'algal_leave', 'anthracnose', 'early_blight' etc. no longer
#  exist - and should be replaced with this block there too.)
LEAF_DISPLAY_MAP = {
    "algal":     "Algal Spot",
    "leaf_rot":  "Leaf Rot",
    "phomopsis": "Phomopsis",
    "pink":      "Pink Disease",
    "root":      "Root Disease",
}

PEST_DISPLAY_MAP = {
    "leafhopper damage": "Leafhopper Damage",
    "psyllid":           "Psyllid",
    "psyllid_damage":    "Psyllid Damage",
    "scale_insect":      "Scale Insect",
    "stem-borer":        "Stem Borer",
    "weevil":            "Weevil",
    "weevil_damage":     "Weevil Damage",
}

PEST_COLORS = {
    "Leafhopper Damage": (0,   165, 255),
    "Psyllid":           (255, 255, 0  ),
    "Psyllid Damage":    (200, 200, 0  ),
    "Scale Insect":      (255, 128, 0  ),
    "Stem Borer":        (128, 0,   255),
    "Weevil":            (0,   255, 128),
    "Weevil Damage":     (0,   200, 100),
}
LEAF_COLOR    = (0, 255, 0)
DEFAULT_COLOR = (200, 200, 200)

def display_name(model_tag, raw_name):
    m = LEAF_DISPLAY_MAP if model_tag == "leaf" else PEST_DISPLAY_MAP
    return m.get(raw_name.lower(), raw_name)

# ================= SYSTEM TELEMETRY HELPERS =================
def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return float(f.read()) / 1000.0
    except Exception:
        return 0.0

def get_cpu_freq():
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") as f:
            return int(f.read()) / 1000.0
    except Exception:
        return 0.0

def get_gpu_load():
    if not IS_JETSON:
        return 0.0, 0.0
    gpu_pct = gpu_mhz = 0.0
    try:
        with open("/sys/devices/platform/17000000.gpu/load") as f:
            gpu_pct = int(f.read().strip()) / 10.0
    except Exception:
        pass
    try:
        with open("/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/cur_freq") as f:
            gpu_mhz = int(f.read().strip()) / 1e6
    except Exception:
        pass
    return gpu_pct, gpu_mhz

# ================= BATTERY =================
# Jetson: INA219 at 0x40, identical to the repo's main.py.
# RPi5 (Waveshare UPS HAT (E)): the register map below follows the repo
# convention but MUST be checked once against the vendor reference code
# you validated for the paper's Section 3.5 ambient trial - if your
# validated read differs (address or registers), replace _read_rpi5_ups
# with that code verbatim. A one-off cross-check against the HAT's own
# display/readout before the real sessions is enough.
_smbus_warned = False

def _read_ina219(bus, addr):
    volt = (bus.read_word_data(addr, 0x02) >> 3) * 4 / 1000.0   # mV
    curr = bus.read_word_data(addr, 0x01) / 1000.0              # mA (signed-ish)
    return volt, curr

def _read_rpi5_ups(bus):
    # Try the common Waveshare UPS I2C addresses in order; first that
    # answers wins. Replace with your validated Section-3.5 read if it
    # differs.
    for addr in (0x40, 0x41, 0x42, 0x43):
        try:
            volt, curr = _read_ina219(bus, addr)
            if volt > 0:
                return volt, curr
        except Exception:
            continue
    raise IOError("no UPS fuel gauge found on I2C bus 1")

def get_battery_data():
    """(volt_mV, curr_mA, pct, state) - pct is a voltage-derived estimate."""
    global _smbus_warned
    try:
        import smbus2
        bus = smbus2.SMBus(1)
        if IS_JETSON:
            volt, curr = _read_ina219(bus, 0x40)
            v_empty, v_full = 9.0, 12.6      # 3S pack window used in ambient trial
        else:
            volt, curr = _read_rpi5_ups(bus)
            v_empty, v_full = 12.0, 16.8     # 4S pack window (UPS HAT (E))
        bus.close()
        pct = max(0.0, min(100.0, (volt / 1000.0 - v_empty) / (v_full - v_empty) * 100))
        state = "discharging" if curr < -100 else ("charging" if curr > 100 else "idle")
        return volt, curr, pct, state
    except Exception as e:
        if not _smbus_warned:
            print(f"[BATT] telemetry unavailable ({e}); logging zeros")
            _smbus_warned = True
        return 0.0, 0.0, 0.0, "N/A"

# ================= STATE =================
running = True
phase = "idle"                    # idle | inspecting
phase_lock = threading.Lock()
last_lats = {"leaf": 0.0, "pest": 0.0}

# ================= TELEMETRY THREAD =================
def monitor_worker(csv_path):
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow([
            "Timestamp", "Phase", "Leaf_Lat_ms", "Pest_Lat_ms",
            "CPU_%", "RAM_MB", "Temp_C", "Freq_MHz", "GPU_%", "GPU_MHz",
            "Batt_Voltage_mV", "Batt_Current_mA", "Batt_Percent", "Batt_State"
        ])
    while running:
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().used / 1024 / 1024
        temp, freq = get_cpu_temp(), get_cpu_freq()
        gpct, gmhz = get_gpu_load()
        bv, bc, bp, bs = get_battery_data()
        with phase_lock:
            ph = phase
        try:
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    datetime.now().strftime('%H:%M:%S.%f')[:-3], ph,
                    f"{last_lats['leaf']:.1f}", f"{last_lats['pest']:.1f}",
                    f"{cpu:.1f}", f"{ram:.1f}", f"{temp:.1f}", f"{freq:.0f}",
                    f"{gpct:.1f}", f"{gmhz:.0f}",
                    f"{bv:.0f}", f"{bc:.0f}", f"{bp:.0f}", bs
                ])
        except Exception:
            traceback.print_exc()
        time.sleep(LOG_INTERVAL)

# ================= INFERENCE =================
def load_models():
    from ultralytics import YOLO
    print("[LOAD] models ...")
    return (YOLO(LEAF_MODEL_PATH, task="detect"),
            YOLO(PEST_MODEL_PATH, task="detect"))

def _run(model, model_tag, frame_bgr):
    t0 = time.time()
    results = model(frame_bgr, imgsz=INFERENCE_SIZE, conf=CONF_THRESH,
                    iou=0.45, agnostic_nms=True, verbose=False)
    lat = (time.time() - t0) * 1000
    dets = []
    frame_area = INFERENCE_SIZE * INFERENCE_SIZE
    if results[0].boxes:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            if (x2 - x1) * (y2 - y1) > frame_area * MAX_BOX_AREA_RATIO:
                continue
            dets.append(display_name(model_tag, model.names[int(box.cls[0])]))
    return dets, lat

# ================= CAMERA =================
def open_camera():
    cap = cv2.VideoCapture(USB_CAMERA_INDEX, cv2.CAP_V4L2) if IS_JETSON \
        else cv2.VideoCapture(USB_CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, USB_CAMERA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, USB_CAMERA_RESOLUTION[1])
    return cap

# ================= ONE INSPECTION EVENT =================
def inspect(cap, leaf_model, pest_model, event_id):
    global phase
    t_trigger = time.time()
    with phase_lock:
        phase = "inspecting"
    cap_local = cap if cap is not None else open_camera()
    try:
        t0 = time.time()
        for _ in range(3):          # flush stale buffered frames
            cap_local.grab()
        ret, frame = cap_local.read()
        t_capture = (time.time() - t0) * 1000
        if not ret:
            print(f"[#{event_id:03d}] camera returned no frame; skipped")
            return None
        leaf_dets, leaf_lat = _run(leaf_model, "leaf", frame)
        pest_dets, pest_lat = _run(pest_model, "pest", frame)
        last_lats["leaf"], last_lats["pest"] = leaf_lat, pest_lat
        e2e = (time.time() - t_trigger) * 1000
    finally:
        if cap is None:
            cap_local.release()
        with phase_lock:
            phase = "idle"
    row = [event_id, datetime.now().strftime('%H:%M:%S.%f')[:-3],
           f"{t_capture:.1f}", f"{leaf_lat:.1f}", f"{pest_lat:.1f}", f"{e2e:.1f}",
           "|".join(leaf_dets) or "None", "|".join(pest_dets) or "None"]
    print(f"[#{event_id:03d}] verdict {e2e:.0f} ms "
          f"(cap {t_capture:.0f} + leaf {leaf_lat:.0f} + pest {pest_lat:.0f})  "
          f"leaf={row[6]} pest={row[7]}")
    return row

# ================= ANALYSIS =================
def analyze(telemetry_csv, events_csv):
    """Session energy + duty ratio + verdict latency stats from the CSVs."""
    # ---- events ----
    e2es, caps, leafs, pests = [], [], [], []
    with open(events_csv) as f:
        for r in csv.DictReader(f):
            e2es.append(float(r["EndToEnd_ms"]))
            caps.append(float(r["Capture_ms"]))
            leafs.append(float(r["Leaf_Lat_ms"]))
            pests.append(float(r["Pest_Lat_ms"]))
    # ---- telemetry ----
    n_idle = n_insp = 0
    p_idle, p_insp, p_all = [], [], []
    t_idle, t_insp = [], []
    batt_pct_first = batt_pct_last = None
    with open(telemetry_csv) as f:
        for r in csv.DictReader(f):
            v = float(r["Batt_Voltage_mV"]) / 1000.0
            i = abs(float(r["Batt_Current_mA"])) / 1000.0
            w = v * i if v > 0 else 0.0
            temp = float(r["Temp_C"])
            pct = float(r["Batt_Percent"])
            if v > 0:
                if batt_pct_first is None:
                    batt_pct_first = pct
                batt_pct_last = pct
                p_all.append(w)
            if r["Phase"] == "inspecting":
                n_insp += 1
                t_insp.append(temp)
                if v > 0:
                    p_insp.append(w)
            else:
                n_idle += 1
                t_idle.append(temp)
                if v > 0:
                    p_idle.append(w)
    dur_h = (n_idle + n_insp) * LOG_INTERVAL / 3600.0
    duty = n_insp / max(1, n_idle + n_insp)

    def stats(xs):
        if not xs:
            return "n/a"
        xs = sorted(xs)
        q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
        return (f"mean {statistics.mean(xs):.0f} | median {q(.5):.0f} | "
                f"p95 {q(.95):.0f} | max {xs[-1]:.0f} ms")

    print("\n================ SESSION ANALYSIS ================")
    print(f"platform            : {PLATFORM}")
    print(f"session duration    : {dur_h:.2f} h  ({n_idle + n_insp} samples)")
    print(f"inspections         : {len(e2es)}")
    print(f"duty ratio          : {duty * 100:.1f}% of samples in 'inspecting'")
    print(f"end-to-end verdict  : {stats(e2es)}")
    print(f"  capture component : {stats(caps)}")
    print(f"  leaf component    : {stats(leafs)}")
    print(f"  pest component    : {stats(pests)}")
    if t_idle and t_insp:
        print(f"temp: idle mean {statistics.mean(t_idle):.1f} C | "
              f"inspecting mean {statistics.mean(t_insp):.1f} C | "
              f"session max {max(t_idle + t_insp):.1f} C")
    if p_all:
        e_wh = statistics.mean(p_all) * dur_h
        print(f"mean pack-side power: {statistics.mean(p_all):.2f} W "
              f"(idle {statistics.mean(p_idle):.2f} W / "
              f"inspecting {statistics.mean(p_insp):.2f} W)"
              if p_idle and p_insp else
              f"mean pack-side power: {statistics.mean(p_all):.2f} W")
        print(f"session energy      : ~{e_wh:.1f} Wh over {dur_h:.2f} h "
              f"-> ~{e_wh / dur_h * 4:.1f} Wh per 4-h session "
              f"(paper Table 5 sustained extrapolation: 33 Wh RPi5 / 47 Wh Jetson)")
        if batt_pct_first is not None:
            print(f"pack SoC            : {batt_pct_first:.0f}% -> {batt_pct_last:.0f}%")
    else:
        print("battery telemetry absent - energy not computed")
    print("==================================================")

# ================= MAIN =================
def main():
    global running
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_tel = os.path.join(OUTPUT_DIR, f"ondemand_{PLATFORM}_{TRIGGER_MODE}_{_stamp}_telemetry.csv")
    csv_evt = os.path.join(OUTPUT_DIR, f"ondemand_{PLATFORM}_{TRIGGER_MODE}_{_stamp}_events.csv")

    print(f"[PLATFORM] {'Jetson Orin Nano Super' if IS_JETSON else 'Raspberry Pi 5'}")
    print(f"[SESSION] {SESSION_MIN:.0f} min | trigger={TRIGGER_MODE} "
          f"gap={TRIG_MIN:.0f}-{TRIG_MAX:.0f}s | camera_open={KEEP_CAMERA_OPEN}")

    leaf_model, pest_model = load_models()

    with open(csv_evt, 'w', newline='') as f:
        csv.writer(f).writerow(["Event", "Timestamp", "Capture_ms",
                                "Leaf_Lat_ms", "Pest_Lat_ms", "EndToEnd_ms",
                                "Leaf_Detections", "Pest_Detections"])

    cap = open_camera() if KEEP_CAMERA_OPEN else None

    # warm-up (graph init on first call, both EPs) - excluded from events
    wcap = cap if cap is not None else open_camera()
    ret, wframe = wcap.read()
    if cap is None:
        wcap.release()
    if ret:
        _run(leaf_model, "leaf", wframe)
        _run(pest_model, "pest", wframe)
        print("[WARMUP] done (excluded from events)")
    else:
        print("[WARMUP] camera gave no frame - check camera before the real run")

    threading.Thread(target=monitor_worker, args=(csv_tel,), daemon=True).start()

    t_end = time.time() + SESSION_MIN * 60
    eid = 0
    try:
        while time.time() < t_end:
            if TRIGGER_MODE == "manual":
                print(f"[WAIT] Enter to inspect "
                      f"({(t_end - time.time()) / 60:.1f} min left, Ctrl+C to stop)")
                try:
                    input()
                except EOFError:
                    break
                if time.time() >= t_end:
                    break
            else:
                time.sleep(min(random.uniform(TRIG_MIN, TRIG_MAX),
                               max(0.0, t_end - time.time())))
                if time.time() >= t_end:
                    break
            eid += 1
            row = inspect(cap, leaf_model, pest_model, eid)
            if row:
                with open(csv_evt, 'a', newline='') as f:
                    csv.writer(f).writerow(row)
    except KeyboardInterrupt:
        print("\n[STOP] interrupted")
    finally:
        running = False
        time.sleep(LOG_INTERVAL * 2)
        if cap:
            cap.release()

    print(f"\ntelemetry: {csv_tel}\nevents   : {csv_evt}")
    analyze(csv_tel, csv_evt)

if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "analyze":
        analyze(sys.argv[2], sys.argv[3])
    else:
        main()
