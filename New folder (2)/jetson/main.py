#!/usr/bin/env python3
# =========================================================
# Durian AI – DUAL MODEL (Leaf + Pest) ONNX
# UNIFIED CROSS-PLATFORM SCRIPT
# Jetson Orin Nano Super & Raspberry Pi 5
#
# Configurable scheduling: staggered / parallel / sequential
# Duty-cycled or continuous operation
# Unified 17-column telemetry (GPU/Battery columns; RPi uses NA)
# =========================================================

import cv2
import time
import sys
import threading
import numpy as np
import psutil
import csv
import os
import gc
import traceback
import requests
import urllib3
from datetime import datetime
from ultralytics import YOLO

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ================= PLATFORM DETECTION =================
# Auto-detect via /etc/os-release or environment hint
PLATFORM = os.environ.get("DURIAN_PLATFORM", "").lower()
if not PLATFORM:
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("ID="):
                    osid = line.split("=")[1].strip().lower()
                    if "jetson" in osid or "l4t" in osid:
                        PLATFORM = "jetson"
                    elif "raspberry" in osid or "raspian" in osid:
                        PLATFORM = "rpi5"
                    break
    except Exception:
        pass

if not PLATFORM:
    # Fallback: check for Jetson-specific paths
    if os.path.exists("/sys/devices/platform/17000000.gpu"):
        PLATFORM = "jetson"
    else:
        PLATFORM = "rpi5"

IS_JETSON = PLATFORM == "jetson"
print(f"[PLATFORM] Detected: {'Jetson Orin Nano Super' if IS_JETSON else 'Raspberry Pi 5'}")

try:
    from picamera2 import Picamera2
    USE_PICAMERA = True
except ImportError:
    USE_PICAMERA = False

HAS_DISPLAY = bool(os.environ.get('DISPLAY'))

# ================= CAMERA CONFIGURATION =================
# Both platforms default to USB webcam with unified settings
USE_USB_CAMERA = os.environ.get("USE_USB_CAMERA", "True").lower() == "true"
USB_CAMERA_INDEX = int(os.environ.get("USB_CAMERA_INDEX", "0"))
USB_CAMERA_RESOLUTION = (1280, 720)  # MJPG @ 1280x720, universal
USB_CAMERA_FPS = 30

# Jetson-specific: CSI camera via Argus (legacy, fallback only)
CSI_SENSOR_ID = 0
def gstreamer_pipeline(width=1280, height=720, framerate=30):
    return (
        f"nvarguscamerasrc sensor-id={CSI_SENSOR_ID} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"framerate={framerate}/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink"
    )

# ================= CONFIG =================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LEAF_MODEL_PATH = os.environ.get("LEAF_MODEL_PATH", os.path.join(_SCRIPT_DIR, "yolov11s.onnx"))
PEST_MODEL_PATH = os.environ.get("PEST_MODEL_PATH", os.path.join(_SCRIPT_DIR, "yolov11n.onnx"))

# ---- SCHEDULING MODE ----
SCHEDULE_MODE      = os.environ.get("SCHEDULE_MODE", "sequential")
CONF_THRESH        = 0.35
INFERENCE_SIZE     = 640
MAX_BOX_AREA_RATIO = 0.5
MAX_TEMP_LIMIT     = 82.0

CYCLE_ACTIVE_SEC   = 180
CYCLE_SLEEP_SEC    = 45
DUTY_CYCLE_ENABLED = os.environ.get("DUTY_CYCLE_ENABLED", "True").lower() == "true"
LOG_INTERVAL       = 0.5

# Inference intervals (seconds)
LEAF_INTERVAL      = 0.8
PEST_INTERVAL      = 1.2
STAGGER_DELAY      = 0.4

# ---- TELEGRAM ----
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")
TELEGRAM_COOLDOWN  = 30.0

SCREEN_W, SCREEN_H = 1024, 600
_dc_tag = "dutycycle" if DUTY_CYCLE_ENABLED else "nodutycycle"

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(_SCRIPT_DIR, "output"))
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_FILENAME = os.path.join(
    OUTPUT_DIR,
    f"dual_{SCHEDULE_MODE}_{_dc_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)

# ================= CLASS MAPS =================
LEAF_MERGE_MAP = {
    "algal_leave":  "Necrotic",
    "anthracnose":  "Necrotic",
    "early_blight": "Early Blight",
    "leaf_rot":     "Leaf Rot",
    "pink_disease": "Pink Disease",
    "root_disease": "Root Disease",
    "phomopsis":    "Phomopsis",
}

PEST_COLORS = {
    "Leafhopper_damage": (0,   165, 255),
    "Mealybugs":         (255, 0,   255),
    "Psyllid":           (255, 255, 0  ),
    "Red_Spider":        (0,   0,   255),
    "Scale_insect":      (255, 128, 0  ),
    "Stem_Borer":        (128, 0,   255),
    "Weevil":            (0,   255, 128),
}

LEAF_COLOR    = (0, 255, 0)
DEFAULT_COLOR = (200, 200, 200)

# ================= GLOBALS =================
current_frame   = None
leaf_detections = []
pest_detections = []
lock            = threading.Lock()
running         = True

perf_data = {
    "fps": 0.0, "leaf_lat": 0.0, "pest_lat": 0.0,
    "cpu": 0.0, "ram": 0.0, "temp": 0.0, "freq": 0.0,
    "gpu_pct": 0.0, "gpu_mhz": 0.0,
    "batt_volt": 0.0, "batt_curr": 0.0, "batt_pct": 0.0, "batt_state": "N/A"
}

# ================= HELPERS: THERMAL =================
def get_cpu_temp():
    if IS_JETSON:
        THERMAL_ZONE = 0  # /sys/class/thermal/thermal_zone0
        try:
            with open(f"/sys/class/thermal/thermal_zone{THERMAL_ZONE}/temp") as f:
                return float(f.read()) / 1000.0
        except Exception:
            return 0.0
    else:  # RPi5
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

# ================= HELPERS: GPU (Jetson only) =================
def get_gpu_load():
    if not IS_JETSON:
        return 0.0, 0.0  # (gpu_pct, gpu_mhz)
    try:
        # Orin: /sys/devices/platform/17000000.gpu/load (0–1000, divide by 10 for %)
        with open("/sys/devices/platform/17000000.gpu/load") as f:
            load_raw = int(f.read().strip())
            gpu_pct = load_raw / 10.0
    except Exception:
        gpu_pct = 0.0
    try:
        # Orin GPU freq: /sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/cur_freq (in Hz)
        with open("/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/cur_freq") as f:
            gpu_hz = int(f.read().strip())
            gpu_mhz = gpu_hz / 1e6
    except Exception:
        gpu_mhz = 0.0
    return gpu_pct, gpu_mhz

# ================= HELPERS: BATTERY (Jetson + INA219, not RPi5) =================
def get_battery_data():
    """
    Returns (batt_volt_mV, batt_curr_mA, batt_pct, batt_state).
    On Jetson with I2C INA219 module: reads from /sys/bus/i2c/devices/...
    On RPi5: no battery module (returns 0/0/0/NA).
    """
    if not IS_JETSON:
        return 0.0, 0.0, 0.0, "N/A"
    
    try:
        import smbus2
        bus = smbus2.SMBus(1)
        addr = 0x40  # INA219 default address
        
        # Voltage register (0x02): raw_mv = (reading << 3) / 1e6
        volt_reg = bus.read_word_data(addr, 0x02)
        batt_volt = (volt_reg >> 3) * 4 / 1000  # in mV
        
        # Current register (0x01): mA
        curr_reg = bus.read_word_data(addr, 0x01)
        batt_curr = curr_reg / 1000  # in mA (may be negative if discharging)
        
        # Estimate % from voltage (12V nominal 11–12V range)
        batt_pct = max(0, min(100, (batt_volt - 11.0) / (12.0 - 11.0) * 100))
        
        # Discharge state from current sign
        batt_state = "discharging" if batt_curr < -100 else ("charging" if batt_curr > 100 else "idle")
        
        bus.close()
        return batt_volt, batt_curr, batt_pct, batt_state
    except Exception:
        return 0.0, 0.0, 0.0, "N/A"

# ================= STARTUP CHECK =================
def startup_check():
    passed = True
    print()
    print("=" * 60)
    print(f"  Durian AI – Dual ONNX  |  {PLATFORM.upper()} Startup")
    print(f"  Schedule mode: {SCHEDULE_MODE.upper()}")
    print(f"  Duty cycle: {'ON (' + str(CYCLE_ACTIVE_SEC) + 's active / ' + str(CYCLE_SLEEP_SEC) + 's sleep)' if DUTY_CYCLE_ENABLED else 'OFF (continuous run)'}")
    print("=" * 60)

    for label, path in [("Leaf ONNX", LEAF_MODEL_PATH),
                         ("Pest ONNX", PEST_MODEL_PATH)]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"  [OK] {label:<20} {size_mb:.1f} MB")
        else:
            print(f"  [FAIL] {label} NOT FOUND: {path}")
            passed = False

    # Camera check
    if USE_USB_CAMERA:
        try:
            cap = cv2.VideoCapture(USB_CAMERA_INDEX, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, USB_CAMERA_RESOLUTION[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, USB_CAMERA_RESOLUTION[1])
            ret, f = cap.read()
            cap.release()
            if ret:
                print(f"  [OK] USB Camera         {f.shape}")
            else:
                print(f"  [FAIL] USB Camera: no frame")
                passed = False
        except Exception as e:
            print(f"  [FAIL] USB Camera: {e}")
            passed = False
    elif USE_PICAMERA and not IS_JETSON:
        try:
            pc = Picamera2()
            pc.configure(pc.create_preview_configuration(main={"size":(640,640),"format":"RGB888"}))
            pc.start(); time.sleep(0.5)
            f = pc.capture_array(); pc.stop(); pc.close()
            print(f"  [OK] Picamera2 (CSI)    {f.shape}")
        except Exception as e:
            print(f"  [FAIL] Picamera2: {e}")
            passed = False
    else:
        try:
            cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
            ret, f = cap.read(); cap.release()
            if ret:
                print(f"  [OK] CSI Camera (Argus) {f.shape}")
            else:
                print(f"  [FAIL] CSI: no frame")
                passed = False
        except Exception as e:
            print(f"  [FAIL] CSI: {e}")
            passed = False

    temp = get_cpu_temp()
    ram  = psutil.virtual_memory()
    cpu  = psutil.cpu_percent(interval=1)
    tw = " ⚠️ HIGH" if temp > 70 else ""
    print(f"  [OK] CPU {cpu:.0f}%  RAM {ram.available/1024**2:.0f}MB  Temp {temp:.1f}C{tw}")

    try:
        test_dir = os.path.join(OUTPUT_DIR, ".test")
        os.makedirs(test_dir, exist_ok=True)
        tf = os.path.join(test_dir, ".marker")
        open(tf, 'w').write("ok"); os.remove(tf)
        print(f"  [OK] Log directory writable")
    except Exception as e:
        print(f"  [FAIL] Log dir: {e}")
        passed = False

    if HAS_DISPLAY:
        print(f"  [OK] Display: {os.environ.get('DISPLAY')}")
    else:
        print(f"  [--] Headless — running via SSH")

    if not TELEGRAM_BOT_TOKEN:
        print("  [--] Telegram alerting DISABLED")

    print("-" * 60)
    if passed:
        print("  ✅  All checks passed — starting in 3 seconds...")
        print("=" * 60)
        time.sleep(3)
    else:
        print("  ❌  Checks FAILED")
        print("=" * 60)
        sys.exit(1)
    print()

# ================= LOAD MODELS =================
startup_check()

print("Loading Leaf model (yolov11s.onnx)...")
leaf_model = YOLO(LEAF_MODEL_PATH, task='detect')
leaf_model(np.zeros((INFERENCE_SIZE, INFERENCE_SIZE, 3), dtype=np.uint8), verbose=False)
print(f"  Leaf classes: {leaf_model.names}")

print("Loading Pest model (yolov11n.onnx)...")
pest_model = YOLO(PEST_MODEL_PATH, task='detect')
pest_model(np.zeros((INFERENCE_SIZE, INFERENCE_SIZE, 3), dtype=np.uint8), verbose=False)
print(f"  Pest classes: {pest_model.names}")
print("Both models loaded.\n")

# ================= INFER =================
def run_leaf_inference(frame_rgb):
    t0 = time.time()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    results = leaf_model(frame_bgr, imgsz=INFERENCE_SIZE, conf=CONF_THRESH,
                         iou=0.45, agnostic_nms=True, verbose=False)
    latency = (time.time() - t0) * 1000
    frame_area = INFERENCE_SIZE * INFERENCE_SIZE
    dets = []
    if results[0].boxes:
        for box in results[0].boxes:
            conf   = float(box.conf[0])
            cls_id = int(box.cls[0])
            raw_name = leaf_model.names[cls_id].lower()
            disease  = LEAF_MERGE_MAP.get(raw_name, leaf_model.names[cls_id])
            bbox = box.xyxy[0].cpu().numpy().astype(int).tolist()
            x1, y1, x2, y2 = bbox
            if (x2-x1)*(y2-y1) > frame_area * MAX_BOX_AREA_RATIO:
                continue
            dets.append({"type":"leaf","disease":disease,"conf":conf,"bbox":bbox})
    return dets, latency

def run_pest_inference(frame_rgb):
    t0 = time.time()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    results = pest_model(frame_bgr, imgsz=INFERENCE_SIZE, conf=CONF_THRESH,
                         iou=0.45, agnostic_nms=True, verbose=False)
    latency = (time.time() - t0) * 1000
    frame_area = INFERENCE_SIZE * INFERENCE_SIZE
    dets = []
    if results[0].boxes:
        for box in results[0].boxes:
            conf   = float(box.conf[0])
            cls_id = int(box.cls[0])
            name   = pest_model.names[cls_id]
            bbox   = box.xyxy[0].cpu().numpy().astype(int).tolist()
            x1, y1, x2, y2 = bbox
            if (x2-x1)*(y2-y1) > frame_area * MAX_BOX_AREA_RATIO:
                continue
            dets.append({"type":"pest","disease":name,"conf":conf,"bbox":bbox})
    return dets, latency

# ================= THREADS =================
def leaf_worker():
    global current_frame, leaf_detections, perf_data
    while running:
        if current_frame is None:
            time.sleep(0.1); continue
        frame = current_frame.copy()
        try:
            dets, lat = run_leaf_inference(frame)
        except Exception:
            traceback.print_exc()
            dets, lat = [], 0.0
        perf_data["leaf_lat"] = lat
        with lock:
            leaf_detections = dets
        time.sleep(LEAF_INTERVAL)

def pest_worker():
    global current_frame, pest_detections, perf_data
    if SCHEDULE_MODE == "staggered":
        time.sleep(STAGGER_DELAY)
    while running:
        if current_frame is None:
            time.sleep(0.1); continue
        frame = current_frame.copy()
        try:
            dets, lat = run_pest_inference(frame)
        except Exception:
            traceback.print_exc()
            dets, lat = [], 0.0
        perf_data["pest_lat"] = lat
        with lock:
            pest_detections = dets
        time.sleep(PEST_INTERVAL)

def sequential_worker():
    global current_frame, leaf_detections, pest_detections, perf_data
    cycle_sleep = max(LEAF_INTERVAL, PEST_INTERVAL)
    while running:
        if current_frame is None:
            time.sleep(0.1); continue
        frame = current_frame.copy()
        try:
            l_dets, l_lat = run_leaf_inference(frame)
        except Exception:
            traceback.print_exc()
            l_dets, l_lat = [], 0.0
        perf_data["leaf_lat"] = l_lat
        with lock:
            leaf_detections = l_dets
        try:
            p_dets, p_lat = run_pest_inference(frame)
        except Exception:
            traceback.print_exc()
            p_dets, p_lat = [], 0.0
        perf_data["pest_lat"] = p_lat
        with lock:
            pest_detections = p_dets
        time.sleep(cycle_sleep)

def monitor_worker():
    global perf_data
    with open(CSV_FILENAME, 'w', newline='') as f:
        csv.writer(f).writerow([
            "Timestamp","Schedule_Mode","FPS","Leaf_Lat_ms","Pest_Lat_ms",
            "CPU_%","RAM_MB","Temp_C","Freq_MHz","GPU_%","GPU_MHz",
            "Batt_Voltage_mV","Batt_Current_mA","Batt_Percent","Batt_State",
            "Leaf_Detections","Pest_Detections"
        ])
    print(f"Logging: {CSV_FILENAME}")
    while running:
        perf_data["cpu"]  = psutil.cpu_percent(interval=None)
        perf_data["ram"]  = psutil.virtual_memory().used / 1024 / 1024
        perf_data["temp"] = get_cpu_temp()
        perf_data["freq"] = get_cpu_freq()
        perf_data["gpu_pct"], perf_data["gpu_mhz"] = get_gpu_load()
        perf_data["batt_volt"], perf_data["batt_curr"], perf_data["batt_pct"], perf_data["batt_state"] = get_battery_data()
        
        with lock:
            l_dets = [d["disease"] for d in leaf_detections]
            p_dets = [d["disease"] for d in pest_detections]
        with open(CSV_FILENAME, 'a', newline='') as f:
            csv.writer(f).writerow([
                datetime.now().strftime('%H:%M:%S.%f')[:-3],
                SCHEDULE_MODE,
                f"{perf_data['fps']:.1f}",
                f"{perf_data['leaf_lat']:.1f}",
                f"{perf_data['pest_lat']:.1f}",
                f"{perf_data['cpu']:.1f}",
                f"{perf_data['ram']:.1f}",
                f"{perf_data['temp']:.1f}",
                f"{perf_data['freq']:.0f}",
                f"{perf_data['gpu_pct']:.1f}" if IS_JETSON else "0.0",
                f"{perf_data['gpu_mhz']:.0f}" if IS_JETSON else "0.0",
                f"{perf_data['batt_volt']:.0f}" if IS_JETSON else "0",
                f"{perf_data['batt_curr']:.0f}" if IS_JETSON else "0",
                f"{perf_data['batt_pct']:.0f}" if IS_JETSON else "0",
                perf_data['batt_state'] if IS_JETSON else "N/A",
                "|".join(l_dets) if l_dets else "None",
                "|".join(p_dets) if p_dets else "None",
            ])
        time.sleep(LOG_INTERVAL)

def send_telegram(img_path, message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    def _send():
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        try:
            with open(img_path, 'rb') as photo:
                r = requests.post(url,
                    data={'chat_id': TELEGRAM_CHAT_ID, 'caption': message},
                    files={'photo': photo},
                    timeout=15.0)
            print("✅ Telegram sent!" if r.status_code==200 else f"❌ HTTP {r.status_code}")
        except Exception as e:
            print(f"❌ Telegram: {e}")
    threading.Thread(target=_send, daemon=True).start()

def draw_dashboard(img_bgr, leaf_dets, pest_dets):
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (10,10), (340,225), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, img_bgr, 0.4, 0, img_bgr)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_bgr, f"FPS: {perf_data['fps']:.1f}  [{SCHEDULE_MODE}]",
                (20,40), font, 0.6, (0,255,0), 2)
    cv2.putText(img_bgr, f"Leaf:{perf_data['leaf_lat']:.0f}ms Pest:{perf_data['pest_lat']:.0f}ms",
                (20,68), font, 0.55, (0,255,255), 1)
    cv2.putText(img_bgr, f"CPU:{perf_data['cpu']:.0f}% RAM:{perf_data['ram']:.0f}MB",
                (20,93), font, 0.55, (255,255,255), 1)
    temp = perf_data['temp']
    tc = (0,0,255) if temp > 80 else (255,255,255)
    cv2.putText(img_bgr, f"Temp:{temp:.1f}C {perf_data['freq']:.0f}MHz",
                (20,118), font, 0.55, tc, 1)
    l_str = "|".join([d["disease"] for d in leaf_dets]) if leaf_dets else "Clear"
    cv2.putText(img_bgr, f"Leaf:{l_str[:32]}",
                (20,150), font, 0.5, (0,255,0), 1)
    p_str = "|".join([d["disease"] for d in pest_dets]) if pest_dets else "Clear"
    cv2.putText(img_bgr, f"Pest:{p_str[:32]}",
                (20,175), font, 0.5, (0,165,255), 1)
    cv2.putText(img_bgr, f"YOLOv11s leaf  YOLOv11n pest (ONNX, {PLATFORM})",
                (20,215), font, 0.45, (180,180,180), 1)

# ================= MAIN =================
def main():
    global current_frame, running

    picam2 = cap = None
    
    if USE_USB_CAMERA:
        cap = cv2.VideoCapture(USB_CAMERA_INDEX, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, USB_CAMERA_RESOLUTION[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, USB_CAMERA_RESOLUTION[1])
        cap.set(cv2.CAP_PROP_FPS, USB_CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    elif USE_PICAMERA and not IS_JETSON:
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(
            main={"size":(640,640),"format":"RGB888"}))
        picam2.start()
        try:
            picam2.set_controls({"AfMode":2,"AfSpeed":1})
            print("Autofocus enabled")
        except Exception:
            pass
    else:
        cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    WIN_NAME = "Durian AI"
    if HAS_DISPLAY:
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Start workers
    worker_threads = [threading.Thread(target=monitor_worker, daemon=True)]
    if SCHEDULE_MODE in ("staggered", "parallel"):
        worker_threads += [
            threading.Thread(target=leaf_worker, daemon=True),
            threading.Thread(target=pest_worker, daemon=True),
        ]
    elif SCHEDULE_MODE == "sequential":
        worker_threads += [
            threading.Thread(target=sequential_worker, daemon=True),
        ]
    else:
        raise ValueError(f"Unknown SCHEDULE_MODE: {SCHEDULE_MODE}")

    for t in worker_threads:
        t.start()

    print(f"🟢 Running [{SCHEDULE_MODE.upper()}] (CONF={CONF_THRESH}, "
          f"Leaf every {LEAF_INTERVAL}s, Pest every {PEST_INTERVAL}s)")

    cycle_start        = time.time()
    is_active          = True
    fps_start          = time.time()
    fps_cnt            = 0
    frame_cnt          = 0
    last_telegram_time = 0.0

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(OUTPUT_DIR, "detections", f"run_{SCHEDULE_MODE}_{ts}", "images")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving to: {save_dir}")

    try:
        while True:
            now = time.time()

            if DUTY_CYCLE_ENABLED:
                elapsed = now - cycle_start
                if is_active:
                    if elapsed > CYCLE_ACTIVE_SEC:
                        print("Sleep..."); is_active = False; cycle_start = now; continue
                else:
                    if elapsed > CYCLE_SLEEP_SEC:
                        print("Active."); is_active = True; cycle_start = now; fps_start = time.time()
                    else:
                        time.sleep(0.5); continue

            temp = get_cpu_temp()
            if temp > MAX_TEMP_LIMIT:
                print(f"OVERHEAT {temp:.1f}C"); time.sleep(5); continue

            if frame_cnt % 100 == 0:
                gc.collect()

            if picam2:
                try:
                    frame_rgb = picam2.capture_array()
                except Exception:
                    continue
            else:
                ret, frame_bgr = cap.read()
                if not ret: break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            current_frame = frame_rgb

            fps_cnt += 1
            if fps_cnt >= 10:
                perf_data["fps"] = 10 / (time.time() - fps_start)
                fps_start = time.time(); fps_cnt = 0

            with lock:
                l_dets = list(leaf_detections)
                p_dets = list(pest_detections)

            all_dets = l_dets + p_dets
            vis_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            for d in l_dets:
                x1,y1,x2,y2 = d['bbox']
                cv2.rectangle(vis_bgr,(x1,y1),(x2,y2),LEAF_COLOR,2)
                cv2.putText(vis_bgr,f"{d['disease']} {d['conf']:.2f}",
                            (x1,max(y1-8,12)),cv2.FONT_HERSHEY_SIMPLEX,0.55,LEAF_COLOR,2)

            for d in p_dets:
                x1,y1,x2,y2 = d['bbox']
                color = PEST_COLORS.get(d['disease'],DEFAULT_COLOR)
                cv2.rectangle(vis_bgr,(x1,y1),(x2,y2),color,2)
                cv2.putText(vis_bgr,f"{d['disease']} {d['conf']:.2f}",
                            (x1,max(y1-8,12)),cv2.FONT_HERSHEY_SIMPLEX,0.55,color,2)

            draw_dashboard(vis_bgr, l_dets, p_dets)

            if HAS_DISPLAY:
                final = cv2.resize(vis_bgr,(SCREEN_W,SCREEN_H))
                cv2.imshow(WIN_NAME, final)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if all_dets and (now - last_telegram_time > TELEGRAM_COOLDOWN):
                img_path = os.path.join(save_dir, f"det_{frame_cnt:06d}.jpg")
                cv2.imwrite(img_path, vis_bgr)
                lines = [f"🍃 {d['disease']}: {d['conf']*100:.1f}%" for d in l_dets]
                lines += [f"🐛 {d['disease']}: {d['conf']*100:.1f}%" for d in p_dets]
                msg = (f"🚨 Durian Alert! [{SCHEDULE_MODE}]\n\n" + "\n".join(lines) +
                       f"\n\n🌡️{temp:.1f}°C ⏱️{perf_data['fps']:.1f}FPS")
                send_telegram(img_path, msg)
                last_telegram_time = now

            frame_cnt += 1

    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception:
        traceback.print_exc()
    finally:
        running = False
        if HAS_DISPLAY:
            cv2.destroyAllWindows()
        try:
            if picam2: picam2.stop()
            elif cap: cap.release()
        except Exception:
            pass
        print(f"CSV: {CSV_FILENAME}")

if __name__ == "__main__":
    main()
