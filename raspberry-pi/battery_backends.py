"""
battery_backends.py — drop-in replacement for get_battery_data() in main.py.

Select the backend with an environment variable:

    BATT_BACKEND=waveshare_ups_hat_e # Waveshare UPS HAT (E) on Raspberry Pi 5: BQ4050 fuel gauge via MCU @0x2D
    BATT_BACKEND=waveshare_ina219    # Waveshare UPS Power Module (C) on Jetson Orin: INA219 @0x41,
                                     #   SoC = vendor linear voltage map (NOT a fuel gauge)
    BATT_BACKEND=sbs                 # SMBus smart battery at 0x0b (V, I, SoC)
    BATT_BACKEND=bq27xxx             # TI BQ27441 / BQ2750x at 0x55 (V, I, SoC)
    BATT_BACKEND=max17048            # MAX17048/9 at 0x36 (V, SoC) — no current
    BATT_BACKEND=max17048+ina219     # MAX17048 for SoC + INA219 for V, I
    BATT_BACKEND=ina219              # INA219 only (V, I; SoC estimated from V — NOT what the paper used)

    BATT_I2C_BUS=1                   # bus number (RPi: 1; Jetson: often 1, 7 or 8)
    BATT_INA_ADDR=0x40               # override INA219 address if needed
    BATT_INA_SHUNT_OHM=0.1           # shunt resistor on the INA219 board (check the vendor schematic)

Returned tuple matches main.py: (volt_mV, curr_mA, pct, state).
Sign convention: current is NEGATIVE when discharging, matching the released CSVs.

Validate before trusting it: run for 60 s on pack power and compare against the
first rows of raspberry_battery.csv / jetson_battery.csv (RPi ~16.5 V, -250..-550 mA,
SoC 99 %; Jetson ~12.0 V, -0.9..-1.1 A, SoC 82 %).
"""
import os

_BUS = None


def _bus():
    global _BUS
    if _BUS is None:
        import smbus2
        _BUS = smbus2.SMBus(int(os.environ.get("BATT_I2C_BUS", "1")))
    return _BUS


def _state(curr_mA):
    if curr_mA < -100:
        return "discharging"
    if curr_mA > 100:
        return "charging"
    return "idle"


def _s16(v):
    return v - 65536 if v > 32767 else v


# ---------------------------------------------------------------- SBS (0x0b)
def _read_sbs():
    b = _bus(); a = 0x0b
    volt = b.read_word_data(a, 0x09)                 # mV
    curr = _s16(b.read_word_data(a, 0x0a))           # mA, negative = discharge
    pct = b.read_word_data(a, 0x0d)                  # RelativeStateOfCharge, %
    return float(volt), float(curr), float(pct), _state(curr)


# ---------------------------------------------------------------- BQ27xxx (0x55)
def _read_bq27xxx():
    b = _bus(); a = 0x55
    volt = b.read_word_data(a, 0x04)                 # mV
    curr = _s16(b.read_word_data(a, 0x10))           # AverageCurrent, mA (BQ27441 map)
    pct = b.read_word_data(a, 0x1c)                  # StateOfCharge, %
    return float(volt), float(curr), float(pct), _state(curr)


# ---------------------------------------------------------------- MAX17048 (0x36)
def _read_max17048():
    b = _bus(); a = 0x36
    raw = b.read_word_data(a, 0x02)                  # VCELL
    raw = ((raw & 0xff) << 8) | (raw >> 8)           # byte-swap (big-endian register)
    volt = raw * 78.125 / 1000                       # µV/LSB -> mV
    raw = b.read_word_data(a, 0x04)                  # SOC
    raw = ((raw & 0xff) << 8) | (raw >> 8)
    pct = raw / 256.0
    return float(volt), float(pct)


# ---------------------------------------------------------------- INA219 (0x40..0x45)
def _read_ina219():
    b = _bus(); a = int(os.environ.get("BATT_INA_ADDR", "0x40"), 16)
    shunt_ohm = float(os.environ.get("BATT_INA_SHUNT_OHM", "0.1"))
    raw = b.read_word_data(a, 0x02)
    raw = ((raw & 0xff) << 8) | (raw >> 8)
    volt = (raw >> 3) * 4.0                          # mV
    raw = b.read_word_data(a, 0x01)
    raw = _s16(((raw & 0xff) << 8) | (raw >> 8))
    shunt_mV = raw * 0.01
    curr = shunt_mV / shunt_ohm                      # mA; sign depends on wiring
    return float(volt), float(curr)


# ---------------------------------------------------------------- Waveshare UPS Power Module (C) — Jetson Orin
# INA219 with Waveshare's 32V/2A calibration; "percentage" is the vendor's linear
# voltage map (9.0 V -> 0 %, 12.6 V -> 100 % for a 3S pack). This reproduces the
# Batt_* columns of jetson_battery.csv exactly. It is NOT a fuel gauge.
_WS_CAL_DONE = False


def _read_waveshare_ina219():
    global _WS_CAL_DONE
    b = _bus(); a = int(os.environ.get("BATT_INA_ADDR", "0x41"), 16)   # Waveshare default for this module
    if not _WS_CAL_DONE:
        # config: 32 V bus range, ±320 mV shunt (gain /8), 12-bit 32-sample averaging, continuous
        cfg = 0x2000 | 0x1800 | 0x0780 | 0x0078 | 0x07
        b.write_word_data(a, 0x00, ((cfg & 0xff) << 8) | (cfg >> 8))
        cal = 4096                                   # current_LSB = 0.1 mA, shunt 0.1 ohm
        b.write_word_data(a, 0x05, ((cal & 0xff) << 8) | (cal >> 8))
        _WS_CAL_DONE = True
    raw = b.read_word_data(a, 0x02); raw = ((raw & 0xff) << 8) | (raw >> 8)
    volt = (raw >> 3) * 4.0                          # mV
    raw = b.read_word_data(a, 0x04); raw = _s16(((raw & 0xff) << 8) | (raw >> 8))
    curr = raw * 0.1                                 # mA, negative = discharging (Waveshare convention)
    cells = int(os.environ.get("BATT_CELLS", "3"))
    v_empty, v_full = 3.0 * cells * 1000, 4.2 * cells * 1000
    pct = max(0.0, min(100.0, (volt - v_empty) / (v_full - v_empty) * 100))
    return float(volt), float(curr), float(int(pct)), _state(curr)


# ---------------------------------------------------------------- Waveshare UPS HAT (E) — Raspberry Pi 5
# MCU at 0x2D exposing the BQ4050 fuel gauge. True state of charge from the gauge.
# Register map: https://www.waveshare.com/wiki/UPS_HAT_(E)_Register
def _read_waveshare_ups_hat_e():
    b = _bus(); a = int(os.environ.get("BATT_HAT_ADDR", "0x2d"), 16)
    def u16(reg):
        return b.read_byte_data(a, reg) | (b.read_byte_data(a, reg + 1) << 8)
    volt = u16(0x20)                                 # mV
    curr = _s16(u16(0x22))                           # mA, negative = discharging
    pct = u16(0x24)                                  # %
    chg = b.read_byte_data(a, 0x02)
    if chg & 0x80:
        state = "charging"
    elif curr < -100:
        state = "discharging"
    else:
        state = "idle"
    return float(volt), float(curr), float(pct), state


# ---------------------------------------------------------------- dispatcher
def get_battery_data():
    """Returns (volt_mV, curr_mA, pct, state). Falls back to (0, 0, 0, 'N/A') on any error."""
    backend = os.environ.get("BATT_BACKEND", "").lower()
    try:
        if backend == "waveshare_ups_hat_e":
            return _read_waveshare_ups_hat_e()
        if backend == "waveshare_ina219":
            return _read_waveshare_ina219()
        if backend == "sbs":
            return _read_sbs()
        if backend == "bq27xxx":
            return _read_bq27xxx()
        if backend == "max17048":
            v, p = _read_max17048()
            return v, 0.0, p, "N/A"
        if backend == "max17048+ina219":
            _, p = _read_max17048()
            v, i = _read_ina219()
            return v, i, p, _state(i)
        if backend == "ina219":
            v, i = _read_ina219()
            # voltage-estimated SoC: placeholder only, not a fuel gauge
            cells = int(os.environ.get("BATT_CELLS", "3"))
            p = max(0.0, min(100.0, (v / cells - 3300) / (4200 - 3300) * 100))
            return v, i, p, _state(i)
        return 0.0, 0.0, 0.0, "N/A"
    except Exception:
        return 0.0, 0.0, 0.0, "N/A"


if __name__ == "__main__":
    import time
    for _ in range(10):
        print(get_battery_data()); time.sleep(1)
