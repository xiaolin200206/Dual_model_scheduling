# Errata

Defects found in this repository after publication of the data, recorded rather
than silently corrected. Raw telemetry is never edited; corrections are made in
documentation and in the code that reads it.

Each entry states the defect, its cause, whether any reported quantity is
affected, and what was done.

---

## E1 — `Batt_State` is mislabelled in the three-hour Raspberry Pi 5 battery trial

**File:** `raspberry-pi/data/raspberry_battery.csv`

**Defect:** the `Batt_State` column reads `idle` on all 21,605 samples, while
`Batt_Current_mA` on the same rows runs from −685 to −257 mA. The node was
discharging for the entire trial.

**Cause:** the runtime of that campaign derived the charge-state label from bit
0x80 of vendor status register 0x02. That register is **unpopulated on this
board** — it reads 0x00 while the pack discharges — so the bit is always clear
and the fall-through label was written instead. The later per-policy campaign
derives the label from the sign of pack current, and
`raspberry-pi/data/policy_battery/*_1h.csv` label the same condition correctly
as `discharging`.

**Affected results:** none. No quantity in the paper reads `Batt_State`. Power,
energy, capacity, endurance, latency and throughput are computed from
`Batt_Voltage_mV`, `Batt_Current_mA`, `Batt_Percent` and the two latency
columns. `analysis/verify_paper_numbers.py` recomputes every reported value
without touching this column.

**Action:** raw file left unmodified. Warning added to
`raspberry-pi/battery_backends.py` and to the data table in `README.md`.
Recorded as a limitation in the paper (Section 7).

**Reproduce:**
```python
import pandas as pd
d = pd.read_csv("raspberry-pi/data/raspberry_battery.csv")
print(d.Batt_State.value_counts())          # idle: 21605
print(d.Batt_Current_mA.min(), d.Batt_Current_mA.max())   # -685 -257
```

---

## E2 — The fuel-gauge part number was asserted without evidence

**Files:** `README.md`, `raspberry-pi/README.md`, `raspberry-pi/main.py`,
`raspberry-pi/battery_backends.py`, and the Jetson copies of the last two.

**Defect:** earlier revisions described the Raspberry Pi 5 gauge as a "TI
BQ4050 fuel gauge". No evidence in this repository supports that part number.
The code reads only the register map Waveshare publishes for the UPS HAT (E)
(0x20 pack voltage, 0x22 pack current, 0x24 state of charge); nothing in the
data or the code identifies the underlying device, and the vendor publishes
neither a part number nor a schematic.

**Affected results:** none. Every claim about this instrument in the paper rests
on measured behaviour — that state of charge tracks integrated current rather
than terminal voltage, and that the reported current cannot resolve the
inter-inference power state — not on the part number.

**Action:** all references to a part number removed. The gauge is now described
as a coulomb-counting gauge identified by behaviour, which is what the evidence
supports.

---

## E3 — CPU frequency governor was not recorded

**Scope:** all Raspberry Pi 5 trials.

**Defect:** the governor was left at the platform default and not logged as a
field. Logged core frequency (`Freq_MHz`) ranges over 1500–2400 MHz within every
trial, which establishes that it was *not* pinned to `performance`, but the
governor's identity is inferred rather than recorded.

**Affected results:** frequency is a free variable of these measurements rather
than a controlled one. Differences between configurations therefore include any
difference in the frequency the governor selected. This is stated in the paper
(Sections 4.1 and 7) and is not corrected here.

**Action:** documented. Pinning and verifying the governor before each trial is
listed as future work.

**Reproduce:**
```python
import pandas as pd
d = pd.read_csv("raspberry-pi/data/raspberry_battery.csv")
print(sorted(d.Freq_MHz.dropna().unique()))   # [1500, 1600, 1700, 1800, 1900, 2400]
```
