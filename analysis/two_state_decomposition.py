#!/usr/bin/env python3
"""
two_state_decomposition.py — pack-rail energy tables of the paper (Section 5.5).

    python analysis/two_state_decomposition.py [--repo .]

For each battery-powered trial:
  * integrates pack-side power (sum V*I*dt) and fits pack capacity C = E / dSoC
  * reports mean draw, endurance as fitted and on a common 60 W h pack, cycles/h,
    energy per cycle
  * splits the power trace into idle / inferring states by removing the discharge
    drift (quadratic) and cutting the residual at the quantile equal to the
    inference share of wall time
  * evaluates the energy floor at 1.25x / 2x / 4x / infinite runtime speed-up and
    the speed-up at which the Jetson would cross the Raspberry Pi 5's measured
    energy per cycle
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

COMMON_PACK_WH = 60.0


def per_inf(s):
    s = s.values
    s = s[s > 0]
    return s[np.r_[True, s[1:] != s[:-1]]]


def decompose(path, verbose=True):
    df = pd.read_csv(path)
    for c in ["Leaf_Lat_ms", "Pest_Lat_ms", "Batt_Voltage_mV", "Batt_Current_mA", "Batt_Percent", "Temp_C"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    t = pd.to_datetime(df["Timestamp"], format="%H:%M:%S.%f", errors="coerce").ffill()
    sec = (t - t.iloc[0]).dt.total_seconds().values.copy()
    for i in np.where(np.diff(sec) < 0)[0]:
        sec[i + 1:] += 86400
    df["sec"] = sec
    hours = (sec[-1] - sec[0]) / 3600

    d = df.dropna(subset=["Batt_Voltage_mV", "Batt_Current_mA"])
    V = d["Batt_Voltage_mV"].values / 1000
    I = np.abs(d["Batt_Current_mA"].values) / 1000
    P = V * I
    dt = np.clip(np.diff(d["sec"].values, prepend=d["sec"].values[0]), 0, 5)
    E_trial = (P * dt).sum() / 3600
    soc0 = df["Batt_Percent"].iloc[:20].median()
    soc1 = df["Batt_Percent"].iloc[-20:].median()
    C = E_trial / ((soc0 - soc1) / 100)

    lat_L = per_inf(df["Leaf_Lat_ms"]); lat_S = per_inf(df["Pest_Lat_ms"])
    cycles_h = len(lat_S) / hours
    period = 3600 / cycles_h
    t_comp = (lat_L.mean() + lat_S.mean()) / 1000
    t_idle = period - t_comp
    share = t_comp / period

    trend = np.polyval(np.polyfit(d["sec"].values, P, 2), d["sec"].values)
    res = P - trend
    q = np.quantile(res, 1 - share)
    P_idle = P[res <= q].mean(); P_inf = P[res > q].mean()

    e_cycle = P.mean() / cycles_h * 1000          # mWh
    floor = P_idle * t_idle / 3.6                  # mWh, instantaneous inference
    sens = []
    for f in (0.5, 2.0):
        q2 = np.quantile(res, 1 - min(share * f, 0.99))
        sens.append(P[res <= q2].mean() * t_idle / 3.6)

    r = dict(name=Path(path).stem, hours=hours, P_mean=P.mean(), V_mean=V.mean(), I_mean=I.mean(),
             E_trial=E_trial, soc=(soc0, soc1), C=C, end_fit=C / P.mean(), end_common=COMMON_PACK_WH / P.mean(),
             cycles_h=cycles_h, e_cycle=e_cycle, period=period, t_compute=t_comp, t_idle=t_idle, share=share,
             P_idle=P_idle, P_infer=P_inf, idle_share_of_mean=P_idle / P.mean(), floor=floor, floor_sens=tuple(sens),
             res_gt_05=(res > 0.5).mean(), T_mean=df["Temp_C"].mean(), T_max=df["Temp_C"].max())
    if verbose:
        print(f"\n=== {r['name']} ({hours:.2f} h) ===")
        print(f"mean draw {r['P_mean']:.2f} W  ({r['I_mean']:.2f} A @ {r['V_mean']:.2f} V)")
        print(f"trial energy {E_trial:.1f} Wh, SoC {soc0:.0f} -> {soc1:.0f} %, fitted capacity {C:.1f} Wh")
        print(f"endurance as fitted {r['end_fit']:.2f} h, on {COMMON_PACK_WH:.0f} Wh {r['end_common']:.2f} h")
        print(f"cycles/h {cycles_h:.0f}, energy per cycle {e_cycle:.2f} mWh")
        print(f"period {period:.3f} s = compute {t_comp*1000:.0f} ms + non-inferring {t_idle:.3f} s; inference share {100*share:.1f}%")
        print(f"P_idle {P_idle:.2f} W, P_infer {P_inf:.2f} W, idle floor = {100*r['idle_share_of_mean']:.1f}% of mean draw")
        print(f"residual > +0.5 W in {100*r['res_gt_05']:.1f}% of samples")
        for k in (1.25, 2, 4):
            print(f"  {k}x runtime speed-up -> {(P_idle*t_idle + P_inf*t_comp/k)/3.6:.2f} mWh/cycle")
        print(f"  instantaneous inference -> {floor:.2f} mWh/cycle (split sensitivity {sens[0]:.2f}-{sens[1]:.2f})")
        print(f"  ceiling on runtime optimisation: {100*(1-floor/e_cycle):.1f}% of energy per cycle")
        print(f"die temperature mean {r['T_mean']:.1f} C, max {r['T_max']:.1f} C")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    a = ap.parse_args()
    rp = decompose(Path(a.repo) / "raspberry-pi" / "data" / "raspberry_battery.csv")
    jt = decompose(Path(a.repo) / "jetson" / "data" / "jetson_battery.csv")
    target = rp["e_cycle"] * 3.6
    cs = (target - jt["P_idle"] * jt["t_idle"]) / jt["P_infer"]
    print(f"\nJetson crosses the Raspberry Pi 5's {rp['e_cycle']:.2f} mWh/cycle at a runtime speed-up of "
          f"{jt['t_compute']/cs:.2f}x" if cs > 0 else "\nJetson cannot reach the Raspberry Pi 5's energy per cycle")
    print(f"power ratio {jt['P_mean']/rp['P_mean']:.2f}x, capacity ratio {rp['C']/jt['C']:.2f}x, "
          f"endurance ratio as fitted {rp['end_fit']/jt['end_fit']:.2f}x, on common pack {rp['end_common']/jt['end_common']:.2f}x")


if __name__ == "__main__":
    main()
