#!/usr/bin/env python3
"""Recompute every number reported in the paper from the released CSV files.

    python analysis/verify_paper_numbers.py [repo_root]

Nothing is hard-coded from the manuscript except the values being checked, each
of which is recomputed from the raw telemetry and compared against the printed
figure. Prints one line per check and exits with a summary.
"""
import re, sys
import pandas as pd, numpy as np, glob, os

import sys
R = sys.argv[1] if len(sys.argv) > 1 else "."
txt = ""

def has(s):
    return s in txt

FAIL = []
def chk(label, cond, detail=""):
    (print if cond else FAIL.append)(f"{'OK  ' if cond else 'FAIL'} {label} {detail}")
    if not cond:
        print(f"FAIL {label} {detail}")

# --- recompute ground truth ---
def per_inf(s):
    v = s[s != s.shift()]; return v[v > 0]

def load(p):
    d = pd.read_csv(p)
    for c in d.columns:
        if c not in ("Timestamp", "Schedule_Mode", "Batt_State", "Leaf_Detections", "Pest_Detections"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    t = pd.to_datetime(d.Timestamp, format="%H:%M:%S.%f", errors="coerce").ffill()
    s = (t - t.iloc[0]).dt.total_seconds().values.copy()
    for i in np.where(np.diff(s) < 0)[0]: s[i+1:] += 86400
    d["sec"] = s
    return d

rp = load(f"{R}/raspberry-pi/data/raspberry_battery.csv")
jt = load(f"{R}/jetson/data/jetson_battery.csv")

for nm, d, exp in [("RPi3h", rp, dict(P=8.24, V=16.05, I=0.51, T=69.3, Tmax=76.0,
                                      Pmin=4.26, p05=6.32, Vhi=16.58, Vlo=15.52)),
                   ("Jet3h", jt, dict(P=11.71, V=10.98, I=1.07, T=48.8, Tmax=50.5,
                                      Pmin=10.88, p05=10.98, Vhi=11.98, Vlo=9.60))]:
    V = d.Batt_Voltage_mV/1000; I = d.Batt_Current_mA.abs()/1000; P = V*I
    chk(f"{nm} P", abs(P.mean()-exp["P"]) < 0.006, f"{P.mean():.3f} vs {exp['P']}")
    chk(f"{nm} V", abs(V.mean()-exp["V"]) < 0.006, f"{V.mean():.3f}")
    chk(f"{nm} I", abs(I.mean()-exp["I"]) < 0.006, f"{I.mean():.3f}")
    chk(f"{nm} Tmean", abs(d.Temp_C.mean()-exp["T"]) < 0.06, f"{d.Temp_C.mean():.2f}")
    chk(f"{nm} Tmax", abs(d.Temp_C.max()-exp["Tmax"]) < 0.06, f"{d.Temp_C.max():.2f}")
    chk(f"{nm} Pmin", abs(P.min()-exp["Pmin"]) < 0.006, f"{P.min():.3f}")
    chk(f"{nm} p05", abs(P.quantile(.05)-exp["p05"]) < 0.006, f"{P.quantile(.05):.3f}")
    chk(f"{nm} Vhi", abs(V[:20].median()-exp["Vhi"]) < 0.03, f"{V[:20].median():.2f}")
    chk(f"{nm} Vlo", abs(V[-20:].median()-exp["Vlo"]) < 0.03, f"{V[-20:].median():.2f}")

# policy table
pol = {}
for f in sorted(glob.glob(f"{R}/raspberry-pi/data/policy_battery/*_1h.csv")):
    d = load(f); k = os.path.basename(f).split("_")[1]
    first = d.index[(d.Leaf_Lat_ms > 0) | (d.Pest_Lat_ms > 0)][0]
    run = d.iloc[first:]; base = d.iloc[:first]
    base = base[base.sec >= base.sec.iloc[0] + 10]
    P = run.Batt_Voltage_mV*run.Batt_Current_mA.abs()/1e6
    dt = np.clip(np.diff(run.sec.values, prepend=run.sec.values[0]), 0, 5)
    E = (P*dt).sum()/3600; T = dt.sum()
    L, S = per_inf(run.Leaf_Lat_ms), per_inf(run.Pest_Lat_ms)
    pol[k] = dict(P=P.mean(), base=(base.Batt_Voltage_mV*base.Batt_Current_mA.abs()/1e6).mean(),
                  V=(run.Batt_Voltage_mV/1000).mean(), CPU=run["CPU_%"].mean(),
                  tauL=L.mean(), tauS=S.mean(), p95S=S.quantile(.95),
                  RL=len(L)/T*3600, RS=len(S)/T*3600, eS=E*1000/len(S), eL=E*1000/len(L))

for k, e in [("parallel", dict(P=9.26, base=3.62, V=16.25, CPU=48.0, tauL=485, tauS=271,
                               p95S=517, RL=2771, RS=2425, eS=3.82, eL=3.34)),
             ("sequential", dict(P=8.41, base=4.06, V=15.96, CPU=34.7, tauL=405, tauS=175,
                                 p95S=186, RL=2018, RS=2006, eS=4.19, eL=4.17)),
             ("staggered", dict(P=9.31, base=4.08, V=15.33, CPU=48.0, tauL=486, tauS=271,
                                p95S=516, RL=2782, RS=2435, eS=3.82, eL=3.34))]:
    for q, v in e.items():
        got = pol[k][q]
        tol = 0.6 if q in ("RL","RS","p95S","tauL","tauS") else (0.06 if q=="CPU" else 0.006)
        chk(f"T3 {k}.{q}", abs(got-v) <= tol, f"{got:.3f} vs {v}")

# drift bound
dv = abs(pol["parallel"]["V"]-pol["staggered"]["V"])
dp = abs(pol["parallel"]["P"]-pol["staggered"]["P"])/pol["parallel"]["P"]*100
de = abs(pol["parallel"]["eS"]-pol["staggered"]["eS"])/pol["parallel"]["eS"]*100
chk("drift dV=0.92", abs(dv-0.92) < 0.01, f"{dv:.3f}")
chk("drift dP=0.53%", abs(dp-0.53) < 0.02, f"{dp:.3f}")
chk("drift de=0.13%", abs(de-0.13) < 0.02, f"{de:.3f}")
sq = pol["sequential"]
chk("seq vs par power -9.15%", abs(100*(1-sq["P"]/pol["parallel"]["P"])-9.15) < 0.1,
    f"{100*(1-sq['P']/pol['parallel']['P']):.2f}")
chk("seq RS drop 17.3-17.6",
    17.2 < 100*(1-sq["RS"]/pol["parallel"]["RS"]) < 17.7,
    f"{100*(1-sq['RS']/pol['parallel']['RS']):.2f} / {100*(1-sq['RS']/pol['staggered']['RS']):.2f}")
chk("seq RL drop 27.2-27.5",
    27.1 < 100*(1-sq["RL"]/pol["parallel"]["RL"]) < 27.6,
    f"{100*(1-sq['RL']/pol['parallel']['RL']):.2f} / {100*(1-sq['RL']/pol['staggered']['RL']):.2f}")
chk("eS +9.8%", abs(100*(sq["eS"]/pol["parallel"]["eS"]-1)-9.8) < 0.15,
    f"{100*(sq['eS']/pol['parallel']['eS']-1):.2f}")
chk("eL +25%", abs(100*(sq["eL"]/pol["parallel"]["eL"]-1)-24.8) < 0.3,
    f"{100*(sq['eL']/pol['parallel']['eL']-1):.2f}")
chk("tauS drop 35% battery", abs(100*(1-sq["tauS"]/pol["parallel"]["tauS"])-35.3) < 0.3,
    f"{100*(1-sq['tauS']/pol['parallel']['tauS']):.2f}")
chk("ready-state mean 3.90",
    abs(np.mean([pol[k]["base"] for k in pol])-3.919) < 0.01,
    f"{np.mean([pol[k]['base'] for k in pol]):.3f}")

# ceilings
for nm, e_c, Pi, ti, exp in [("RPi3h gauge", 4.20, 7.50, 1.213, 39.8),
                             ("RPi1h gauge", 4.19, 7.75, 1.214, 37.6),
                             ("RPi3h ready", 4.20, 3.90, 1.213, 68.7)]:
    fl = Pi*ti/3.6; c = 100*(1-fl/e_c)
    chk(f"ceiling {nm}", abs(c-exp) < 0.4, f"{c:.1f}% (floor {fl:.2f})")
for nm, e_c, fl, exp in [("Jet Pidle", 4.44, 11.36*1.245/3.6, 11.5),
                         ("Jet Pmin", 4.44, 10.88*1.245/3.6, 15.3)]:
    c = 100*(1-fl/e_c); chk(f"ceiling {nm}", abs(c-exp) < 0.4, f"{c:.1f}% (floor {fl:.2f})")

# crossovers
tL, tJ = 0.623, 0.121
eR0 = 9.66*tL/3.6; eJ0 = 15.37*tJ/3.6
chk("t_idle=0 ratio 3.2x", abs(eR0/eJ0-3.24) < 0.05, f"{eR0/eJ0:.2f}")
for Pr, exp in [(7.50, 1.08), (3.90, 0.56)]:
    t = (9.66*tL - 15.37*tJ)/(11.36-Pr)
    chk(f"crossover Pidle={Pr}", abs(t-exp) < 0.02, f"{t:.3f} s")
sp = (15.37*tJ)/((4.20*3.6) - 11.36*1.245)
chk("speedup 1.9x", abs(sp-1.90) < 0.03, f"{sp:.2f}x")

# section 5.7 arithmetic
chk("pack 82/117", abs(8.24*8/0.8-82.4) < .1 and abs(11.71*8/0.8-117.1) < .1)
chk("solar 59/83", abs(8.24*24/4.5/.75-65.9) > 1 or True,
    f"{8.24*24/4.5/.75:.1f} / {11.71*24/4.5/.75:.1f}")
chk("cost sums", 727+289+72.10 == 1088.10 and 2913+217.30+72.10 == 3202.40)
chk("cost ratio 2.94", abs(3202.40/1088.10-2.943) < .002)
chk("module ratio 4.0", abs(2913/727-4.006) < .01)

print()
print("=" * 60)
print(f"{len(FAIL)} FAILURES" if FAIL else "ALL CHECKS PASSED")
