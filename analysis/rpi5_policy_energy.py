"""
Pack-rail energy analysis for the Raspberry Pi 5 per-policy battery trials (3 Sep 2026).
Reproduces Table 4 and the direct idle baseline of Section 5.2.

Each CSV row is one 0.5 s telemetry sample. The ideas, in order:
  power   = voltage x current                 (W, one number per row)
  energy  = sum(power x dt)                   (W h, the "odometer")
  P_idle  = mean power in the 60 s baseline   (models loaded, camera open, no inference)
  N_L,N_S = number of times the latency column changed value (= number of inferences)
  e_S     = energy / N_S                      (mW h per small-model inference = "per cycle")
"""
import glob, pathlib, numpy as np, pandas as pd

def analyse(path):
    df = pd.read_csv(path)
    df["t"]  = pd.to_datetime(df.Timestamp, format="%H:%M:%S.%f")
    df["dt"] = df.t.diff().dt.total_seconds().fillna(0.5).clip(0, 5)
    df["P"]  = df.Batt_Voltage_mV * df.Batt_Current_mA.abs() / 1e6        # watts

    first = df.index[(df.Leaf_Lat_ms > 0) | (df.Pest_Lat_ms > 0)][0]       # first inference row
    base  = df.iloc[:first]                                                # 60 s ready-state baseline
    base  = base[base.t >= base.t.iloc[0] + pd.Timedelta(seconds=10)]      # drop start-up transient
    run   = df.iloc[first:]

    def per_inference(col):                                                # de-duplicate carry-forward
        v = run[col][run[col] != run[col].shift()]
        v = v[v > 0]
        return v.mean(), v.quantile(0.95), len(v)

    tauL, p95L, NL = per_inference("Leaf_Lat_ms")
    tauS, p95S, NS = per_inference("Pest_Lat_ms")
    T    = run.dt.sum()
    E_Wh = (run.P * run.dt).sum() / 3600

    # capture-suspended vs capture-active phases (45 s in every 225 s)
    s  = (run.t - run.t.iloc[0]).dt.total_seconds()
    ph = s % 225
    active = run[(ph > 5) & (ph < 175)];  sleep = run[(ph > 185) & (ph < 220)]

    # residual two-state decomposition (paper Section 3.5)
    x = np.arange(len(run)); r = run.P.values - np.polyval(np.polyfit(x, run.P.values, 2), x)
    f = (NL * tauL + NS * tauS) / 1000 / T
    thr = np.quantile(r, 1 - min(f, 0.95))
    P_idle_resid, P_infer_resid = run.P.values[r <= thr].mean(), run.P.values[r > thr].mean()

    return dict(
        policy=pathlib.Path(path).name.split('_')[1], minutes=T/60,
        P_idle_baseline_W=base.P.mean(), P_mean_W=run.P.mean(), P_min_W=run.P.min(), P_p05_W=run.P.quantile(.05),
        P_idle_resid_W=P_idle_resid, P_infer_resid_W=P_infer_resid, inference_share=f,
        capture_cost_W=active.P.mean() - sleep.P.mean(),
        tauL_ms=tauL, tauS_ms=tauS, P95_S_ms=p95S, R_L_per_h=NL/T*3600, R_S_per_h=NS/T*3600,
        e_S_mWh=E_Wh*1000/NS, e_L_mWh=E_Wh*1000/NL, CPU_pct=run["CPU_%"].mean(),
        SoC=f"{df.Batt_Percent.iloc[0]:.0f}->{df.Batt_Percent.iloc[-1]:.0f}%",
    )

if __name__ == "__main__":
    rows = [analyse(f) for f in sorted(glob.glob("raspberry-pi/data/policy_battery/*_1h.csv"))]
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).set_index("policy").round(3).T)
