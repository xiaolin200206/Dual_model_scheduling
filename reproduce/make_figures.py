#!/usr/bin/env python3
"""
make_figures.py — regenerate Figs. 1–4 of the paper from the released telemetry.

    python reproduce/make_figures.py --repo . --out figures/

Fig. 1  per-inference latency distributions, twelve configurations
Fig. 2  achieved throughput against mean latency
Fig. 3  battery-powered trials: SoC, temperature, current, energy vs cycles
Fig. 4  energy per cycle vs inference-runtime speed-up (two-state model)

The telemetry carries the last inference latency forward between inferences;
per_inf() de-duplicates consecutive values to recover one sample per inference.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman"],
                     "font.size": 9, "axes.linewidth": 0.8, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.05})
C_RPI, C_JET = "#4a6fa5", "#c4703c"
MODES = ["sequential", "parallel", "staggered"]
DUTY = ["duty", "nonduty"]
PLATFORMS = (("RPi5", "raspberry-pi", "raspberry"), ("Jetson", "jetson", "jetson"))
# CSV column -> paper name
COL = {"Pest_Lat_ms": "Small model $M_S$", "Leaf_Lat_ms": "Large model $M_L$"}


def load(path):
    df = pd.read_csv(path)
    for c in df.columns:
        if c not in ("Timestamp", "Schedule_Mode", "Batt_State", "Leaf_Detections", "Pest_Detections"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    t = pd.to_datetime(df["Timestamp"], format="%H:%M:%S.%f", errors="coerce").ffill()
    sec = (t - t.iloc[0]).dt.total_seconds().values.copy()
    for i in np.where(np.diff(sec) < 0)[0]:      # midnight wrap
        sec[i + 1:] += 86400
    df["sec"] = sec
    return df


def per_inf(s):
    s = s.values
    s = s[s > 0]
    return s[np.r_[True, s[1:] != s[:-1]]]


def load_runs(repo):
    runs = {}
    for plat, sub, pre in PLATFORMS:
        for m in MODES:
            for d in DUTY:
                p = Path(repo) / sub / "data" / f"{pre}_{m}_{d}.csv"
                if p.exists():
                    runs[(plat, m, d)] = load(p)
                else:
                    print("  missing:", p)
    return runs


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=300)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    print(name, "ok")


def fig1_latency(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    for ax, key in zip(axes, ["Pest_Lat_ms", "Leaf_Lat_ms"]):
        data, labels, colors = [], [], []
        for plat in ("RPi5", "Jetson"):
            for d in DUTY:
                for m in MODES:
                    df = runs.get((plat, m, d))
                    if df is None:
                        continue
                    data.append(per_inf(df[key]))
                    labels.append(f"{m[:4]}\n{'duty' if d == 'duty' else 'cont'}")
                    colors.append(C_RPI if plat == "RPi5" else C_JET)
        bp = ax.boxplot(data, whis=(5, 95), showfliers=False, widths=0.62, patch_artist=True,
                        medianprops=dict(color="0.15", lw=1.1), showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="0.2", markersize=3.2))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.55); patch.set_edgecolor("0.25"); patch.set_linewidth(0.8)
        for el in ("whiskers", "caps"):
            for ln in bp[el]:
                ln.set_color("0.35"); ln.set_linewidth(0.8)
        ax.set_yscale("log")
        ax.set_xticklabels(labels, fontsize=6.4)
        ax.set_ylabel("Per-inference latency (ms, log scale)", fontsize=8.5)
        ax.set_title(COL[key], fontsize=9.5, pad=6)
        ax.grid(axis="y", ls=":", lw=0.5, color="0.75", alpha=0.7)
        ax.set_axisbelow(True)
        ax.axvline(6.5, color="0.5", lw=0.9, ls="--")
        ax.text(3.5, ax.get_ylim()[1] * 0.72, "Raspberry Pi 5", ha="center", fontsize=7.5, color=C_RPI, fontweight="bold")
        ax.text(9.5, ax.get_ylim()[1] * 0.72, "Jetson Orin Nano Super", ha="center", fontsize=7.5, color=C_JET, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "Fig3_latency_distributions")


def fig2_throughput(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    mk = {"sequential": "o", "parallel": "s", "staggered": "^"}
    for (plat, m, d), df in runs.items():
        c = C_RPI if plat == "RPi5" else C_JET
        h = (df["sec"].iloc[-1] - df["sec"].iloc[0]) / 3600
        for ax, key in zip(axes, ["Pest_Lat_ms", "Leaf_Lat_ms"]):
            x = per_inf(df[key])
            ax.scatter(x.mean(), len(x) / h, marker=mk[m], s=34,
                       facecolor=c if d == "duty" else "white", edgecolor=c, lw=1.1, zorder=3)
    for ax, key in zip(axes, ["Pest_Lat_ms", "Leaf_Lat_ms"]):
        ax.set_xscale("log")
        ax.set_xlabel("Mean per-inference latency (ms, log)", fontsize=8)
        ax.set_ylabel("Achieved throughput (inferences h$^{-1}$)", fontsize=8)
        ax.set_title(COL[key], fontsize=9)
        ax.grid(ls=":", lw=0.5, color="0.8"); ax.set_axisbelow(True); ax.tick_params(labelsize=7)
    handles = [Line2D([], [], marker="o", ls="", color="0.3", label="sequential"),
               Line2D([], [], marker="s", ls="", color="0.3", label="parallel"),
               Line2D([], [], marker="^", ls="", color="0.3", label="staggered"),
               Line2D([], [], marker="o", ls="", markerfacecolor="0.3", markeredgecolor="0.3", label="duty-cycled"),
               Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="0.3", label="continuous"),
               Line2D([], [], marker="s", ls="", color=C_RPI, label="Raspberry Pi 5"),
               Line2D([], [], marker="s", ls="", color=C_JET, label="Jetson Orin Nano Super")]
    axes[0].legend(handles=handles, fontsize=6, frameon=False, loc="lower left", ncol=2)
    fig.tight_layout()
    save(fig, out, "Fig4_throughput_latency")


def fig3_battery(repo, out):
    rp = load(Path(repo) / "raspberry-pi" / "data" / "raspberry_battery.csv")
    jt = load(Path(repo) / "jetson" / "data" / "jetson_battery.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    for df, lab, c in ((rp, "Raspberry Pi 5", C_RPI), (jt, "Jetson Orin Nano Super", C_JET)):
        m = df["sec"] / 60
        axes[0, 0].plot(m, df["Batt_Percent"], lw=1.2, color=c, label=lab)
        axes[0, 1].plot(m, df["Temp_C"].rolling(120, min_periods=1).mean(), lw=1.2, color=c)
        axes[1, 0].plot(m, df["Batt_Current_mA"].abs().rolling(120, min_periods=1).mean(), lw=1.2, color=c)
        P = (df["Batt_Voltage_mV"] / 1000 * df["Batt_Current_mA"].abs() / 1000).fillna(0)
        dt = np.clip(np.diff(df["sec"], prepend=df["sec"].iloc[0]), 0, 5)
        E = (P * dt).cumsum() / 3600
        p = df["Pest_Lat_ms"].values
        cycles = np.cumsum(np.r_[False, p[1:] != p[:-1]] & (p > 0))
        axes[1, 1].plot(cycles, E, lw=1.2, color=c)
    axes[0, 0].set_ylabel("Pack state of charge (%)", fontsize=8); axes[0, 0].set_title("(a) Pack discharge", fontsize=9)
    axes[0, 0].legend(fontsize=6.5, frameon=False)
    axes[0, 1].axhline(82, color="0.35", ls="--", lw=0.9)
    axes[0, 1].text(5, 83, "82 °C software cut-off", fontsize=6.5, color="0.35")
    axes[0, 1].set_ylabel("Die temperature (°C)", fontsize=8); axes[0, 1].set_title("(b) Thermal trajectory (60-sample mean)", fontsize=9)
    axes[0, 1].set_ylim(40, 90)
    axes[1, 0].set_ylabel("Discharge current (mA)", fontsize=8); axes[1, 0].set_title("(c) Pack-side current (60-sample mean)", fontsize=9)
    axes[1, 1].set_xlabel("Cycles completed (one inference of each model)", fontsize=8)
    axes[1, 1].set_ylabel("Cumulative energy (W h)", fontsize=8); axes[1, 1].set_title("(d) Energy against work done", fontsize=9)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xlabel("Elapsed time (min)", fontsize=8)
    for ax in axes.flat:
        ax.grid(ls=":", lw=0.5, color="0.8"); ax.set_axisbelow(True); ax.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, out, "Fig1_battery_trials")


def fig4_energy_floor(decomp, out):
    """decomp: {platform: (P_idle, P_infer, t_idle_s, t_compute_s)} from two_state_decomposition.py"""
    k = np.logspace(0, np.log10(32), 200)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    curves = {}
    for lab, c in (("Raspberry Pi 5", C_RPI), ("Jetson Orin Nano Super", C_JET)):
        idle, inf, over, comp = decomp[lab]
        e = (idle * over + inf * comp / k) / 3.6
        curves[lab] = e
        ax.plot(k, e, color=c, lw=1.4, label=lab)
        ax.axhline(idle * over / 3.6, color=c, ls=":", lw=0.9)
    # crossover: Jetson curve falls below RPi measured (k=1) value
    rpi0 = curves["Raspberry Pi 5"][0]
    idx = np.argmax(curves["Jetson Orin Nano Super"] <= rpi0)
    if idx > 0:
        kx = k[idx]
        ax.axvline(kx, color="0.4", ls="--", lw=0.8)
        ax.text(kx * 1.03, rpi0 + 0.25, f"Jetson crosses\nRPi 5 at {kx:.1f}×", fontsize=6.5, color="0.3")
    ax.set_xscale("log")
    ax.set_xlabel("Inference-runtime speed-up (×)", fontsize=8)
    ax.set_ylabel("Energy per cycle (mW h)", fontsize=8)
    ax.set_xticks([1, 2, 4, 8, 16, 32]); ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    ax.legend(fontsize=6.5, frameon=False); ax.grid(ls=":", lw=0.5, color="0.8"); ax.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, out, "Fig2_energy_floor")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(a.repo)
    if runs:
        fig1_latency(runs, out)
        fig2_throughput(runs, out)
    fig3_battery(a.repo, out)
    import sys
    sys.path.insert(0, str(Path(a.repo) / "analysis"))
    from two_state_decomposition import decompose
    decomp = {}
    for lab, sub, pre in (("Raspberry Pi 5", "raspberry-pi", "raspberry"), ("Jetson Orin Nano Super", "jetson", "jetson")):
        r = decompose(Path(a.repo) / sub / "data" / f"{pre}_battery.csv", verbose=False)
        decomp[lab] = (r["P_idle"], r["P_infer"], r["t_idle"], r["t_compute"])
    fig4_energy_floor(decomp, out)
    print("\nfigures written to", out.resolve())


if __name__ == "__main__":
    main()
