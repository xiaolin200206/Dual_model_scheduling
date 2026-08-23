#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py — 从遥测数据重绘 Paper5 的全部图件 (300 dpi, 投稿规格)

Fig. 1  研究定位图 (与修订后的定位一致: Paper2 审计 + Paper4 单平台 -> 本文跨平台采购)
Fig. 2  十二配置的 pest / leaf 延迟分布 (箱线图)
Fig. 3  户外试验: 电量轨迹、温度轨迹

用法:
    pip install matplotlib pandas numpy
    python make_figures.py --repo <Dual_model_scheduling 根目录> --out figures/
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

DPI = 300
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "0.2",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

MODES = ["sequential", "parallel", "staggered"]
DUTY = ["duty", "nonduty"]
C_RPI, C_JET = "#4a6fa5", "#c4703c"



# ------------------------------------------------------------------ loading
def load_runs(repo):
    runs = {}
    for plat, sub, pre in (("RPi5", "raspberry-pi", "raspberry"),
                           ("Jetson", "jetson", "jetson")):
        for mode in MODES:
            for d in DUTY:
                p = Path(repo) / sub / "data" / f"{pre}_{mode}_{d}.csv"
                if not p.exists():
                    print("  missing:", p); continue
                df = pd.read_csv(p)
                for c in ("Leaf_Lat_ms", "Pest_Lat_ms", "Temp_C", "CPU_%"):
                    if c in df:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                runs[(plat, mode, d)] = df
    return runs


def clean(s):
    """去掉启动期的 0 值样本"""
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s[s > 0]


# ------------------------------------------------------------------ Fig. 2
def fig2_latency(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=False)
    for ax, model, key in zip(axes, ["Pest model", "Leaf model"],
                              ["Pest_Lat_ms", "Leaf_Lat_ms"]):
        data, labels, colors = [], [], []
        for plat in ("RPi5", "Jetson"):
            for d in DUTY:
                for mode in MODES:
                    df = runs.get((plat, mode, d))
                    if df is None or key not in df:
                        continue
                    data.append(clean(df[key]).values)
                    labels.append(f"{mode[:4]}\n{'duty' if d=='duty' else 'cont'}")
                    colors.append(C_RPI if plat == "RPi5" else C_JET)
        bp = ax.boxplot(data, whis=(5, 95), showfliers=False, widths=0.62,
                        patch_artist=True, medianprops=dict(color="0.15", lw=1.1),
                        showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="white",
                                       markeredgecolor="0.2", markersize=3.2))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.55)
            patch.set_edgecolor("0.25"); patch.set_linewidth(0.8)
        for el in ("whiskers", "caps"):
            for ln in bp[el]:
                ln.set_color("0.35"); ln.set_linewidth(0.8)
        ax.set_yscale("log")
        ax.set_xticklabels(labels, fontsize=6.4)
        ax.set_ylabel("Inference latency (ms, log scale)", fontsize=8.5)
        ax.set_title(model, fontsize=9.5, pad=6)
        ax.grid(axis="y", ls=":", lw=0.5, color="0.75", alpha=0.7)
        ax.set_axisbelow(True)
        ax.axvline(6.5, color="0.5", lw=0.9, ls="--")
        ax.text(3.5, ax.get_ylim()[1] * 0.72, "Raspberry Pi 5",
                ha="center", fontsize=7.5, color=C_RPI, fontweight="bold")
        ax.text(9.5, ax.get_ylim()[1] * 0.72, "Jetson Orin Nano Super",
                ha="center", fontsize=7.5, color=C_JET, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "Fig2_latency_distributions.png", dpi=DPI)
    fig.savefig(out / "Fig2_latency_distributions.pdf")
    plt.close(fig)
    print("Fig. 2 ok")


# ------------------------------------------------------------------ Fig. 4
def to_minutes(ts):
    t = pd.to_datetime(ts, format="%H:%M:%S.%f", errors="coerce")
    t = t.ffill()
    return (t - t.iloc[0]).dt.total_seconds() / 60.0


def fig4_ambient(repo, out):
    rp = pd.read_csv(Path(repo) / "raspberry-pi" / "data" / "raspberry_battery.csv")
    jt = pd.read_csv(Path(repo) / "jetson" / "data" / "jetson_battery.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))

    # (a) SoC
    ax = axes[0]
    for df, lab, c in ((rp, "Raspberry Pi 5", C_RPI), (jt, "Jetson Orin Nano Super", C_JET)):
        m = to_minutes(df["Timestamp"])
        pct = pd.to_numeric(df["Batt_Percent"], errors="coerce")
        ax.plot(m, pct, lw=1.2, color=c, label=lab)
    ax.set_xlabel("Elapsed time (min)", fontsize=8)
    ax.set_ylabel("Pack state of charge (%)", fontsize=8)
    ax.set_title("(a) Pack discharge", fontsize=9)
    ax.legend(fontsize=6.4, frameon=False, loc="lower left")

    # (b) temperature
    ax = axes[1]
    for df, lab, c in ((rp, "Raspberry Pi 5", C_RPI), (jt, "Jetson Orin Nano Super", C_JET)):
        m = to_minutes(df["Timestamp"])
        t = pd.to_numeric(df["Temp_C"], errors="coerce").rolling(120, min_periods=1).mean()
        ax.plot(m, t, lw=1.2, color=c, label=lab)
    ax.axhline(82, color="0.35", ls="--", lw=0.9)
    ax.text(ax.get_xlim()[1] * 0.98, 82.8, "82 °C cut-off", ha="right",
            fontsize=6.4, color="0.35")
    ax.set_xlabel("Elapsed time (min)", fontsize=8)
    ax.set_ylabel("Die temperature (°C)", fontsize=8)
    ax.set_title("(b) Thermal trajectory", fontsize=9)

    # (c) discharge current
    ax = axes[2]
    for df, lab, c in ((rp, "Raspberry Pi 5", C_RPI), (jt, "Jetson Orin Nano Super", C_JET)):
        m = to_minutes(df["Timestamp"])
        i = pd.to_numeric(df["Batt_Current_mA"], errors="coerce").abs()
        i = i.rolling(120, min_periods=1).mean()
        ax.plot(m, i, lw=1.2, color=c, label=lab)
    ax.set_xlabel("Elapsed time (min)", fontsize=8)
    ax.set_ylabel("Discharge current (mA)", fontsize=8)
    ax.set_title("(c) Pack-side current", fontsize=9)

    for ax in axes:
        ax.grid(ls=":", lw=0.5, color="0.8", alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "Fig4_ambient_trials.png", dpi=DPI)
    fig.savefig(out / "Fig4_ambient_trials.pdf")
    plt.close(fig)
    print("Fig. 4 ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fig1_perclass(out)
    runs = load_runs(a.repo)
    if runs:
        fig2_latency(runs, out)
        fig3_ablation(runs, out)
    fig4_ambient(a.repo, out)
    fig5_selection(out)
    print("\n输出目录:", out.resolve())




# ------------------------------------------------------------------ Fig. 3
def fig3_ablation(runs, out):
    """十二配置消融：温度 / CPU / leaf / pest 四指标"""
    metrics = [("Pest_Lat_ms", "Mean pest latency (ms)", True),
               ("Leaf_Lat_ms", "Mean leaf latency (ms)", True),
               ("Temp_C", "Mean temperature (\u00b0C)", False),
               ("CPU_%", "Mean CPU utilisation (%)", False)]
    series = [("RPi5", "duty", C_RPI, 0.9, "Pi 5, duty"),
              ("RPi5", "nonduty", C_RPI, 0.42, "Pi 5, continuous"),
              ("Jetson", "duty", C_JET, 0.9, "Jetson, duty"),
              ("Jetson", "nonduty", C_JET, 0.42, "Jetson, continuous")]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.9))
    w = 0.19
    xs = np.arange(len(MODES))
    for ax, (key, lab, log) in zip(axes, metrics):
        for idx, (plat, d, c, al, leg) in enumerate(series):
            vals = []
            for mode in MODES:
                df = runs.get((plat, mode, d))
                vals.append(clean(df[key]).mean()
                            if df is not None and key in df else np.nan)
            ax.bar(xs + (idx - 1.5) * w, vals, w * 0.9, color=c, alpha=al,
                   edgecolor="0.25", linewidth=0.4,
                   label=leg if ax is axes[0] else None)
        if log:
            ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(["sequential", "parallel", "staggered"], fontsize=6.6, rotation=30, ha="right")
        ax.set_ylabel(lab, fontsize=7.8)
        ax.grid(axis="y", ls=":", lw=0.5, color="0.8", alpha=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7)
        ax.set_xlim(-0.6, len(MODES) - 0.4)
    axes[0].legend(fontsize=5.8, frameon=False, loc="lower left",
                   bbox_to_anchor=(-0.02, 1.0), ncol=2, columnspacing=0.8,
                   handlelength=1.1, handletextpad=0.4)
    fig.tight_layout()
    fig.savefig(out / "Fig3_ablation_summary.png", dpi=DPI)
    fig.savefig(out / "Fig3_ablation_summary.pdf")
    plt.close(fig)
    print("Fig. 3 ok")


# ------------------------------------------------------------------ Fig. 1
LEAF_AP = [("algal", 254, 0.476), ("leaf_rot", 54, 0.748), ("Phomopsis", 84, 0.650),
           ("pink", 20, 0.439), ("root", 59, 0.474)]
PEST_AP = [("leafhopper damage", 39, 0.409), ("Psyllid", 539, 0.366),
           ("Psyllid_damage", 418, 0.295), ("Scale_insect", 57, 0.252),
           ("Stem-borer", 33, 0.199), ("weevil", 4, 0.773),
           ("weevil_damage", 78, 0.864)]
MIN_INST = 15


def fig1_perclass(out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2),
                             gridspec_kw={"width_ratios": [1, 1.25]})
    for ax, data, title, agg in (
            (axes[0], LEAF_AP, "Leaf model (YOLOv11s)", 0.557),
            (axes[1], PEST_AP, "Pest model (YOLOv11n)", 0.451)):
        data = sorted(data, key=lambda r: r[1])
        names = [d[0] for d in data]
        inst = [d[1] for d in data]
        ap = [d[2] for d in data]
        cols = ["0.72" if n < MIN_INST else C_RPI for n in inst]
        y = np.arange(len(names))
        ax.barh(y, ap, 0.66, color=cols, edgecolor="0.25", linewidth=0.5, alpha=0.85)
        for i, (a, n) in enumerate(zip(ap, inst)):
            ax.text(a + 0.015, i, f"{a:.3f}  (n={n})", va="center", fontsize=6.8,
                    color="0.25")
        ax.axvline(agg, color="0.2", ls="--", lw=1.0, zorder=0)
        ax.annotate(f"aggregate {agg:.3f}", xy=(agg, len(names) - 0.42),
                    xytext=(agg + 0.30, len(names) - 0.15), fontsize=6.6,
                    color="0.25", ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="0.45"))
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7.5)
        ax.set_xlim(0, 1.18)
        ax.set_ylim(-0.6, len(names) + 0.15)
        ax.set_xlabel("AP@0.5 on the audited partition", fontsize=8)
        ax.set_title(title, fontsize=9, pad=5)
        ax.grid(axis="x", ls=":", lw=0.5, color="0.8", alpha=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7)
    fig.text(0.5, -0.06, "Classes ordered by validation support; grey bars carry fewer than "
             f"{MIN_INST} instances and are not interpreted as class-level performance.",
             ha="center", fontsize=7, color="0.35")
    fig.tight_layout()
    fig.savefig(out / "Fig1_per_class_ap.png", dpi=DPI)
    fig.savefig(out / "Fig1_per_class_ap.pdf")
    plt.close(fig)
    print("Fig. 1 ok")


# ------------------------------------------------------------------ Fig. 5
def fig5_selection(out):
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.set_xlim(0, 9.15); ax.set_ylim(0, 6.4); ax.axis("off")

    rows = [
        ("On-demand inspection,\nrecharged between sessions", "Raspberry Pi 5",
         "latency non-binding;\ncost dominates", C_RPI),
        ("Extended session\naway from power", "Raspberry Pi 5",
         "endurance 8.2 W vs 11.8 W;\n\u2248 two sessions vs one", C_RPI),
        ("Several operators\nor blocks in parallel", "Raspberry Pi 5",
         "2.94 units per Jetson unit\nfor equal capital", C_RPI),
        ("Hot enclosure\n(preliminary evidence)", "Jetson Orin Nano Super",
         "thermal margin: Pi 5 at 76.0 \u00b0C,\n6 \u00b0C below threshold", C_JET),
        ("Continuous scanning\nfrom a moving platform", "Jetson Orin Nano Super",
         "latency binds: 5.36\u00d7 faster\nat 2.94\u00d7 cost", C_JET),
    ]
    ax.text(1.55, 6.05, "Deployment condition", ha="center", fontsize=8.5,
            fontweight="bold", color="0.2")
    ax.text(4.8, 6.05, "Platform", ha="center", fontsize=8.5,
            fontweight="bold", color="0.2")
    ax.text(7.55, 6.05, "Binding quantity", ha="center", fontsize=8.5,
            fontweight="bold", color="0.2")
    ax.plot([0.1, 9.05], [5.82, 5.82], lw=0.9, color="0.4")

    for i, (cond, plat, why, c) in enumerate(rows):
        y = 5.15 - i * 1.02
        ax.add_patch(FancyBboxPatch((0.1, y - 0.38), 2.9, 0.78,
                                    boxstyle="round,pad=0.03,rounding_size=0.08",
                                    linewidth=0.8, edgecolor="0.6", facecolor="0.96"))
        ax.text(1.55, y, cond, ha="center", va="center", fontsize=7.4, color="0.2")
        ax.add_patch(FancyArrowPatch((3.08, y), (3.43, y), arrowstyle="-|>",
                                     mutation_scale=8, lw=0.9, color="0.45"))
        ax.add_patch(FancyBboxPatch((3.5, y - 0.34), 2.6, 0.7,
                                    boxstyle="round,pad=0.03,rounding_size=0.08",
                                    linewidth=1.1, edgecolor=c, facecolor=c, alpha=0.16))
        ax.text(4.8, y, plat, ha="center", va="center", fontsize=7.6,
                fontweight="bold", color=c)
        ax.text(6.3, y, why, ha="left", va="center", fontsize=7.1,
                style="italic", color="0.35")
        if i < len(rows) - 1:
            ax.plot([0.1, 9.05], [y - 0.51, y - 0.51], lw=0.4, color="0.85")

    ax.plot([0.1, 9.05], [0.32, 0.32], lw=0.9, color="0.4")
    ax.text(0.16, 0.02, "Independent of the choice:  sequential rather than concurrent "
            "scheduling (39.0\u201344.3% pest-latency reduction) and Docker containerisation, "
            "both at no measurable cost.", ha="left", va="center", fontsize=6.9, color="0.2")
    fig.savefig(out / "Fig5_selection_rule.png", dpi=DPI)
    fig.savefig(out / "Fig5_selection_rule.pdf")
    plt.close(fig)
    print("Fig. 5 ok")


if __name__ == "__main__":
    main()
