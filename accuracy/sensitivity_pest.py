# -*- coding: utf-8 -*-
"""
sensitivity_pest.py — 定位 Paper5 实测 (mAP50=0.451) 与 Paper4 已报
                      (mAP50=0.495) 之间差异的来源

Paper4 送审中不可改, 因此必须弄清两个数字的关系, 否则同一模型两篇论文
两个精度。本脚本把所有可能造成差异的变量逐一扫一遍:

  A. NMS IoU        : 0.5 / 0.6 (ultralytics 默认) / 0.7 (Paper5 用的)
  B. 权重格式       : .onnx (部署) vs .pt (训练检查点, 若存在)
  C. conf 阈值      : 0.001 (mAP 标准) vs 0.25 vs 0.35 (部署阈值)
  D. 输入尺寸       : 640 vs 1280 (若 Paper4 用过其它尺寸)
  E. 评估划分       : valid / test / train (若存在多个划分)
  F. rect / augment : ultralytics 的两个会影响数值的开关

输出一张对照表, 并自动标出最接近 0.495 的组合。

用法:
    pip install ultralytics
    python sensitivity_pest.py
"""

import csv
import itertools
import json
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================ CONFIG ============================

BASE      = Path(r"C:\Users\Lim Ding Shan\Desktop\Durian project and paper")
MODEL_DIR = BASE / "Fifth paper" / "model"
PEST_ROOT = BASE / "paepr4 data" / "pest dataset" / "pest_dataset_merged"

# 要对照的目标值 (Paper4 已报)
TARGET_MAP50 = 0.495
TARGET_NOTE  = "Paper4 reported mAP@0.5 = 0.495 (excl. sparse classes: 0.49)"

# 权重候选: ONNX 必有; .pt 若在别处, 补进来
WEIGHTS = [MODEL_DIR / "yolov11n_pest.onnx"]
for cand in [
    MODEL_DIR / "yolov11n_pest.pt",
    PEST_ROOT / "weights" / "best.pt",
    BASE / "paepr4 data" / "pest dataset" / "best.pt",
]:
    if cand.exists():
        WEIGHTS.append(cand)

# 扫描网格
GRID = {
    "iou":    [0.5, 0.6, 0.7],
    "conf":   [0.001, 0.25],
    "imgsz":  [640],            # 需要时加 1280
    "split":  ["valid"],        # 自动补上实际存在的其它划分
    "rect":   [False],          # 需要时加 True
}

OUT_CSV = MODEL_DIR / "sensitivity_pest.csv"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ================================================================


def read_onnx_names(p: Path):
    import onnxruntime as ort
    s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    n = s.get_modelmeta().custom_metadata_map.get("names")
    try:
        n = json.loads(n)
    except Exception:
        import ast
        n = ast.literal_eval(n)
    return {int(k): v for k, v in n.items()}


def pt_names(p: Path):
    from ultralytics import YOLO
    return YOLO(str(p)).names


def dirs_of(root: Path):
    return (root / "images" if (root / "images").is_dir() else root,
            root / "labels" if (root / "labels").is_dir() else root)


def available_splits():
    out = []
    for s in ("valid", "val", "test", "train"):
        d = PEST_ROOT / s
        if d.is_dir():
            img, _ = dirs_of(d)
            n = sum(1 for p in img.rglob("*") if p.suffix.lower() in IMG_EXT)
            if n:
                out.append((s, n))
    return out


def make_yaml(split: str, names: dict) -> Path:
    root = PEST_ROOT / split
    img_dir, _ = dirs_of(root)
    rel = "images" if img_dir != root else "."
    yp = Path(tempfile.gettempdir()) / f"sens_pest_{split}.yaml"
    lines = [f"path: {root.as_posix()}", f"train: {rel}", f"val: {rel}", "names:"]
    lines += [f"  {i}: {names[i]}" for i in sorted(names)]
    yp.write_text("\n".join(lines), encoding="utf-8")
    return yp


def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("pip install ultralytics")

    splits = available_splits()
    print("可用划分:", ", ".join(f"{s} ({n} 张)" for s, n in splits))
    grid_splits = [s for s, _ in splits if s in GRID["split"]] or [splits[0][0]]
    # 若存在 test, 也纳入 (Paper4 可能用的是 test)
    for s, _ in splits:
        if s == "test" and s not in grid_splits:
            grid_splits.append(s)

    print("权重候选:")
    for w in WEIGHTS:
        print(f"  {w.name}")
    if len(WEIGHTS) == 1:
        print("  (未找到 .pt — 若 Paper4 评估的是训练检查点, 把它的路径加进 WEIGHTS)")

    combos = list(itertools.product(WEIGHTS, grid_splits, GRID["imgsz"],
                                    GRID["conf"], GRID["iou"], GRID["rect"]))
    print(f"\n共 {len(combos)} 组配置, 开始扫描 ...\n")

    rows = [["weights", "split", "imgsz", "conf", "nms_iou", "rect",
             "P", "R", "mAP50", "mAP50-95", "delta_vs_target"]]
    results = []

    for w, split, imgsz, conf, iou, rect in combos:
        try:
            names = read_onnx_names(w) if w.suffix == ".onnx" else pt_names(w)
            m = YOLO(str(w), task="detect").val(
                data=str(make_yaml(split, names)), split="val",
                imgsz=imgsz, conf=conf, iou=iou, rect=rect,
                device=0, plots=False, verbose=False)
            b = m.box
            d = b.map50 - TARGET_MAP50
            rows.append([w.name, split, imgsz, conf, iou, rect,
                         f"{b.mp:.4f}", f"{b.mr:.4f}", f"{b.map50:.4f}",
                         f"{b.map:.4f}", f"{d:+.4f}"])
            results.append((abs(d), w.name, split, imgsz, conf, iou, rect,
                            b.map50, b.map))
            print(f"  {w.name:<22s} {split:<6s} sz={imgsz} conf={conf:<5} "
                  f"iou={iou}  mAP50={b.map50:.4f}  ({d:+.4f})")
        except Exception as e:
            print(f"  {w.name} {split} iou={iou} — 失败: {e}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    print("\n" + "=" * 74)
    print(f"目标: {TARGET_NOTE}")
    print("=" * 74)
    results.sort()
    print("\n最接近的 5 组配置:")
    for d, wn, sp, sz, cf, iou, rect, m50, m in results[:5]:
        print(f"  Δ={d:.4f}   mAP50={m50:.4f}  mAP50-95={m:.4f}   "
              f"{wn}  {sp}  imgsz={sz} conf={cf} iou={iou} rect={rect}")

    if results and results[0][0] < 0.01:
        print("\n=> 找到几乎吻合的配置: 差异来自评估设置而非模型或数据。")
        print("   Paper5 中注明所用设置即可, 两篇不冲突。")
    elif results:
        best = results[0]
        print(f"\n=> 所有配置都无法复现 {TARGET_MAP50}, 最接近的差 {best[0]:.4f}。")
        print("   差异可能来自: (a) Paper4 用的是另一份评估划分或更早的权重;")
        print("   (b) Paper4 的数字来自训练日志的 best epoch 而非独立验证;")
        print("   (c) 权重在 Paper4 之后重训过 (对照 model_report 的导出日期)。")
        print("   建议翻 Paper4 当时的 runs/ 目录或评估脚本确认。")

    print(f"\n已保存: {OUT_CSV}")


if __name__ == "__main__":
    main()
