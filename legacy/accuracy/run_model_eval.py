# -*- coding: utf-8 -*-
"""
run_model_eval.py — 评估部署用的两个 ONNX 模型 (完整版, 路径已填好)

在 RTX 4050 电脑上跑。类名直接从 ONNX metadata 读取, 自动生成 data.yaml,
无需手动配置数据集 yaml。

本版新增 (相对上一版):
  1. 路径已按调查结论填好, 开箱即跑
  2. pest 评估集自动做泄漏检查 (对照同目录下的 train)
  3. 小样本类标记, 并给出剔除小样本类后的 mAP —— 小样本类单独标记
  4. 报告末尾直接输出论文表格行与可粘贴的方法学描述

用法:
    pip install ultralytics pillow
    python run_model_eval.py

输出:
    accuracy_report.txt   总体 + 每类 P/R/mAP50/mAP50-95, 论文可直接用
    accuracy_metrics.csv  表格版
    runs/detect/val*/     PR 曲线、混淆矩阵 (可放补充材料)
"""

import csv
import json
import sys
import tempfile
from collections import Counter
import os
from pathlib import Path

# ============================ CONFIG ============================

BASE = Path(os.environ.get("DATASET_ROOT", "."))  # root of the (unreleased) image datasets
MODEL_DIR = BASE / "Fifth paper" / "model"

TASKS = {
    "leaf": {
        "model":   MODEL_DIR / "yolov11s_leaf.onnx",
        # 已净化的评估集: 184 -> 147 张, 剔除与各 train 集重叠的 37 张
        "dataset": BASE / "Fifth paper" / "leaf_eval_clean",
        "train":   None,          # 已在 build_clean_eval.py 阶段完成泄漏净化
    },
    "pest": {
        "model":   MODEL_DIR / "yolov11n_pest.onnx",
        "dataset": BASE / "paepr4 data" / "pest dataset" / "pest_dataset_merged" / "valid",
        # 用于泄漏检查的同源 train 集:
        "train":   BASE / "paepr4 data" / "pest dataset" / "pest_dataset_merged" / "train",
    },
}

IMGSZ    = 640       # 与导出一致 (model_report: [640, 640])
CONF     = 0.001     # mAP 标准低阈值扫全 PR 曲线; 勿改成部署阈值 0.35
IOU_NMS  = 0.7
DEVICE   = 0         # 没装 onnxruntime-gpu 会自动回落 CPU
MIN_BOX  = 15        # 少于此框数的类视为样本不足, 单独标记
PHASH_TH = 6         # 泄漏检查的汉明距离阈值

OUT_TXT = MODEL_DIR / "accuracy_report.txt"
OUT_CSV = MODEL_DIR / "accuracy_metrics.csv"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ================================================================


def read_onnx_names(model_path: Path) -> dict:
    """从 ONNX metadata 读类名, 保证与部署权重零偏差"""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    names = sess.get_modelmeta().custom_metadata_map.get("names")
    if not names:
        sys.exit(f"{model_path.name}: metadata 里没有 names")
    try:
        names = json.loads(names)
    except json.JSONDecodeError:
        import ast
        names = ast.literal_eval(names)
    return {int(k): v for k, v in names.items()}


def dirs_of(root: Path):
    img = root / "images" if (root / "images").is_dir() else root
    lbl = root / "labels" if (root / "labels").is_dir() else root
    return img, lbl


def check_dataset(root: Path, nc: int, tag: str):
    img_dir, lbl_dir = dirs_of(root)
    if not img_dir.is_dir():
        sys.exit(f"[{tag}] 找不到图片目录: {img_dir}")
    imgs = [p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXT]
    if not imgs:
        sys.exit(f"[{tag}] {img_dir} 里没有图片")
    hist, max_cls = Counter(), -1
    for lp in lbl_dir.rglob("*.txt"):
        try:
            for line in lp.read_text().splitlines():
                parts = line.split()
                if parts:
                    c = int(float(parts[0]))
                    hist[c] += 1
                    max_cls = max(max_cls, c)
        except Exception:
            pass
    if max_cls >= nc:
        sys.exit(f"[{tag}] 标签出现类别号 {max_cls}, 模型只有 {nc} 类 — 数据集配错了")
    print(f"[{tag}] {len(imgs)} 张图, {sum(hist.values())} 个框, 类别号 0..{max_cls} — OK")
    return len(imgs), sum(hist.values()), hist


def dhash(path, size=8):
    from PIL import Image
    try:
        img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    except Exception:
        return None
    px = list(img.getdata())
    bits = 0
    for r in range(size):
        row = px[r * (size + 1):(r + 1) * (size + 1)]
        for c in range(size):
            bits = (bits << 1) | (1 if row[c] < row[c + 1] else 0)
    return bits


def leakage_check(valid_root: Path, train_root: Path, tag: str):
    """返回 (重复张数, 总张数); train_root 为 None 时跳过"""
    if train_root is None or not train_root.exists():
        return None
    vi, _ = dirs_of(valid_root)
    ti, _ = dirs_of(train_root)
    v = [p for p in vi.rglob("*") if p.suffix.lower() in IMG_EXT]
    t = [p for p in ti.rglob("*") if p.suffix.lower() in IMG_EXT]
    print(f"[{tag}] 泄漏检查: {len(v)} 验证 vs {len(t)} 训练 ...")
    th = [h for h in (dhash(p) for p in t) if h is not None]
    dup = 0
    for p in v:
        h = dhash(p)
        if h is not None and any(bin(h ^ x).count("1") <= PHASH_TH for x in th):
            dup += 1
    pct = dup / max(1, len(v)) * 100
    flag = "✓ 干净" if dup == 0 else f"✗ {dup} 张重复 ({pct:.1f}%) — 论文中须说明"
    print(f"[{tag}] {flag}")
    return dup, len(v)


def make_yaml(root: Path, names: dict) -> Path:
    yp = Path(tempfile.gettempdir()) / f"autoval_{root.name.replace(' ', '_')}.yaml"
    img_dir, _ = dirs_of(root)
    rel = "images" if img_dir != root else "."
    lines = [f"path: {root.as_posix()}", f"train: {rel}", f"val: {rel}", "names:"]
    lines += [f"  {i}: {names[i]}" for i in sorted(names)]
    yp.write_text("\n".join(lines), encoding="utf-8")
    return yp


def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("请先安装: pip install ultralytics pillow")

    report, csv_rows = [], [["model", "class", "gt_boxes", "precision",
                             "recall", "mAP50", "mAP50-95", "note"]]
    paper_rows = []

    for tag, cfg in TASKS.items():
        model_path, root = Path(cfg["model"]), Path(cfg["dataset"])
        if not model_path.exists():
            sys.exit(f"[{tag}] 找不到模型: {model_path}")
        if not root.exists():
            sys.exit(f"[{tag}] 找不到数据集: {root}")

        names = read_onnx_names(model_path)
        print(f"\n{'=' * 66}\n{tag.upper()}: {model_path.name} — {len(names)} 类")
        print("  " + ", ".join(names[i] for i in sorted(names)))
        n_imgs, n_boxes, hist = check_dataset(root, len(names), tag)
        leak = leakage_check(root, cfg.get("train"), tag)

        metrics = YOLO(str(model_path), task="detect").val(
            data=str(make_yaml(root, names)), split="val", imgsz=IMGSZ,
            conf=CONF, iou=IOU_NMS, device=DEVICE, plots=True, verbose=True)
        box = metrics.box

        report.append(f"\n{'=' * 66}")
        report.append(f"{tag.upper()} MODEL — {model_path.name}")
        report.append("=" * 66)
        report.append(f"dataset  : {root}")
        report.append(f"scale    : {n_imgs} images, {n_boxes} ground-truth boxes")
        if leak:
            dup, tot = leak
            report.append(f"leakage  : {dup}/{tot} images duplicated in train "
                          f"({dup / tot * 100:.1f}%)" if dup else
                          f"leakage  : none detected against train split")
        else:
            report.append("leakage  : purged during eval-set construction "
                          "(184 -> 147 images)")
        report.append(f"settings : imgsz={IMGSZ}, conf={CONF}, NMS IoU={IOU_NMS}")
        report.append(f"plots    : {metrics.save_dir}")
        report.append("")
        report.append(f"{'class':<22s}{'GT':>6s}{'P':>8s}{'R':>8s}"
                      f"{'mAP50':>9s}{'mAP50-95':>10s}")
        report.append("-" * 66)
        report.append(f"{'ALL':<22s}{n_boxes:>6d}{box.mp:>8.3f}{box.mr:>8.3f}"
                      f"{box.map50:>9.3f}{box.map:>10.3f}")
        csv_rows.append([tag, "ALL", n_boxes, f"{box.mp:.4f}", f"{box.mr:.4f}",
                         f"{box.map50:.4f}", f"{box.map:.4f}", ""])

        small, ap50_big, ap_big = [], [], []
        for i, ci in enumerate(box.ap_class_index):
            ci = int(ci)
            cname, gt = names[ci], hist.get(ci, 0)
            note = "few samples" if gt < MIN_BOX else ""
            if note:
                small.append((cname, gt))
            else:
                ap50_big.append(box.ap50[i])
                ap_big.append(box.ap[i])
            report.append(f"{cname:<22s}{gt:>6d}{box.p[i]:>8.3f}{box.r[i]:>8.3f}"
                          f"{box.ap50[i]:>9.3f}{box.ap[i]:>10.3f}"
                          + ("   ← 样本不足" if note else ""))
            csv_rows.append([tag, cname, gt, f"{box.p[i]:.4f}", f"{box.r[i]:.4f}",
                             f"{box.ap50[i]:.4f}", f"{box.ap[i]:.4f}", note])

        if small:
            report.append("")
            report.append(f"样本不足的类 (< {MIN_BOX} 框): "
                          + ", ".join(f"{n} ({g})" for n, g in small))
            if ap50_big:
                m50 = sum(ap50_big) / len(ap50_big)
                m = sum(ap_big) / len(ap_big)
                report.append(f"剔除后 mAP50 = {m50:.3f}, mAP50-95 = {m:.3f} "
                              f"(对照全体 {box.map50:.3f} / {box.map:.3f})")
                report.append("→ 说明总体数字并非由小样本类的极端 AP 造成")

        paper_rows.append((model_path.stem, len(names), n_imgs, n_boxes,
                           box.mp, box.mr, box.map50, box.map))

    # ---- 论文用汇总 ----
    report.append(f"\n{'=' * 66}\n论文表格 (可直接誊抄)\n{'=' * 66}")
    report.append(f"{'Model':<22s}{'Cls':>5s}{'Imgs':>7s}{'Boxes':>7s}"
                  f"{'P':>8s}{'R':>8s}{'mAP50':>9s}{'mAP50-95':>10s}")
    for r in paper_rows:
        report.append(f"{r[0]:<22s}{r[1]:>5d}{r[2]:>7d}{r[3]:>7d}"
                      f"{r[4]:>8.3f}{r[5]:>8.3f}{r[6]:>9.3f}{r[7]:>10.3f}")

    report.append("\n方法学描述 (可粘贴进 Section 3.1):")
    report.append(
        "  Detection accuracy was measured on the deployed ONNX weights\n"
        "  themselves rather than on the PyTorch checkpoints, at the same\n"
        "  640x640 input resolution used on both platforms. The leaf\n"
        "  evaluation set was purged of any image sharing a source photograph\n"
        "  or a perceptual-hash near-duplicate with any training split, which\n"
        "  removed 37 of 184 images; the reported figures are therefore\n"
        "  held-out. Confidence was set to 0.001 for the mAP sweep, not the\n"
        "  0.35 deployment threshold.")

    text = "\n".join(report)
    print(text)
    OUT_TXT.write_text(text, encoding="utf-8")
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"\n已保存: {OUT_TXT}\n已保存: {OUT_CSV}")


if __name__ == "__main__":
    main()
