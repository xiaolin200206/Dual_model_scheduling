# -*- coding: utf-8 -*-
"""
build_clean_eval.py — 从 B/valid 剔除所有与任一 train 集重叠的图,
                      生成一个干净的 held-out 评估集

排除标准 (满足任一即剔除, 双重保险):
  1. 感知哈希 dHash 与任一 train 图汉明距离 <= THRESH  (抓重导出/缩放/连拍近重复)
  2. 文件名中的原图 ID (IMG_xxxx) 出现在任一 train 集中 (抓 Roboflow 增广变体)

对照的 train 集取三个候选的并集 (A/train, D/train, C/train), 因为无法确知
2026-06-17 那次训练用的是哪一个 — 取并集是保守做法。

输出: leaf_eval_clean/{images,labels}/  + 一份 manifest.txt 记录剔除明细
依赖: pip install pillow
用法: python build_clean_eval.py
"""

import re
import shutil
from collections import Counter
from pathlib import Path

BASE = Path(r"C:\Users\Lim Ding Shan\Desktop\Durian project and paper")

SOURCE = BASE / "six paper" / "_refair_eval" / "leaf_1280" / "valid"   # B/valid
TRAIN_SETS = [
    BASE / "Combined_model" / "Leaf" / "_merged" / "train",            # A/train
    BASE / "Leave_disease_merged" / "train",                           # D/train
    BASE / "dataset_all" / "merged_dataset" / "train",                 # C/train
]
OUT = BASE / "Fifth paper" / "leaf_eval_clean"

CLASS_NAMES = ["algal", "leaf_rot", "Phomopsis", "pink", "root"]  # 部署 ONNX 顺序
THRESH = 6
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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


def img_id(name):
    m = re.search(r"(IMG[_-]?\d+)", name, re.IGNORECASE)
    return m.group(1).upper().replace("-", "_") if m else None


def img_dir_of(split):
    return split / "images" if (split / "images").is_dir() else split


def lbl_dir_of(split):
    return split / "labels" if (split / "labels").is_dir() else split


def list_imgs(split):
    d = img_dir_of(split)
    return [p for p in sorted(d.rglob("*"))
            if p.suffix.lower() in IMG_EXT and p.is_file()] if d.is_dir() else []


def ham(a, b):
    return bin(a ^ b).count("1")


def class_hist(label_paths):
    c = Counter()
    for lp in label_paths:
        try:
            for line in lp.read_text().splitlines():
                parts = line.split()
                if parts:
                    c[int(float(parts[0]))] += 1
        except Exception:
            pass
    return c


def main():
    print("读取 train 集 (并集) ...")
    train_hashes, train_ids = [], set()
    for ts in TRAIN_SETS:
        imgs = list_imgs(ts)
        print(f"  {ts.name:<20s} {len(imgs):>5d} 张   {ts}")
        for p in imgs:
            h = dhash(p)
            if h is not None:
                train_hashes.append(h)
            k = img_id(p.name)
            if k:
                train_ids.add(k)
    print(f"  合计 {len(train_hashes)} 张, {len(train_ids)} 个可识别原图 ID")

    src_imgs = list_imgs(SOURCE)
    print(f"\n源评估集: {len(src_imgs)} 张   {SOURCE}")
    if not src_imgs:
        raise SystemExit("源目录为空, 检查 SOURCE 路径")

    keep, drop = [], []
    for p in src_imgs:
        reasons = []
        k = img_id(p.name)
        if k and k in train_ids:
            reasons.append(f"原图ID {k} 在 train 中")
        h = dhash(p)
        if h is not None and any(ham(h, th) <= THRESH for th in train_hashes):
            reasons.append("感知哈希与 train 近重复")
        (drop if reasons else keep).append((p, reasons))

    print(f"保留 {len(keep)} 张, 剔除 {len(drop)} 张 "
          f"({len(drop) / len(src_imgs) * 100:.1f}%)")

    # ---- 复制 ----
    out_img, out_lbl = OUT / "images", OUT / "labels"
    if OUT.exists():
        shutil.rmtree(OUT)
    out_img.mkdir(parents=True)
    out_lbl.mkdir(parents=True)

    src_lbl = lbl_dir_of(SOURCE)
    kept_labels, missing = [], 0
    for p, _ in keep:
        shutil.copy2(p, out_img / p.name)
        lp = src_lbl / (p.stem + ".txt")
        if lp.exists():
            shutil.copy2(lp, out_lbl / lp.name)
            kept_labels.append(out_lbl / lp.name)
        else:
            (out_lbl / (p.stem + ".txt")).write_text("")   # 背景图: 空标签
            missing += 1

    # ---- 统计 ----
    before = class_hist([src_lbl / (p.stem + ".txt") for p in src_imgs
                         if (src_lbl / (p.stem + ".txt")).exists()])
    after = class_hist(kept_labels)

    print("\n" + "=" * 66)
    print(f"{'类别':<14s}{'原有框数':>10s}{'保留框数':>10s}{'保留率':>10s}")
    print("=" * 66)
    for i, name in enumerate(CLASS_NAMES):
        b, a = before.get(i, 0), after.get(i, 0)
        rate = f"{a / b * 100:.0f}%" if b else "-"
        warn = "  ⚠ 样本过少" if 0 < a < 15 else ""
        print(f"{name:<14s}{b:>10d}{a:>10d}{rate:>10s}{warn}")
    print("=" * 66)
    print(f"{'合计':<14s}{sum(before.values()):>10d}{sum(after.values()):>10d}")
    print(f"图片: {len(src_imgs)} -> {len(keep)}  (其中无标注背景图 {missing} 张)")

    # ---- manifest ----
    man = [f"clean eval set built from: {SOURCE}",
           f"purged against train union: {[str(t) for t in TRAIN_SETS]}",
           f"criteria: dHash hamming <= {THRESH}  OR  shared IMG id",
           f"kept {len(keep)} / {len(src_imgs)} images",
           "", "=== DROPPED ==="]
    for p, reasons in drop:
        man.append(f"{p.name}\t{'; '.join(reasons)}")
    (OUT / "manifest.txt").write_text("\n".join(man), encoding="utf-8")

    print(f"\n输出: {OUT}")
    print(f"剔除明细: {OUT / 'manifest.txt'}")
    print("\n下一步: 把下面这行填进 run_model_eval.py 的 leaf dataset:")
    print(f'    r"{OUT}"')


if __name__ == "__main__":
    main()
