# -*- coding: utf-8 -*-
"""
check_leakage_phash.py — 用感知哈希复查泄漏 (字节哈希查不出重导出的重复图)

为什么需要这个:
  check_leakage.py 用 md5 比对, 只要图片被重新导出/缩放/重新压缩过, 字节
  就完全不同, 重复图会被漏掉。B/valid 与 D/valid 标注完全相同 (597 框,
  每类 275/83/160/20/59) 却零字节重叠, 强烈提示 B 是同一批图的 1280 重导版。

方法:
  dHash (difference hash): 缩放到 9x8 灰度, 比较相邻像素亮度, 得到 64 位指纹。
  对缩放、重压缩、轻微调整鲁棒。汉明距离 <= THRESH 视为同一张图。
  另外用文件名尾部的 Roboflow 原始 ID 做交叉验证 (rf.<hash> 前的原图名)。

依赖: pip install pillow
用法: python check_leakage_phash.py
"""

import re
from pathlib import Path
from collections import defaultdict

BASE = Path(r"C:\Users\Lim Ding Shan\Desktop\Durian project and paper")

SPLITS = {
    "A/valid": BASE / "Combined_model" / "Leaf" / "_merged" / "valid",
    "A/train": BASE / "Combined_model" / "Leaf" / "_merged" / "train",
    "B/valid": BASE / "six paper" / "_refair_eval" / "leaf_1280" / "valid",
    "D/valid": BASE / "Leave_disease_merged" / "valid",
    "D/train": BASE / "Leave_disease_merged" / "train",
    "C/train": BASE / "dataset_all" / "merged_dataset" / "train",
}

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
THRESH = 6          # 汉明距离阈值; 0-4 几乎肯定同图, 5-8 很可能, >12 不同


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


def stem_key(name: str):
    """提取 Roboflow 原始图名: xxx__IMG_0185_jpg.rf.<hash>.jpg -> IMG_0185"""
    m = re.search(r"(IMG[_-]?\d+)", name, re.IGNORECASE)
    return m.group(1).upper().replace("-", "_") if m else None


def load(split_dir: Path):
    img_dir = split_dir / "images" if (split_dir / "images").is_dir() else split_dir
    items = []
    if not img_dir.is_dir():
        return items
    for p in sorted(img_dir.rglob("*")):
        if p.suffix.lower() in IMG_EXT and p.is_file():
            h = dhash(p)
            if h is not None:
                items.append((h, p.name, stem_key(p.name)))
    return items


def ham(a, b):
    return bin(a ^ b).count("1")


def main():
    data = {}
    print("计算感知哈希中 (比 md5 慢, 稍等) ...")
    for tag, d in SPLITS.items():
        items = load(d)
        data[tag] = items
        print(f"  {tag:<10s} {len(items):>5d} 张")

    valids = [t for t in SPLITS if t.endswith("/valid") and data[t]]
    trains = [t for t in SPLITS if t.endswith("/train") and data[t]]

    print("\n" + "=" * 74)
    print(f"感知哈希泄漏判定 (汉明距离 <= {THRESH} 视为同图)")
    print("=" * 74)
    for v in valids:
        print(f"\n  {v}  ({len(data[v])} 张)")
        for tr in trains:
            dup = []
            for hv, nv, kv in data[v]:
                for ht, nt, kt in data[tr]:
                    if ham(hv, ht) <= THRESH:
                        dup.append((nv, nt))
                        break
            pct = len(dup) / max(1, len(data[v])) * 100
            mark = "✓" if not dup else "✗"
            print(f"    {mark} 与 {tr}: {len(dup)} 张疑似重复 ({pct:.1f}%)")
            for nv, nt in dup[:3]:
                print(f"        {nv[:60]}")
                print(f"     ~= {nt[:60]}")

    # --- 用 Roboflow 原图 ID 交叉验证 ---
    print("\n" + "=" * 74)
    print("原始图名 (IMG_xxxx) 交叉验证")
    print("=" * 74)
    keys = {t: {k for _, _, k in data[t] if k} for t in data}
    for v in valids:
        print(f"\n  {v}  ({len(keys[v])} 个可识别原图 ID)")
        for tr in trains:
            inter = keys[v] & keys[tr]
            mark = "✓" if not inter else "✗"
            print(f"    {mark} 与 {tr}: {len(inter)} 个相同原图 ID")
            if inter:
                print(f"        {', '.join(sorted(inter)[:8])}")

    # --- B 是否 D/valid 的重导版 ---
    print("\n" + "=" * 74)
    print("B/valid 与 D/valid 是否同源 (验证 md5 零重叠是否为重导出假象)")
    print("=" * 74)
    if data.get("B/valid") and data.get("D/valid"):
        same = 0
        for hb, nb, kb in data["B/valid"]:
            for hd, nd, kd in data["D/valid"]:
                if ham(hb, hd) <= THRESH:
                    same += 1
                    break
        pct = same / len(data["B/valid"]) * 100
        print(f"  {same} / {len(data['B/valid'])} 张匹配 ({pct:.1f}%)")
        if pct > 80:
            print("  => B 确实是 D/valid 的重导出版本, 字节零重叠是假象;")
            print("     B 的泄漏状况应视同 D/valid。")
        elif pct < 20:
            print("  => B 与 D/valid 内容不同, 是独立划分, 之前的『干净』判定成立。")
        else:
            print("  => 部分重叠, 需人工检查。")

    print("\n" + "=" * 74)
    print("结论用法: 选『两项检查都判 ✓』的 valid 集评估。若全部有泄漏,")
    print("论文中必须写明评估集与训练集存在部分重叠及其比例。")


if __name__ == "__main__":
    main()
