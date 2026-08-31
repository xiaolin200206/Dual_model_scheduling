# -*- coding: utf-8 -*-
"""
find_dataset.py — 扫描目录, 找出和两个部署 ONNX 类名完全匹配的数据集

把 ROOT 指向截图里那个总目录 (装着 Detection_leaf_disease 等文件夹的那层),
运行后它会:
  1. 递归找出所有 *.yaml / *.yml
  2. 解析其中的 names 字段, 和 ONNX 里的类名逐一比对
  3. 对匹配的数据集, 检查目录结构并统计每个 split 的图片/标签数
  4. 直接打印出应该填进 run_model_eval.py 的 dataset 路径

用法:  python find_dataset.py
"""

import sys
import os
from pathlib import Path

# ============================ CONFIG ============================

# 截图里那个总目录 (含 Detection_leaf_disease / Detection_pest_disease 等):
ROOT = Path(os.environ.get("DATASET_ROOT", "."))  # root of the (unreleased) image datasets

MAX_DEPTH = 4          # 递归深度, 够用了; 找不到就调大

# 来自 model_report.txt 的部署权重类名 (顺序敏感)
LEAF_NAMES = ["algal", "leaf_rot", "Phomopsis", "pink", "root"]
PEST_NAMES = ["leafhopper damage", "Psyllid", "Psyllid_damage",
              "Scale_insect", "Stem-borer", "weevil", "weevil_damage"]

# ================================================================

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_names(yaml_path: Path):
    """解析 yaml 的 names 字段 -> 有序类名列表, 解析失败返回 None"""
    try:
        import yaml
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None
    if not isinstance(data, dict) or "names" not in data:
        return None
    names = data["names"]
    if isinstance(names, dict):
        try:
            return [names[k] for k in sorted(names, key=int)]
        except Exception:
            return [v for _, v in sorted(names.items())]
    if isinstance(names, list):
        return names
    return None


def count_split(split_dir: Path):
    """split 目录下 images/ labels/ 的文件数; 兼容图片直接平铺的情况"""
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    if img_dir.is_dir():
        n_img = sum(1 for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
        n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.is_dir() else 0
        return n_img, n_lbl
    n_img = sum(1 for p in split_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMG_EXT)
    n_lbl = len(list(split_dir.glob("*.txt")))
    return n_img, n_lbl


def inspect_dataset(yaml_path: Path):
    """围绕 yaml 所在目录找 train/valid(val)/test 结构"""
    base = yaml_path.parent
    found = []
    for split in ("test", "valid", "val", "train"):
        d = base / split
        if d.is_dir():
            n_img, n_lbl = count_split(d)
            found.append((split, d, n_img, n_lbl))
    # ultralytics 原生结构: images/<split>, labels/<split>
    if not found and (base / "images").is_dir():
        for split in ("test", "val", "train"):
            d = base / "images" / split
            if d.is_dir():
                n_img = sum(1 for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
                ld = base / "labels" / split
                n_lbl = len(list(ld.glob("*.txt"))) if ld.is_dir() else 0
                found.append((split, d, n_img, n_lbl))
    return found


def match_label(names, ref):
    if names is None:
        return "no-names"
    if [n.strip() for n in names] == ref:
        return "EXACT"
    if sorted(n.strip().lower() for n in names) == sorted(r.lower() for r in ref):
        return "same-set-different-order"
    if len(names) == len(ref):
        return "same-count-different-names"
    return "different"


def main():
    if not ROOT.exists():
        sys.exit(f"ROOT 不存在: {ROOT} — 改成截图那层目录的实际路径")

    yamls = []
    def walk(d: Path, depth: int):
        if depth > MAX_DEPTH:
            return
        try:
            for p in d.iterdir():
                if p.is_dir() and p.name not in {"runs", "__pycache__", ".git",
                                                 "node_modules", "venv", ".venv"}:
                    walk(p, depth + 1)
                elif p.suffix.lower() in (".yaml", ".yml"):
                    yamls.append(p)
        except PermissionError:
            pass
    walk(ROOT, 0)
    print(f"扫描 {ROOT}\n共找到 {len(yamls)} 个 yaml\n" + "=" * 70)

    hits = {"leaf": [], "pest": []}
    for yp in yamls:
        names = load_names(yp)
        if names is None:
            continue
        for tag, ref in (("leaf", LEAF_NAMES), ("pest", PEST_NAMES)):
            verdict = match_label(names, ref)
            if verdict in ("EXACT", "same-set-different-order"):
                hits[tag].append((yp, verdict, names))
            elif verdict == "same-count-different-names" and len(names) == len(ref):
                # 类数相同但名字不同 — 大概率旧版数据集, 也报告出来供排除
                hits[tag].append((yp, verdict, names))

    for tag, ref in (("leaf", LEAF_NAMES), ("pest", PEST_NAMES)):
        print(f"\n########  {tag.upper()}  (部署权重类名: {ref})  ########")
        if not hits[tag]:
            print("  没有找到类数匹配的 yaml — 调大 MAX_DEPTH 或换 ROOT 再扫")
            continue
        for yp, verdict, names in hits[tag]:
            print(f"\n  {yp}")
            print(f"    判定: {verdict}")
            if verdict != "EXACT":
                print(f"    该 yaml 类名: {names}")
                if verdict == "same-count-different-names":
                    print("    → 类数相同但名字不同, 是旧版数据集, 不要用")
                    continue
                if verdict == "same-set-different-order":
                    print("    → 类名相同但顺序不同 — 顺序即类别号, 配错会静默算错 mAP,")
                    print("      除非确认标签号与该顺序一致, 否则不要用")
                    continue
            splits = inspect_dataset(yp)
            if not splits:
                print("    (yaml 旁边没找到 train/val/test 目录结构)")
            for split, d, n_img, n_lbl in splits:
                mark = "  ← 推荐用这个填进 run_model_eval.py" \
                       if split in ("test", "valid", "val") else ""
                print(f"    {split:<6s}: {n_img:>5d} images, {n_lbl:>5d} labels   {d}{mark}")

    print("\n" + "=" * 70)
    print("下一步: 把上面标 EXACT 的数据集里 test (没有就 valid/val) 目录的路径")
    print("填进 run_model_eval.py 的 dataset 字段, 再运行 python run_model_eval.py")


if __name__ == "__main__":
    main()
