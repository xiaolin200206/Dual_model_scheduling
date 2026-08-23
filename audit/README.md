# Dataset-integrity audit

The procedure applied here originates in Lin (2026a) and its released script
(github.com/xiaolin200206/Edge-Disease-Inference-Engine, `audit/`). Two signals
recover source grouping across a partition boundary:

1. **Export stem.** Roboflow exports are `<original-stem>_jpg.rf.<uuid>.jpg`; the
   text before `.rf.` is the pre-export filename, so two exports of one source
   photograph share it. This is exact and invariant to geometric augmentation.
2. **Perceptual hash over the eight dihedral transforms**, catching variants whose
   stems were rewritten but whose content survives flips and right-angle rotations.

Groups are the connected components of the union of both relations plus MD5
identity. A component is assigned wholly to one side of the boundary, so no source
photograph can straddle it.

This is an **upper-bound correction, not a proof of leak freedom**: perceptual
hashing is not invariant to cropping, shear or heavy colour jitter, so some variants
remain unmatched, and two genuinely distinct photographs of one lesion may be
merged. Both are stated rather than assumed away.

## Order of use

```bash
python find_dataset.py            # which partitions match the deployed class list
python check_leakage_phash.py     # how much cross-partition duplication exists
python build_clean_eval.py        # emit the leak-free evaluation set + manifest
```

`build_clean_eval.py` writes `manifest.txt` recording every excluded image and the
reason for its exclusion. Images are identified there by 64-bit dHash and by the
minimum Hamming distance to a training image, never by filename or by the image
itself, so the exclusion is checkable without releasing the dataset. That manifest,
not the images, is what the paper releases; a copy generated from the partitions used
in the paper is committed here as `manifest.txt`.
