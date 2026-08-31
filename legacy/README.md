# legacy/ — model-evaluation tooling (not used by the paper)

These scripts evaluate the detection accuracy of the two deployed ONNX artefacts
and audit their validation partitions for train/validation leakage. They were
written for an earlier version of the study and are kept for completeness.

**No result in the paper depends on them.** The paper characterises latency,
throughput, utilisation, temperature, pack-side power and cost, none of which
depends on what the models have learned.

They cannot be run by a third party as-is: they require the unreleased image
datasets and weights. Set `DATASET_ROOT` to the dataset directory if you have
access, or adapt the path tables at the top of each script to your own data.

- `audit/` — source-stem and perceptual-hash leakage check across partition
  boundaries; emits a manifest of excluded images by hash.
- `accuracy/` — mAP@0.5 and per-class AP on the exported ONNX artefacts at the
  deployment resolution.
