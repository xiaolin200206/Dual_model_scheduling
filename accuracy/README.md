# Detection accuracy

`run_model_eval.py` evaluates the **exported ONNX artefacts**, not the PyTorch
checkpoints, at the deployment input resolution of 640 × 640. Class names are read
from the ONNX metadata so that the evaluated taxonomy cannot drift from the deployed
one.

Two settings are deliberate and should not be changed when reproducing:

- `conf = 0.001` is the standard operating point for a mAP sweep. The deployment
  threshold is 0.35 and is not used for any accuracy figure.
- Classes below fifteen instances are flagged and excluded from a secondary
  aggregate, so that a sparse class cannot carry the headline number. The paper
  reports both aggregates.

`sensitivity_pest.py` sweeps NMS IoU, confidence, partition and weight format to
localise how the reported mAP depends on evaluation settings.
