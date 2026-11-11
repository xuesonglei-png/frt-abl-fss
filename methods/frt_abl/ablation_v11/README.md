# Independent ablation implementations

This directory contains independent training implementations for the full method and its ablation variants on MNIST-add2 and HWF. Each variant has its own settings, protocol, and training entry point, while reusing the core model and numerical utilities.

The variants cover confidence-based approximation, fixed thresholds, curriculum learning, reliability weighting, boundary supervision, standard abductive learning, and supervised anchor training. A combined fixed-threshold and no-curriculum variant is also included.

Run an entry point from the repository root to inspect its plan:

```powershell
python -m methods.frt_abl.ablation_v11.full.train --task hwf --seed 4
```

Training starts only when `--execute-training` is supplied. Dataset files must be prepared separately.
