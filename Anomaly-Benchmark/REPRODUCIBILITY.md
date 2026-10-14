# Reproducibility

## Install

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## Validate the frozen release

```bash
python injection/validate_release.py \
  --config injection/anomaly_config.json \
  --results-dir data/augmented
```

The validator checks source and output hashes, frozen point totals, label consistency, natural/synthetic non-overlap, duplicate-trace handling, event-catalog reconstruction, event counts, exact split rules, and the invariant that waveform values change only at injected positions.

## Regenerate

```bash
python injection/augment_seismic_anomalies.py \
  --config injection/anomaly_config.json \
  --output-dir reproduced-augmented
```

Use a new output directory. The checked-in `data/augmented/` directory is the frozen publication artifact.

The configuration intentionally separates exact experiment `split_ranges` from `frozen_injection_sampling_ranges`. The latter preserves the original paper artifact's random-call sequence only; it never determines train/calibration/test membership. Regeneration fails if any realized event is outside the exact experiment split declared in `split_ranges`.

## Verify the extracted training augmentation

The dataset validator does not require PyTorch. Install the optional training-augmentation runtime only when this extracted logic is needed:

```bash
python -m pip install -r requirements-training-augmentation.txt
python injection/training_augmentation.py \
  --config injection/training_augmentation_config.json \
  --self-test
```

Expected status is `PASS`. The self-test resets both RNG streams and verifies deterministic replay for the background stage and the online self-supervised stage. The reported experiments used PyTorch 1.13; strict bit-for-bit replay across machines should use the same PyTorch version, device type, and seed.

Minimal API usage:

```python
import torch
from injection.training_augmentation import (
    load_config,
    make_self_supervised_view,
    seed_all,
)

cfg = load_config("injection/training_augmentation_config.json")
seed_all(cfg["randomness"]["default_seed"])
windows = torch.zeros((16, 512), dtype=torch.float32)
augmented, procedural_mask = make_self_supervised_view(windows, cfg)
```

Do not use `training_augmentation_config.json` to regenerate the frozen benchmark. Frozen benchmark regeneration remains controlled only by `anomaly_config.json`.

## Read exact indices

```python
import numpy as np

with np.load("data/splits/split_indices.npz", allow_pickle=False) as splits:
    train = splits["train"]
    calibration = splits["calibration"]
    test = splits["test"]
```

## Verify repository checksums

`SHA256SUMS.txt` lists every published file except itself. Compare each file with its SHA-256 before using the release in a paper or benchmark.
