# Anomaly injection and split protocol

## Two configurations with different purposes

This release deliberately distinguishes two reproducibility targets:

- `injection/anomaly_config.json` and `injection/augment_seismic_anomalies.py` generate the frozen, published anomaly-injected dataset. They use seed `20260913`.
- `injection/training_augmentation_config.json` and `injection/training_augmentation.py` reproduce augmentation performed by the actual experiment program. They use seed `20260520` and do not contain model architecture or training-loss code.

The two configurations must not be interchanged. The first defines a persistent evaluation artifact with exact event catalogs; the second defines stochastic views created during the experimental workflow.

## Determinism and provenance

The generator reads `injection/anomaly_config.json`, verifies the source SHA-256, uses NumPy generator seed `20260913`, and records the canonical configuration hash and program hash in `data/augmented/augmentation_manifest.json`.

Two range fields serve distinct purposes. `split_ranges` is authoritative for membership and exactly matches the experiment constants. `frozen_injection_sampling_ranges` retains the historical candidate envelopes solely to reproduce the originally frozen random event catalog. Changing the candidate envelope and rerunning the same seed would alter later random draws. The generator therefore checks every realized event against `split_ranges` and fails if any event touches a buffer or crosses an exact boundary. In this release, all 165 original events pass that check.

The original waveform is never modified in place. The generated `augmented_waveforms.npz` differs from the original only at positions where `synthetic_class_labels > 0`; this invariant is checked automatically.

## Injection policy

- Synthetic events are injected into train, calibration, and test only.
- Buffer splits remain free of synthetic injection.
- Synthetic events cannot overlap final natural-anomaly labels.
- A within-trace guard separates injected events.
- Traces 101–105 and their exact duplicates 196–200 are excluded from injection.
- Active waveform backgrounds are preferred to avoid unrealistic all-quiet examples.
- Overlap with protected coherent physical signals is permitted only as real background and is explicitly recorded in the event catalog.

## Event counts

For each of the original seven injected types, the release contains 10 train, 4 calibration, and 6 test events. Strong noise contains 13 train, 5 calibration, and 7 test events. The total is 165 synthetic events.

The frozen label totals are 80,614 synthetic points and 80,989 combined natural-plus-synthetic points. The exact test split contains 49 synthetic events and 24,645 synthetic points; it contains no reviewed natural-anomaly point.

## Exact parameters

Parameter ranges are stored in `injection/anomaly_config.json`. Every realized event is stored in `data/augmented/synthetic_events.csv`, including split, trace IDs, start/end samples, duration, class, severity, and transform-specific values.

## Split policy

The split boundaries are copied directly from the experiment program constants: train `[0, 32000)`, validation/calibration `[36096, 44192)`, and test `[48288, 60000)`. The three usable intervals are separated by two 4,096-sample guard buffers. Hyperparameters and thresholds may be selected only on train/calibration. Test labels are reserved for final evaluation.

## Extracted experiment augmentation

The extracted module preserves the main program's float32 storage and random-call order.

### Background stage

For every selected trace, the program draws and applies, in order:

1. an integer shift from `[-4, 4]` samples with replicated edge padding;
2. a multiplicative scale from `[0.90, 1.10]`;
3. independent zero-mean Gaussian noise whose standard deviation is drawn once per trace from `[0.01, 0.05]`.

The trace set is then cleaned using four robust cross-trace passes with neighbor radius 4, residual clip factor 0.25, and minimum robust scale 0.05. `augment_background_traces` accepts a caller-owned `random.Random` object when the augmentation must continue an already-advanced random stream.

### Online self-supervised stage

For a normal training tensor shaped `[batch, time]`, `make_self_supervised_view` first adds Gaussian noise with standard deviation `0.012`. For each row, augmentation is activated with probability `0.96`; 2–7 positions are selected from shuffled, dispersed bins within 5%–96% of the window. One of the ten configured anomaly types is then sampled at each position. The exact duration, amplitude, local-noise, clipping, gain, and drift ranges are all machine-readable in `training_augmentation_config.json`.

The function returns `(corrupted_window, procedural_mask)`. Mask value 1 identifies samples covered by an online anomaly event; mask value 0 can still contain the base Gaussian perturbation. Event decisions use Python `random`, whereas random tensors use PyTorch. `seed_all` seeds both streams.
