# Dataset card

## Scope

This release supports research on seismic waveform data-quality anomaly detection. It contains original waveform values, two-expert final consensus labels for reviewed natural anomalies, and exact procedural ground truth for injected transformations.

It is not a geophysical event catalog. Natural and synthetic results must be reported separately.

## Source

| Item | Value |
|---|---|
| Shape | 60,000 samples × 200 traces |
| Sampling rate | 100 Hz |
| Duration | 600 seconds |
| Source SHA-256 | `bd807a59ff60d307b2a94801ab15d9f97f7ff9d40ceceb184e6feb897998fdc7` |

## Labels

- Class 0: background/no published anomaly label.
- Class 1: natural data-quality anomaly, final consensus after manual review by two domain experts.
- Class 2: step.
- Class 3: zero dropout/disconnection.
- Class 4: flatline.
- Class 5: drift.
- Class 6: saturation clipping.
- Class 7: gain jump.
- Class 8: high-frequency jitter.
- Class 9: broadband strong noise.

`combined_labels.npz` contains `binary_labels`, `class_labels`, `natural_reviewed_labels`, `synthetic_class_labels`, `split_ids`, class/split maps, sample indices, trace IDs, sampling rate, and source/config hashes.

Frozen counts: 35 reviewed natural intervals/375 natural points, 165 synthetic events/80,614 synthetic points, and 80,989 combined points. The exact test interval `[48288, 60000)` contains 49 synthetic events and 24,645 synthetic points.

## Conventions

- Trace identifiers in CSV catalogs are one-based.
- NumPy array columns are zero-based.
- Sample indices are zero-based.
- Split intervals use `[start, end)`.
- Natural labels and injected labels are disjoint.

## Limitations

- The 100 Hz sampling rate should be confirmed against acquisition metadata because the source CSV is headerless.
- Synthetic transformations do not represent every possible real-world fault.
- Synthetic F1 must not be interpreted as real deployment precision.
- A zero label is not proof that an unknown natural anomaly is absent.
