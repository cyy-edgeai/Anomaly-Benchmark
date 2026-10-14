# Natural seismic anomaly annotation quality report

## Release status

**human_reviewed_final**. Two domain experts manually reviewed the candidate intervals and produced the final consensus labels released here. Candidate-generation code was used only to focus inspection and did not determine ground truth. The label set is binary and contains no uncertain state. No anomalies were injected and no waveform samples were modified during this natural-label stage.

## Dataset and provenance

- Published source file: `data/raw/seismic_200.csv`
- SHA-256: `bd807a59ff60d307b2a94801ab15d9f97f7ff9d40ceceb184e6feb897998fdc7`
- Shape: 60,000 samples x 200 traces
- Sampling rate: 100 Hz; duration: 600 s per trace
- Original waveform copied into this output directory: no

## Label definition

- `0`: reviewed normal/background or physically coherent seismic signal
- `1`: human-reviewed data-quality anomaly
- Sample indices are zero-based with inclusive `start_sample` and `end_sample`.

## Findings

- Exact duplicate groups detected during QC: [[101, 196], [102, 197], [103, 198], [104, 199], [105, 200]]
- Duplicate follower traces without full-record duplicate labels: [196, 197, 198, 199, 200]
- Localized anomaly intervals propagated to identical follower traces: 1
- Propagated anomaly points on identical follower traces: 29
- Candidate intervals reviewed independently by two domain experts: 48
- Accepted source anomaly events after expert consensus: 34
- Rejected physical/normal candidates: 14
- Published anomaly intervals: 35 (34 source intervals plus 1 propagation to an exactly duplicated trace)
- Point labels: 11,999,625 normal, 375 anomaly
- Anomaly prevalence: 0.003125%

Traces 196--200 remain documented as exact waveform duplicates of traces 101--105 for data-quality provenance. Their former full-record duplicate labels were removed, while each human-reviewed localized anomaly is propagated point-for-point from the corresponding identical source trace. Strong broad events near 182 s and 400 s, and similar moveout/ringing patterns, remain protected as physical signals rather than labeled by amplitude alone.

## Expert-review protocol

Each expert examined the target trace together with neighboring traces, waveform morphology, temporal context, and cross-trace moveout/coherence. Disagreements were reconciled to a single consensus decision before export. Physically coherent seismic arrivals were retained as background even when their amplitudes were large. Expert identities are not included in the public release; the frozen consensus labels are provided in `point_labels.npz` and `anomaly_intervals.csv`.

- Number of domain experts: 2
- Final decision rule: expert consensus
- Exported label values: 0 and 1 only

Primary paper metrics should report both sample-level and event-level results. Split train/calibration/test by contiguous time blocks with guard gaps; do not randomly split overlapping windows.

## Publication checklist

1. Report the anomaly taxonomy, index convention, sampling rate, source hash, and label prevalence.
2. Freeze the decision file and regenerated outputs; cite their hashes in the manuscript or supplement.
3. State that exact waveform duplicates were not assigned full-record duplicate labels, while localized reviewed labels were propagated to their identical copies.
