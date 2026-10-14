# Synthetic anomaly augmentation quality report

Overall status: **PASS**

## Scope

Existing human-reviewed natural labels are preserved as class 1. Newly generated labels are deterministic procedural ground truth (classes 2–9); they must not be described as manually verified natural anomalies.

## Automated checks

- PASS — `source_hash_unchanged`
- PASS — `shape_preserved`
- PASS — `finite_augmented_values`
- PASS — `natural_and_synthetic_do_not_overlap`
- PASS — `natural_binary_labels_preserved`
- PASS — `natural_multiclass_labels_preserved_as_class_1`
- PASS — `outside_synthetic_intervals_bitwise_unchanged`
- PASS — `synthetic_labels_reconstruct_from_catalog`
- PASS — `synthetic_events_do_not_overlap_per_point`
- PASS — `excluded_duplicate_groups_not_injected`
- PASS — `duplicate_waveform_pairs_remain_bitwise_equal`
- PASS — `duplicate_label_pairs_remain_equal`
- PASS — `buffer_splits_not_injected`
- PASS — `event_counts_match_configuration`
- PASS — `all_phenotype_checks_pass`
- PASS — `class_ids_within_taxonomy`

## Counts

- Natural labeled points: 375
- Synthetic labeled points: 80,614
- Combined labeled points: 80,989
- Synthetic events: 165

## Publication notes

- Report the fixed seed, source SHA-256, configuration SHA-256, class taxonomy, event-level parameter ranges, and temporal split boundaries.
- Keep train/calibration/test temporal roles fixed and exclude both buffer gaps to reduce temporal leakage.
- The exact duplicate groups 101/196 through 105/200 were excluded from injection so augmentation does not create artificial disagreement inside duplicate pairs.
- Synthetic events may overlap protected coherent physical signals because the paper calls for injection on active real backgrounds; every such overlap is explicitly recorded in `synthetic_events.csv`.
- Time values use 100 Hz from the current reviewed-label manifest. Verify this sampling rate against acquisition metadata before manuscript submission because the headerless CSV cannot prove it by itself.
- No uncertainty class or uncertainty weighting is present in this release.
- Class 9 strong noise is broadband white-plus-AR(1) additive noise, distinct from class 8 high-frequency jitter. Its added-noise RMS is 2–4 times the local/background reference RMS, with a 5% edge taper and spectral-flatness validation.
