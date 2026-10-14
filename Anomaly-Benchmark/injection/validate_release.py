#!/usr/bin/env python3
"""Independently validate the published benchmark release without modifying it."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_SPLIT_RANGES = {
    "train": (0, 32000),
    "buffer_1": (32000, 36096),
    "calibration": (36096, 44192),
    "buffer_2": (44192, 48288),
    "test": (48288, 60000),
}
FROZEN_INJECTION_SAMPLING_RANGES = {
    "train": (0, 31980),
    "calibration": (36090, 44190),
    "test": (48300, 60000),
}
FROZEN_COUNTS = {
    "synthetic_events": 165,
    "natural_points": 375,
    "synthetic_points": 80614,
    "combined_points": 80989,
    "test_synthetic_events": 49,
    "test_synthetic_points": 24645,
}


def sha256_file(path: Path, block_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate(config_path: Path, results_dir: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads((results_dir / "augmentation_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((results_dir / "quality_audit.json").read_text(encoding="utf-8"))
    source = Path(config["source_csv"])
    if not source.is_absolute():
        source = (config_path.parent / source).resolve()

    check(source.is_file(), f"Source is missing: {source}")
    check(sha256_file(source) == config["source_sha256"], "Source SHA-256 mismatch")
    configured_splits = {
        name: tuple(int(value) for value in bounds)
        for name, bounds in config["split_ranges"].items()
    }
    check(
        configured_splits == EXPERIMENT_SPLIT_RANGES,
        "Configured splits do not match the experiment program constants",
    )
    configured_sampling_ranges = {
        name: tuple(int(value) for value in bounds)
        for name, bounds in config["frozen_injection_sampling_ranges"].items()
    }
    check(
        configured_sampling_ranges == FROZEN_INJECTION_SAMPLING_RANGES,
        "Frozen injection sampling ranges do not match the paper artifact",
    )
    check(audit["status"] == "PASS", "Generation-time audit did not pass")
    check(all(audit["checks"].values()), "At least one generation-time check is false")
    for filename, metadata in manifest["output_files"].items():
        output = results_dir / filename
        check(output.is_file(), f"Manifest output is missing: {filename}")
        check(sha256_file(output) == metadata["sha256"], f"Output SHA-256 mismatch: {filename}")

    waveforms_archive = np.load(results_dir / "augmented_waveforms.npz", allow_pickle=False)
    labels_archive = np.load(results_dir / "combined_labels.npz", allow_pickle=False)
    augmented = waveforms_archive["waveforms"]
    binary = labels_archive["binary_labels"]
    classes = labels_archive["class_labels"]
    natural = labels_archive["natural_reviewed_labels"]
    synthetic = labels_archive["synthetic_class_labels"]
    split_ids = labels_archive["split_ids"]
    split_id_map = json.loads(str(labels_archive["split_id_map_json"][0]))
    expected_shape = tuple(config["expected_shape"])
    check(augmented.shape == expected_shape and augmented.dtype == np.float32, "Waveform shape or dtype mismatch")
    check(binary.shape == expected_shape and classes.shape == expected_shape, "Label shape mismatch")
    check(np.array_equal(binary, (classes > 0).astype(np.uint8)), "Binary and class labels disagree")
    check(not np.any((natural > 0) & (synthetic > 0)), "Natural and synthetic labels overlap")
    check(np.all(classes[natural > 0] == 1), "Natural labels are not class 1")
    check(np.isin(classes, [int(key) for key in config["class_map"]]).all(), "Class IDs are outside the current taxonomy")
    check(int(natural.sum()) == FROZEN_COUNTS["natural_points"], "Frozen natural-point count mismatch")
    check(int((synthetic > 0).sum()) == FROZEN_COUNTS["synthetic_points"], "Frozen synthetic-point count mismatch")
    check(int(binary.sum()) == FROZEN_COUNTS["combined_points"], "Frozen combined-point count mismatch")

    expected_split_id_map = {
        name: split_id for split_id, name in enumerate(EXPERIMENT_SPLIT_RANGES)
    }
    check(split_id_map == expected_split_id_map, "Embedded split ID map mismatch")
    expected_split_ids = np.empty(expected_shape[0], dtype=np.uint8)
    for name, (start, stop) in EXPERIMENT_SPLIT_RANGES.items():
        expected_split_ids[start:stop] = expected_split_id_map[name]
    check(np.array_equal(split_ids, expected_split_ids), "Embedded split IDs mismatch")

    split_dir = config_path.parent.parent / "data" / "splits"
    split_indices_path = split_dir / "split_indices.npz"
    split_json = json.loads((split_dir / "split_indices.json").read_text(encoding="utf-8"))
    split_manifest = pd.read_csv(split_dir / "split_manifest.csv")
    result_split_manifest = pd.read_csv(results_dir / "split_manifest.csv")
    check(split_indices_path.is_file(), "Exact split-index archive is missing")
    check(
        list(split_json["splits"]) == list(EXPERIMENT_SPLIT_RANGES),
        "Split JSON names/order mismatch",
    )
    with np.load(split_indices_path, allow_pickle=False) as split_archive:
        check(
            split_archive.files == list(EXPERIMENT_SPLIT_RANGES),
            "Split NPZ names/order mismatch",
        )
        for name, (start, stop) in EXPERIMENT_SPLIT_RANGES.items():
            expected = np.arange(start, stop, dtype=np.int32)
            check(
                split_archive[name].dtype == np.int32
                and np.array_equal(split_archive[name], expected),
                f"Exact split-index array mismatch: {name}",
            )
            item = split_json["splits"][name]
            check(
                int(item["start"]) == start
                and int(item["end_exclusive"]) == stop
                and int(item["sample_count"]) == stop - start,
                f"Split JSON boundary mismatch: {name}",
            )
    check(
        split_manifest.equals(result_split_manifest),
        "Published split manifests are not byte-semantically identical",
    )
    for row in split_manifest.itertuples(index=False):
        start, stop = EXPERIMENT_SPLIT_RANGES[str(row.split)]
        check(
            int(row.start_sample) == start
            and int(row.end_sample_exclusive) == stop
            and int(row.sample_count) == stop - start,
            f"Split manifest boundary mismatch: {row.split}",
        )

    original = pd.read_csv(source, header=None, dtype=np.float32).to_numpy(copy=False)
    check(original.shape == expected_shape, "Source shape mismatch")
    synthetic_mask = synthetic > 0
    check(np.array_equal(original[~synthetic_mask], augmented[~synthetic_mask]), "Waveforms changed outside synthetic intervals")
    for first, second in config["duplicate_trace_groups_1_based"]:
        check(np.array_equal(augmented[:, first - 1], augmented[:, second - 1]), f"Duplicate waveforms differ: {first}/{second}")
        check(np.array_equal(classes[:, first - 1], classes[:, second - 1]), f"Duplicate labels differ: {first}/{second}")

    events = pd.read_csv(results_dir / "synthetic_events.csv")
    reconstructed = np.zeros(expected_shape, dtype=np.uint8)
    occupancy = np.zeros(expected_shape, dtype=np.uint8)
    for row in events.itertuples(index=False):
        trace_indices = [int(value) - 1 for value in str(row.trace_ids).split(";")]
        split_start, split_stop = EXPERIMENT_SPLIT_RANGES[str(row.split)]
        check(
            split_start <= int(row.start_sample) < int(row.end_sample_exclusive) <= split_stop,
            f"Synthetic event crosses its configured split: {row.event_id}",
        )
        reconstructed[int(row.start_sample) : int(row.end_sample_exclusive), trace_indices] = int(row.class_id)
        occupancy[int(row.start_sample) : int(row.end_sample_exclusive), trace_indices] += 1
    check(np.array_equal(reconstructed, synthetic), "Event catalog does not reconstruct synthetic labels")
    check(int(occupancy.max()) <= 1, "Synthetic events overlap at point level")
    check(len(events) == FROZEN_COUNTS["synthetic_events"], "Frozen synthetic-event count mismatch")
    test_events = events[events["split"] == "test"]
    check(len(test_events) == FROZEN_COUNTS["test_synthetic_events"], "Frozen test-event count mismatch")
    test_start, test_stop = EXPERIMENT_SPLIT_RANGES["test"]
    check(
        int((synthetic[test_start:test_stop] > 0).sum()) == FROZEN_COUNTS["test_synthetic_points"],
        "Frozen test synthetic-point count mismatch",
    )
    for anomaly_type, type_config in config["anomaly_types"].items():
        for split, expected_count in type_config.get("events_per_split", config["events_per_type_by_split"]).items():
            actual_count = int(((events["anomaly_type"] == anomaly_type) & (events["split"] == split)).sum())
            check(actual_count == expected_count, f"Event count mismatch: {anomaly_type}/{split}")
    detailed_stats = pd.read_csv(results_dir / "anomaly_duration_statistics.csv")
    check(int(detailed_stats["labeled_point_count"].sum()) == int(binary.sum()), "Detailed statistics do not reconcile to the point labels")

    portable = pd.read_csv(results_dir / "combined_binary_labels.csv.gz")
    trace_columns = [f"trace_{index:03d}" for index in range(1, expected_shape[1] + 1)]
    check(portable.shape == (expected_shape[0], expected_shape[1] + 3), "Portable CSV shape mismatch")
    check(np.array_equal(portable[trace_columns].to_numpy(dtype=np.uint8), binary), "Portable CSV labels disagree with NPZ")
    expected_split_names = np.empty(expected_shape[0], dtype=object)
    for name, (start, stop) in EXPERIMENT_SPLIT_RANGES.items():
        expected_split_names[start:stop] = name
    check(
        np.array_equal(portable["split"].to_numpy(dtype=object), expected_split_names),
        "Portable CSV split column mismatch",
    )
    source_copies = [path for path in SCRIPT_DIR.rglob(source.name) if path.resolve() != source.resolve()]
    check(not source_copies, "Original source CSV was copied into datayu")

    return {
        "status": "PASS",
        "shape": list(expected_shape),
        "synthetic_events": int(len(events)),
        "natural_points": int(natural.sum()),
        "synthetic_points": int(synthetic_mask.sum()),
        "combined_points": int(binary.sum()),
        "manifest_outputs_verified": len(manifest["output_files"]),
        "exact_split_arrays_verified": len(EXPERIMENT_SPLIT_RANGES),
        "source_copies_in_datayu": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "anomaly_config.json")
    parser.add_argument("--results-dir", type=Path, default=SCRIPT_DIR / "results")
    args = parser.parse_args()
    try:
        summary = validate(args.config.resolve(), args.results_dir.resolve())
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
