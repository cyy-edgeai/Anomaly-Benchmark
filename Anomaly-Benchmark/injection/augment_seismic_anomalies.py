#!/usr/bin/env python3
"""Build a deterministic, auditable seismic anomaly augmentation benchmark.

The source CSV is read in place and never copied. Existing human-reviewed point
labels are immutable. Synthetic labels are procedural ground truth and remain
distinguishable from natural labels through the multiclass label matrix.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIRNAME = "results"
BUFFER_SPLITS = {"buffer_1", "buffer_2"}
INJECTION_SPLITS = ("train", "calibration", "test")


def sha256_file(path: Path, block_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_config_path(value: str, config_dir: Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (config_dir / candidate).resolve()


def robust_location_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(x, axis=0).astype(np.float64)
    mad = np.median(np.abs(x - center[None, :]), axis=0).astype(np.float64)
    scale = 1.4826 * mad
    std = np.std(x, axis=0, dtype=np.float64)
    scale = np.where(scale > 1e-8, scale, std)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return center, scale


def split_for_sample(sample: int, split_ranges: dict[str, list[int]]) -> str:
    for name, (start, stop) in split_ranges.items():
        if start <= sample < stop:
            return name
    raise ValueError(f"Sample {sample} is outside configured splits")


def validate_splits(split_ranges: dict[str, list[int]], n_samples: int) -> None:
    ordered = sorted((int(v[0]), int(v[1]), k) for k, v in split_ranges.items())
    if ordered[0][0] != 0 or ordered[-1][1] != n_samples:
        raise ValueError("Split ranges must cover the full recording")
    for left, right in zip(ordered, ordered[1:]):
        if left[1] != right[0]:
            raise ValueError("Split ranges must be contiguous and non-overlapping")


def validate_frozen_sampling_ranges(
    sampling_ranges: dict[str, list[int]],
    split_ranges: dict[str, list[int]],
    n_samples: int,
) -> None:
    """Validate historical sampling envelopes without redefining data splits."""
    if set(sampling_ranges) != set(INJECTION_SPLITS):
        raise ValueError(
            "frozen_injection_sampling_ranges must contain exactly "
            f"{list(INJECTION_SPLITS)}"
        )
    for name in INJECTION_SPLITS:
        start, stop = (int(v) for v in sampling_ranges[name])
        if not (0 <= start < stop <= n_samples):
            raise ValueError(f"Invalid frozen injection sampling range for {name}: [{start}, {stop})")
        if name not in split_ranges:
            raise ValueError(f"Missing exact experiment split: {name}")


def load_inputs(config: dict[str, Any], config_path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    paths = {
        "source": resolve_config_path(config["source_csv"], config_path.parent),
        "natural_labels": resolve_config_path(config["natural_labels_npz"], config_path.parent),
        "natural_intervals": resolve_config_path(config["natural_intervals_csv"], config_path.parent),
        "protected_events": resolve_config_path(config["protected_events_csv"], config_path.parent),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")

    actual_hash = sha256_file(paths["source"])
    if actual_hash != config["source_sha256"]:
        raise ValueError(f"Source SHA-256 mismatch: {actual_hash}")

    data = pd.read_csv(paths["source"], header=None, dtype=np.float32).to_numpy(copy=True)
    expected_shape = tuple(int(v) for v in config["expected_shape"])
    if data.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError("Source contains NaN or infinite values")

    label_archive = np.load(paths["natural_labels"], allow_pickle=False)
    natural = label_archive["labels"].astype(np.uint8, copy=True)
    if natural.shape != data.shape or not np.isin(natural, [0, 1]).all():
        raise ValueError("Natural label matrix is not a shape-matched binary matrix")
    archived_hash = str(label_archive["source_sha256"].reshape(-1)[0])
    if archived_hash != actual_hash:
        raise ValueError("Natural labels refer to a different source file")

    natural_events = pd.read_csv(paths["natural_intervals"])
    protected_events = pd.read_csv(paths["protected_events"])
    return data, natural, natural_events, protected_events, paths


def dilate_natural_labels(natural: np.ndarray, guard: int) -> np.ndarray:
    blocked = natural.astype(bool, copy=True)
    if guard <= 0:
        return blocked
    samples, traces = np.nonzero(natural)
    by_trace: dict[int, list[int]] = defaultdict(list)
    for sample, trace in zip(samples.tolist(), traces.tolist()):
        by_trace[trace].append(sample)
    for trace, positions in by_trace.items():
        for sample in positions:
            blocked[max(0, sample - guard) : min(blocked.shape[0], sample + guard + 1), trace] = True
    return blocked


def choose_trace_block(
    rng: np.random.Generator,
    eligible_trace_ids: np.ndarray,
    trace_count: int,
    n_traces: int,
    excluded_zero_based: set[int],
) -> list[int]:
    starts = []
    for first_1_based in eligible_trace_ids.tolist():
        first = first_1_based - 1
        block = list(range(first, first + trace_count))
        if block[-1] < n_traces and not any(index in excluded_zero_based for index in block):
            starts.append(block)
    if not starts:
        raise RuntimeError("No eligible contiguous trace block")
    return starts[int(rng.integers(0, len(starts)))]


def segment_is_clear(blocked: np.ndarray, start: int, stop: int, traces: list[int]) -> bool:
    return not blocked[start:stop, traces].any()


def activity_score(data: np.ndarray, start: int, stop: int, traces: list[int], scale: np.ndarray) -> float:
    scores = []
    for trace in traces:
        segment = data[start:stop, trace].astype(np.float64)
        if segment.size < 2:
            scores.append(0.0)
        else:
            scores.append(float(np.std(segment) / scale[trace]))
    return float(np.median(scores))


def select_location(
    rng: np.random.Generator,
    data: np.ndarray,
    blocked: np.ndarray,
    scale: np.ndarray,
    eligible_trace_ids: np.ndarray,
    excluded_zero_based: set[int],
    split_start: int,
    split_stop: int,
    duration: int,
    trace_count: int,
    candidate_pool: int,
) -> tuple[int, int, list[int], float]:
    if duration >= split_stop - split_start:
        raise ValueError("Configured anomaly duration does not fit split")
    candidates: list[tuple[float, int, list[int]]] = []
    attempts = max(candidate_pool * 10, 200)
    for _ in range(attempts):
        traces = choose_trace_block(rng, eligible_trace_ids, trace_count, data.shape[1], excluded_zero_based)
        start = int(rng.integers(split_start, split_stop - duration + 1))
        stop = start + duration
        if segment_is_clear(blocked, start, stop, traces):
            score = activity_score(data, start, stop, traces, scale)
            candidates.append((score, start, traces))
            if len(candidates) >= candidate_pool:
                break
    if not candidates:
        raise RuntimeError("Could not place a non-overlapping anomaly; reduce counts or guards")
    candidates.sort(key=lambda item: item[0], reverse=True)
    top_count = max(1, math.ceil(len(candidates) * 0.25))
    score, start, traces = candidates[int(rng.integers(0, top_count))]
    return start, start + duration, traces, score


def local_reference(data: np.ndarray, start: int, stop: int, trace: int) -> tuple[float, float]:
    left = max(0, start - 256)
    right = min(data.shape[0], stop + 256)
    context = np.concatenate((data[left:start, trace], data[stop:right, trace])).astype(np.float64)
    if context.size < 16:
        context = data[:, trace].astype(np.float64)
    center = float(np.median(context))
    mad = float(np.median(np.abs(context - center)))
    scale = 1.4826 * mad
    if scale <= 1e-8:
        scale = float(np.std(context))
    return center, max(scale, 1.0)


def apply_one_event(
    anomaly_type: str,
    augmented: np.ndarray,
    original: np.ndarray,
    start: int,
    stop: int,
    traces: list[int],
    type_config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], float]:
    parameters: list[dict[str, Any]] = []
    severity_values: list[float] = []
    duration = stop - start
    for trace in traces:
        center, local_scale = local_reference(original, start, stop, trace)
        raw = original[start:stop, trace].astype(np.float64)
        transformed = raw.copy()
        detail: dict[str, Any] = {
            "trace_id": trace + 1,
            "local_center": center,
            "local_robust_scale": local_scale,
        }

        if anomaly_type == "step":
            magnitude = float(rng.uniform(*type_config["severity_sigma"]))
            sign = int(rng.choice([-1, 1]))
            offset = sign * magnitude * local_scale
            transformed = raw + offset
            severity_values.append(magnitude)
            detail.update({"signed_offset": offset, "magnitude_sigma": magnitude})
        elif anomaly_type == "dropout_zero":
            transformed.fill(0.0)
            severity = abs(center) / local_scale
            severity_values.append(severity)
            detail.update({"replacement_value": 0.0, "center_to_zero_sigma": severity})
        elif anomaly_type == "flatline":
            reference_start = max(0, start - 8)
            reference = original[reference_start:start, trace]
            flat_value = float(np.median(reference)) if reference.size else center
            transformed.fill(flat_value)
            severity = float(np.std(raw) / local_scale)
            severity_values.append(severity)
            detail.update({"flat_value": flat_value, "removed_variability_sigma": severity})
        elif anomaly_type == "drift":
            magnitude = float(rng.uniform(*type_config["severity_sigma"]))
            sign = int(rng.choice([-1, 1]))
            curvature = float(rng.uniform(-0.35, 0.35))
            time = np.linspace(0.0, 1.0, duration, endpoint=True)
            profile = time + curvature * (time * time - time)
            offset = sign * magnitude * local_scale * profile
            transformed = raw + offset
            severity_values.append(magnitude)
            detail.update({
                "signed_terminal_offset": float(offset[-1]),
                "magnitude_sigma": magnitude,
                "curvature": curvature,
            })
        elif anomaly_type == "saturation_clipping":
            quantile = float(rng.uniform(*type_config["clip_quantile"]))
            deviations = np.abs(raw - center)
            rail = float(np.quantile(deviations, quantile))
            rail = max(rail, 0.25 * local_scale, np.finfo(np.float32).eps)
            transformed = np.clip(raw, center - rail, center + rail)
            clipped_fraction = float(np.mean(np.abs(raw - center) >= rail))
            severity_values.append(clipped_fraction)
            detail.update({
                "lower_rail": center - rail,
                "upper_rail": center + rail,
                "clip_quantile": quantile,
                "clipped_fraction": clipped_fraction,
            })
        elif anomaly_type == "gain_jump":
            if rng.random() < float(type_config["high_gain_probability"]):
                gain = float(rng.uniform(*type_config["gain_high"]))
            else:
                gain = float(rng.uniform(*type_config["gain_low"]))
            transformed = center + gain * (raw - center)
            severity = max(gain, 1.0 / gain)
            severity_values.append(severity)
            detail.update({"gain_factor": gain, "severity_ratio": severity})
        elif anomaly_type == "high_frequency_jitter":
            magnitude = float(rng.uniform(*type_config["severity_sigma"]))
            frequency = float(rng.uniform(*type_config["normalized_frequency_cycles_per_sample"]))
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            white = rng.normal(0.0, 1.0, duration + 8)
            kernel = np.ones(7, dtype=np.float64) / 7.0
            low = np.convolve(white, kernel, mode="valid")[:duration]
            high = white[4 : 4 + duration] - low
            sinusoid = np.sin(2.0 * np.pi * frequency * np.arange(duration) + phase)
            jitter = 0.65 * high + 0.35 * sinusoid
            jitter /= max(float(np.std(jitter)), 1e-8)
            transformed = raw + magnitude * local_scale * jitter
            severity_values.append(magnitude)
            detail.update({
                "magnitude_sigma": magnitude,
                "normalized_frequency_cycles_per_sample": frequency,
                "phase_radians": phase,
            })
        elif anomaly_type == "strong_noise":
            # Broadband noise, not the high-pass/narrowband jitter transform.
            ratio = float(rng.uniform(*type_config["noise_to_background_rms_ratio"]))
            rho = float(rng.uniform(*type_config["ar1_coefficient"]))
            white_weight = float(rng.uniform(*type_config["white_noise_weight"]))
            innovations = rng.normal(0.0, 1.0, duration)
            colored = np.empty(duration, dtype=np.float64)
            colored[0] = innovations[0]
            for index in range(1, duration):
                colored[index] = rho * colored[index - 1] + math.sqrt(1.0 - rho * rho) * innovations[index]
            noise = white_weight * rng.normal(0.0, 1.0, duration) + (1.0 - white_weight) * colored
            taper_count = max(1, int(duration * float(type_config["taper_fraction"])))
            envelope = np.ones(duration, dtype=np.float64)
            ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, taper_count))
            envelope[:taper_count] = ramp
            envelope[-taper_count:] = ramp[::-1]
            noise *= envelope
            noise -= np.mean(noise)
            noise /= max(float(np.sqrt(np.mean(noise * noise))), 1e-8)
            background_rms = max(float(np.std(raw)), local_scale, 1.0)
            added_noise = ratio * background_rms * noise
            transformed = raw + added_noise
            severity_values.append(ratio)
            detail.update({
                "noise_model": "broadband_white_plus_AR1",
                "noise_to_background_rms_ratio": ratio,
                "background_reference_rms": background_rms,
                "ar1_coefficient": rho,
                "white_noise_weight": white_weight,
                "taper_fraction": float(type_config["taper_fraction"]),
            })
        else:
            raise KeyError(f"Unsupported anomaly type: {anomaly_type}")

        augmented[start:stop, trace] = transformed.astype(np.float32)
        parameters.append(detail)
    return parameters, float(np.median(severity_values))


def protected_overlap(start: int, stop: int, protected: pd.DataFrame) -> tuple[bool, str]:
    overlaps = protected[
        (protected["start_sample"].astype(int) < stop)
        & (protected["end_sample"].astype(int) + 1 > start)
    ]
    return (not overlaps.empty, ";".join(overlaps["protected_event_id"].astype(str).tolist()))


def generate_augmentation(
    original: np.ndarray,
    natural: np.ndarray,
    protected: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(int(config["random_seed"]))
    sampling_ranges = config.get("frozen_injection_sampling_ranges", config["split_ranges"])
    n_samples, n_traces = original.shape
    _, global_scale = robust_location_scale(original)
    excluded_1_based = {int(v) for v in config["excluded_trace_ids_1_based"]}
    excluded_zero_based = {v - 1 for v in excluded_1_based}
    eligible_trace_ids = np.array([v for v in range(1, n_traces + 1) if v not in excluded_1_based])
    blocked = dilate_natural_labels(natural, int(config["natural_label_guard_samples"]))
    synthetic_classes = np.zeros_like(natural, dtype=np.uint8)
    augmented = original.copy()
    records: list[dict[str, Any]] = []
    event_number = 0
    event_guard = int(config["synthetic_event_guard_samples"])

    for anomaly_type, type_config in config["anomaly_types"].items():
        class_id = int(type_config["class_id"])
        duration_min, duration_max = (int(v) for v in type_config["duration_samples"])
        trace_values = np.asarray(type_config["trace_count_values"], dtype=int)
        trace_probs = np.asarray(type_config["trace_count_probabilities"], dtype=float)
        trace_probs = trace_probs / trace_probs.sum()
        for split_name in INJECTION_SPLITS:
            counts_by_split = type_config.get("events_per_split", config["events_per_type_by_split"])
            count = int(counts_by_split[split_name])
            split_start, split_stop = (int(v) for v in sampling_ranges[split_name])
            exact_start, exact_stop = (int(v) for v in config["split_ranges"][split_name])
            for _ in range(count):
                duration = int(rng.integers(duration_min, duration_max + 1))
                trace_count = int(rng.choice(trace_values, p=trace_probs))
                start, stop, traces, activity = select_location(
                    rng,
                    original,
                    blocked,
                    global_scale,
                    eligible_trace_ids,
                    excluded_zero_based,
                    split_start,
                    split_stop,
                    duration,
                    trace_count,
                    int(config["activity_candidate_pool"]),
                )
                if not (exact_start <= start < stop <= exact_stop):
                    raise RuntimeError(
                        "Frozen event placement falls outside the exact experiment split: "
                        f"{split_name} event [{start}, {stop}) versus [{exact_start}, {exact_stop}). "
                        "Do not change the seed or frozen sampling ranges for this release."
                    )
                parameters, severity = apply_one_event(
                    anomaly_type,
                    augmented,
                    original,
                    start,
                    stop,
                    traces,
                    type_config,
                    rng,
                )
                synthetic_classes[start:stop, traces] = class_id
                guard_start = max(split_start, start - event_guard)
                guard_stop = min(split_stop, stop + event_guard)
                blocked[guard_start:guard_stop, traces] = True
                overlaps_protected, protected_ids = protected_overlap(start, stop, protected)
                event_number += 1
                records.append({
                    "event_id": f"SYN{event_number:04d}",
                    "origin": "synthetic_procedural_ground_truth",
                    "anomaly_type": anomaly_type,
                    "class_id": class_id,
                    "split": split_name,
                    "trace_ids": ";".join(str(v + 1) for v in traces),
                    "trace_start": min(traces) + 1,
                    "trace_end": max(traces) + 1,
                    "num_traces": len(traces),
                    "start_sample": start,
                    "end_sample_inclusive": stop - 1,
                    "end_sample_exclusive": stop,
                    "start_time_s": start / float(config["sampling_rate_hz"]),
                    "end_time_s_exclusive": stop / float(config["sampling_rate_hz"]),
                    "duration_samples": duration,
                    "duration_seconds": duration / float(config["sampling_rate_hz"]),
                    "severity": severity,
                    "background_activity_robust_sigma": activity,
                    "overlaps_protected_physical_event": overlaps_protected,
                    "protected_event_ids": protected_ids,
                    "transform_parameters_json": json.dumps(parameters, ensure_ascii=False, separators=(",", ":")),
                })

    events = pd.DataFrame.from_records(records)
    return augmented, synthetic_classes, blocked, events


def build_combined_events(natural_events: pd.DataFrame, synthetic_events: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    fs = float(config["sampling_rate_hz"])
    rows = []
    for _, row in natural_events.iterrows():
        start = int(row["start_sample"])
        end_inclusive = int(row["end_sample"])
        trace = int(row["trace"])
        rows.append({
            "event_id": str(row["event_id"]),
            "origin": "natural_human_reviewed",
            "anomaly_type": str(row["anomaly_type"]),
            "class_id": 1,
            "split": split_for_sample(start, config["split_ranges"]),
            "trace_ids": str(trace),
            "trace_start": trace,
            "trace_end": trace,
            "num_traces": 1,
            "start_sample": start,
            "end_sample_inclusive": end_inclusive,
            "end_sample_exclusive": end_inclusive + 1,
            "start_time_s": start / fs,
            "end_time_s_exclusive": (end_inclusive + 1) / fs,
            "duration_samples": end_inclusive - start + 1,
            "duration_seconds": (end_inclusive - start + 1) / fs,
            "severity": np.nan,
            "background_activity_robust_sigma": np.nan,
            "overlaps_protected_physical_event": False,
            "protected_event_ids": "",
            "transform_parameters_json": "",
        })
    natural_normalized = pd.DataFrame(rows, columns=synthetic_events.columns)
    return pd.concat([natural_normalized, synthetic_events], ignore_index=True)


def build_split_manifest(config: dict[str, Any], n_samples: int) -> pd.DataFrame:
    rows = []
    fs = float(config["sampling_rate_hz"])
    for split, (start, stop) in config["split_ranges"].items():
        rows.append({
            "split": split,
            "start_sample": int(start),
            "end_sample_inclusive": int(stop) - 1,
            "end_sample_exclusive": int(stop),
            "start_time_s": int(start) / fs,
            "end_time_s_exclusive": int(stop) / fs,
            "sample_count": int(stop) - int(start),
            "fraction_of_record": (int(stop) - int(start)) / n_samples,
            "synthetic_injection_enabled": split in INJECTION_SPLITS,
            "recommended_use": {
                "train": "training_background; synthetic labels optional",
                "calibration": "threshold calibration and model selection",
                "test": "final evaluation only",
                "buffer_1": "temporal leakage guard; exclude from fitting and evaluation",
                "buffer_2": "temporal leakage guard; exclude from fitting and evaluation",
            }[split],
        })
    return pd.DataFrame(rows)


def reconstruct_synthetic_labels(events: pd.DataFrame, shape: tuple[int, int]) -> np.ndarray:
    reconstructed = np.zeros(shape, dtype=np.uint8)
    for _, row in events.iterrows():
        start = int(row["start_sample"])
        stop = int(row["end_sample_exclusive"])
        traces = [int(v) - 1 for v in str(row["trace_ids"]).split(";")]
        reconstructed[start:stop, traces] = int(row["class_id"])
    return reconstructed


def event_point_occupancy(events: pd.DataFrame, shape: tuple[int, int]) -> np.ndarray:
    occupancy = np.zeros(shape, dtype=np.uint8)
    for _, row in events.iterrows():
        start = int(row["start_sample"])
        stop = int(row["end_sample_exclusive"])
        traces = [int(v) - 1 for v in str(row["trace_ids"]).split(";")]
        occupancy[start:stop, traces] += 1
    return occupancy


def phenotype_audit(
    original: np.ndarray,
    augmented: np.ndarray,
    events: pd.DataFrame,
) -> tuple[bool, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    for _, row in events.iterrows():
        anomaly_type = str(row["anomaly_type"])
        start, stop = int(row["start_sample"]), int(row["end_sample_exclusive"])
        traces = [int(v) - 1 for v in str(row["trace_ids"]).split(";")]
        passed = True
        metrics: dict[str, float] = {}
        for trace in traces:
            raw = original[start:stop, trace].astype(np.float64)
            changed = augmented[start:stop, trace].astype(np.float64)
            delta = changed - raw
            if anomaly_type == "dropout_zero":
                value = float(np.max(np.abs(changed)))
                metrics["max_abs_value"] = max(metrics.get("max_abs_value", 0.0), value)
                passed &= value == 0.0
            elif anomaly_type == "flatline":
                value = float(np.ptp(changed))
                metrics["max_peak_to_peak"] = max(metrics.get("max_peak_to_peak", 0.0), value)
                passed &= value <= 1e-6
            elif anomaly_type == "saturation_clipping":
                unchanged_fraction = float(np.mean(changed == raw))
                metrics["max_unchanged_fraction"] = max(metrics.get("max_unchanged_fraction", 0.0), unchanged_fraction)
                passed &= unchanged_fraction < 0.8
            elif anomaly_type == "drift":
                if delta.size > 2:
                    correlation = float(np.corrcoef(np.arange(delta.size), delta)[0, 1])
                else:
                    correlation = 0.0
                metrics["min_abs_time_correlation"] = min(metrics.get("min_abs_time_correlation", 1.0), abs(correlation))
                passed &= abs(correlation) > 0.9
            elif anomaly_type == "strong_noise":
                details = json.loads(row["transform_parameters_json"])
                detail = next(item for item in details if int(item["trace_id"]) == trace + 1)
                rms_ratio = float(np.sqrt(np.mean(delta * delta)) / detail["background_reference_rms"])
                spectrum = np.abs(np.fft.rfft(delta - np.mean(delta)))[1:] ** 2
                epsilon = max(float(np.mean(spectrum)) * 1e-12, 1e-12)
                flatness = float(np.exp(np.mean(np.log(spectrum + epsilon))) / np.mean(spectrum + epsilon))
                metrics["min_noise_to_background_rms_ratio"] = min(metrics.get("min_noise_to_background_rms_ratio", np.inf), rms_ratio)
                metrics["min_spectral_flatness"] = min(metrics.get("min_spectral_flatness", 1.0), flatness)
                passed &= rms_ratio >= 1.95 and flatness >= 0.1
            else:
                rms_delta = float(np.sqrt(np.mean(delta * delta)))
                metrics["min_rms_delta"] = min(metrics.get("min_rms_delta", np.inf), rms_delta)
                passed &= rms_delta > 0.0
        checks.append({"event_id": row["event_id"], "anomaly_type": anomaly_type, "passed": bool(passed), **metrics})
    return all(item["passed"] for item in checks), checks


def quality_audit(
    original: np.ndarray,
    augmented: np.ndarray,
    natural: np.ndarray,
    synthetic_classes: np.ndarray,
    combined_binary: np.ndarray,
    combined_classes: np.ndarray,
    synthetic_events: pd.DataFrame,
    config: dict[str, Any],
    source_hash_before: str,
    source_hash_after: str,
) -> dict[str, Any]:
    synthetic_mask = synthetic_classes > 0
    reconstructed = reconstruct_synthetic_labels(synthetic_events, original.shape)
    occupancy = event_point_occupancy(synthetic_events, original.shape)
    excluded = np.asarray(config["excluded_trace_ids_1_based"], dtype=int) - 1
    duplicate_pairs = [tuple(int(v) - 1 for v in pair) for pair in config["duplicate_trace_groups_1_based"]]
    buffer_has_synthetic = False
    for split in BUFFER_SPLITS:
        start, stop = config["split_ranges"][split]
        buffer_has_synthetic |= bool(synthetic_mask[int(start) : int(stop)].any())
    phenotype_ok, phenotype_details = phenotype_audit(original, augmented, synthetic_events)

    expected_counts = {
        f"{anomaly_type}:{split}": int(type_config.get("events_per_split", config["events_per_type_by_split"])[split])
        for anomaly_type, type_config in config["anomaly_types"].items()
        for split in INJECTION_SPLITS
    }
    actual_counts = Counter(
        f"{row.anomaly_type}:{row.split}" for row in synthetic_events.itertuples(index=False)
    )
    checks = {
        "source_hash_unchanged": source_hash_before == source_hash_after == config["source_sha256"],
        "shape_preserved": augmented.shape == original.shape,
        "finite_augmented_values": bool(np.isfinite(augmented).all()),
        "natural_and_synthetic_do_not_overlap": not bool((natural.astype(bool) & synthetic_mask).any()),
        "natural_binary_labels_preserved": bool(np.all(combined_binary[natural.astype(bool)] == 1)),
        "natural_multiclass_labels_preserved_as_class_1": bool(np.all(combined_classes[natural.astype(bool)] == 1)),
        "outside_synthetic_intervals_bitwise_unchanged": bool(np.array_equal(augmented[~synthetic_mask], original[~synthetic_mask])),
        "synthetic_labels_reconstruct_from_catalog": bool(np.array_equal(reconstructed, synthetic_classes)),
        "synthetic_events_do_not_overlap_per_point": int(occupancy.max()) <= 1,
        "excluded_duplicate_groups_not_injected": not bool(synthetic_mask[:, excluded].any()),
        "duplicate_waveform_pairs_remain_bitwise_equal": all(
            np.array_equal(augmented[:, left], augmented[:, right]) for left, right in duplicate_pairs
        ),
        "duplicate_label_pairs_remain_equal": all(
            np.array_equal(combined_classes[:, left], combined_classes[:, right]) for left, right in duplicate_pairs
        ),
        "buffer_splits_not_injected": not buffer_has_synthetic,
        "event_counts_match_configuration": all(actual_counts[key] == value for key, value in expected_counts.items()),
        "all_phenotype_checks_pass": phenotype_ok,
        "class_ids_within_taxonomy": bool(np.isin(combined_classes, [int(k) for k in config["class_map"]]).all()),
    }
    return {
        "schema_version": "1.0",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "counts": {
            "natural_labeled_points": int(natural.sum()),
            "synthetic_labeled_points": int(synthetic_mask.sum()),
            "combined_labeled_points": int(combined_binary.sum()),
            "synthetic_events": int(len(synthetic_events)),
            "synthetic_events_by_type": dict(Counter(synthetic_events["anomaly_type"])),
            "synthetic_events_by_split": dict(Counter(synthetic_events["split"])),
        },
        "expected_event_counts_by_type_and_split": expected_counts,
        "actual_event_counts_by_type_and_split": dict(actual_counts),
        "phenotype_event_checks": phenotype_details,
    }


def build_type_summary(
    combined_classes: np.ndarray,
    combined_events: pd.DataFrame,
    class_map: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for class_id_text, label in class_map.items():
        class_id = int(class_id_text)
        if class_id == 0:
            continue
        events = combined_events[combined_events["class_id"] == class_id]
        durations = events["duration_samples"].astype(float)
        trace_set: set[int] = set()
        for value in events["trace_ids"].astype(str):
            trace_set.update(int(v) for v in value.split(";") if v)
        rows.append({
            "class_id": class_id,
            "label": label,
            "origin": "natural_human_reviewed" if class_id == 1 else "synthetic_procedural_ground_truth",
            "event_count": int(len(events)),
            "point_count": int(np.sum(combined_classes == class_id)),
            "affected_trace_count": len(trace_set),
            "duration_samples_min": float(durations.min()) if len(durations) else np.nan,
            "duration_samples_median": float(durations.median()) if len(durations) else np.nan,
            "duration_samples_max": float(durations.max()) if len(durations) else np.nan,
        })
    return pd.DataFrame(rows)


def save_binary_labels_csv(path: Path, binary: np.ndarray, fs: float, split_ranges: dict[str, list[int]]) -> None:
    frame = pd.DataFrame(binary, columns=[f"trace_{index:03d}" for index in range(1, binary.shape[1] + 1)])
    frame.insert(0, "split", [split_for_sample(index, split_ranges) for index in range(binary.shape[0])])
    frame.insert(0, "time_s", np.arange(binary.shape[0], dtype=np.float64) / fs)
    frame.insert(0, "sample_index", np.arange(binary.shape[0], dtype=np.int32))
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)


def build_duration_statistics(events: pd.DataFrame, by_split: bool = False) -> pd.DataFrame:
    frame = events.copy()
    frame["anomaly_type"] = frame["anomaly_type"].str.replace(" ", "_", regex=False)
    frame["labeled_points"] = frame["duration_samples"].astype(np.int64) * frame["num_traces"].astype(np.int64)
    keys = ["origin", "anomaly_type"] + (["split"] if by_split else [])
    return frame.groupby(keys, sort=False).agg(
        event_count=("event_id", "count"),
        duration_samples_min=("duration_samples", "min"),
        duration_samples_median=("duration_samples", "median"),
        duration_samples_max=("duration_samples", "max"),
        cumulative_duration_samples=("duration_samples", "sum"),
        labeled_point_count=("labeled_points", "sum"),
    ).reset_index()


def plot_overview(
    path: Path,
    augmented: np.ndarray,
    events: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    fs = float(config["sampling_rate_hz"])
    downsample = 10
    view = augmented[::downsample].astype(np.float64)
    center = np.median(view, axis=0)
    mad = np.median(np.abs(view - center[None, :]), axis=0)
    scale = np.where(1.4826 * mad > 1e-8, 1.4826 * mad, 1.0)
    normalized = np.clip((view - center[None, :]) / scale[None, :], -6.0, 6.0).T

    fig, ax = plt.subplots(figsize=(16, 7.2), constrained_layout=True)
    ax.imshow(normalized, aspect="auto", origin="lower", cmap="magma", interpolation="nearest", extent=(0, augmented.shape[0] / fs, 0.5, augmented.shape[1] + 0.5), vmin=-4, vmax=4)
    colors = plt.get_cmap("tab10")
    for class_id_text, label in config["class_map"].items():
        class_id = int(class_id_text)
        if class_id < 2:
            continue
        subset = events[events["class_id"] == class_id]
        x = (subset["start_time_s"].to_numpy() + subset["end_time_s_exclusive"].to_numpy()) / 2.0
        y = (subset["trace_start"].to_numpy() + subset["trace_end"].to_numpy()) / 2.0
        ax.scatter(x, y, s=24, marker="o", facecolors="none", edgecolors=colors((class_id - 2) % 10), linewidths=1.1, label=label)
    for split, (start, _) in config["split_ranges"].items():
        if int(start) > 0:
            ax.axvline(int(start) / fs, color="white", linewidth=0.7, alpha=0.7)
    ax.set_title("Trace-normalized augmented waveform with synthetic event centers")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Trace (1-based)")
    ax.set_xlim(0, augmented.shape[0] / fs)
    ax.set_ylim(0.5, augmented.shape[1] + 0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4, frameon=False)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_examples(
    path: Path,
    original: np.ndarray,
    augmented: np.ndarray,
    events: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    labels = list(config["anomaly_types"])
    fig, axes = plt.subplots(len(labels), 1, figsize=(13, 2.3 * len(labels)), constrained_layout=True)
    fs = float(config["sampling_rate_hz"])
    for ax, anomaly_type in zip(axes, labels):
        row = events[events["anomaly_type"] == anomaly_type].iloc[0]
        trace = int(str(row["trace_ids"]).split(";")[0]) - 1
        start, stop = int(row["start_sample"]), int(row["end_sample_exclusive"])
        margin = max(100, (stop - start) // 2)
        left, right = max(0, start - margin), min(original.shape[0], stop + margin)
        raw = original[left:right, trace].astype(np.float64)
        changed = augmented[left:right, trace].astype(np.float64)
        local_center = np.median(raw)
        local_scale = 1.4826 * np.median(np.abs(raw - local_center))
        local_scale = max(float(local_scale), 1.0)
        time = np.arange(left, right) / fs
        ax.plot(time, (raw - local_center) / local_scale, color="#777777", linewidth=0.8, label="original")
        ax.plot(time, (changed - local_center) / local_scale, color="#C62828", linewidth=1.0, label="augmented")
        ax.axvspan(start / fs, stop / fs, color="#F9A825", alpha=0.18)
        ax.set_title(f"{anomaly_type} — trace {trace + 1}, {start / fs:.2f}–{stop / fs:.2f} s", loc="left", fontsize=10)
        ax.set_ylabel("Robust z")
        ax.grid(alpha=0.15)
    axes[0].legend(loc="upper right", frameon=False)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Representative synthetic anomalies on real seismic backgrounds", fontsize=14)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_quality_report(path: Path, audit: dict[str, Any], config: dict[str, Any]) -> None:
    lines = [
        "# Synthetic anomaly augmentation quality report",
        "",
        f"Overall status: **{audit['status']}**",
        "",
        "## Scope",
        "",
        "Existing human-reviewed natural labels are preserved as class 1. Newly generated labels are deterministic procedural ground truth (classes 2–9); they must not be described as manually verified natural anomalies.",
        "",
        "## Automated checks",
        "",
    ]
    for name, passed in audit["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines.extend([
        "",
        "## Counts",
        "",
        f"- Natural labeled points: {audit['counts']['natural_labeled_points']:,}",
        f"- Synthetic labeled points: {audit['counts']['synthetic_labeled_points']:,}",
        f"- Combined labeled points: {audit['counts']['combined_labeled_points']:,}",
        f"- Synthetic events: {audit['counts']['synthetic_events']:,}",
        "",
        "## Publication notes",
        "",
        "- Report the fixed seed, source SHA-256, configuration SHA-256, class taxonomy, event-level parameter ranges, and temporal split boundaries.",
        "- Keep train/calibration/test temporal roles fixed and exclude both buffer gaps to reduce temporal leakage.",
        "- The exact duplicate groups 101/196 through 105/200 were excluded from injection so augmentation does not create artificial disagreement inside duplicate pairs.",
        "- Synthetic events may overlap protected coherent physical signals because the paper calls for injection on active real backgrounds; every such overlap is explicitly recorded in `synthetic_events.csv`.",
        f"- Time values use {config['sampling_rate_hz']:g} Hz from the current reviewed-label manifest. Verify this sampling rate against acquisition metadata before manuscript submission because the headerless CSV cannot prove it by itself.",
        "- No uncertainty class or uncertainty weighting is present in this release.",
        "- Class 9 strong noise is broadband white-plus-AR(1) additive noise, distinct from class 8 high-frequency jitter. Its added-noise RMS is 2–4 times the local/background reference RMS, with a 5% edge taper and spectral-flatness validation.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(
    path: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    outputs: list[Path],
    original: np.ndarray,
    audit: dict[str, Any],
    config_hash: str,
) -> None:
    manifest = {
        "schema_version": "1.0",
        "release_type": "mixed_natural_and_synthetic_anomaly_benchmark",
        "source_filename": paths["source"].name,
        "source_relative_path": config["source_csv"],
        "source_sha256": config["source_sha256"],
        "source_data_copied_to_datayu": False,
        "shape_samples_by_traces": list(original.shape),
        "sampling_rate_hz": config["sampling_rate_hz"],
        "duration_seconds": original.shape[0] / float(config["sampling_rate_hz"]),
        "random_seed": config["random_seed"],
        "exact_experiment_split_ranges": config["split_ranges"],
        "frozen_injection_sampling_ranges": config.get(
            "frozen_injection_sampling_ranges", config["split_ranges"]
        ),
        "config_sha256": config_hash,
        "program_sha256": sha256_file(Path(__file__).resolve()),
        "class_map": config["class_map"],
        "excluded_duplicate_trace_ids_1_based": config["excluded_trace_ids_1_based"],
        "quality_status": audit["status"],
        "output_files": {
            item.name: {"sha256": sha256_file(item), "size_bytes": item.stat().st_size}
            for item in outputs
            if item.is_file() and item.resolve() != path.resolve()
        },
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(config_path: Path, output_dir: Path, overwrite: bool, preserve_existing: bool = False) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_hash = canonical_sha256(config)
    previous_waveforms = None
    previous_classes = None
    previous_events_text = None
    previous_events = None
    parent_manifest = None
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}. Use --overwrite.")
        if preserve_existing:
            parent_manifest = json.loads((output_dir / "augmentation_manifest.json").read_text(encoding="utf-8"))
            for filename in ("augmented_waveforms.npz", "combined_labels.npz", "synthetic_events.csv"):
                if sha256_file(output_dir / filename) != parent_manifest["output_files"][filename]["sha256"]:
                    raise ValueError(f"Existing release has changed outside its manifest: {filename}")
            with np.load(output_dir / "augmented_waveforms.npz", allow_pickle=False) as archive:
                previous_waveforms = archive["waveforms"].copy()
            with np.load(output_dir / "combined_labels.npz", allow_pickle=False) as archive:
                previous_classes = archive["class_labels"].copy()
            previous_events = pd.read_csv(output_dir / "synthetic_events.csv")
            previous_events_text = (output_dir / "synthetic_events.csv").read_text(encoding="utf-8-sig")
    elif preserve_existing:
        raise ValueError("--preserve-existing requires a nonempty existing release")
    output_dir.mkdir(parents=True, exist_ok=True)

    original, natural, natural_events, protected, paths = load_inputs(config, config_path)
    validate_splits(config["split_ranges"], original.shape[0])
    validate_frozen_sampling_ranges(
        config.get("frozen_injection_sampling_ranges", config["split_ranges"]),
        config["split_ranges"],
        original.shape[0],
    )
    source_hash_before = sha256_file(paths["source"])
    augmented, synthetic_classes, _, synthetic_events = generate_augmentation(original, natural, protected, config)
    synthetic_mask = synthetic_classes > 0
    combined_binary = (natural.astype(bool) | synthetic_mask).astype(np.uint8)
    combined_classes = synthetic_classes.copy()
    combined_classes[natural.astype(bool)] = 1
    combined_events = build_combined_events(natural_events, synthetic_events, config)
    source_hash_after = sha256_file(paths["source"])
    audit = quality_audit(
        original,
        augmented,
        natural,
        synthetic_classes,
        combined_binary,
        combined_classes,
        synthetic_events,
        config,
        source_hash_before,
        source_hash_after,
    )
    if previous_waveforms is not None:
        additional_mask = synthetic_mask & (previous_classes == 0)
        old_ids = set(previous_events["event_id"].astype(str))
        preserved_catalog = synthetic_events[synthetic_events["event_id"].isin(old_ids)].to_csv(index=False, lineterminator="\n")
        extension_checks = {
            "previous_waveforms_unchanged_outside_new_intervals": bool(np.array_equal(augmented[~additional_mask], previous_waveforms[~additional_mask])),
            "previous_nonzero_class_labels_preserved": bool(np.array_equal(combined_classes[previous_classes > 0], previous_classes[previous_classes > 0])),
            "previous_event_catalog_rows_byte_identical": preserved_catalog == previous_events_text,
        }
        audit["checks"].update(extension_checks)
        audit["extension"] = {
            "parent_config_sha256": parent_manifest["config_sha256"],
            "parent_waveform_sha256": parent_manifest["output_files"]["augmented_waveforms.npz"]["sha256"],
            "parent_event_catalog_sha256": parent_manifest["output_files"]["synthetic_events.csv"]["sha256"],
            "previous_synthetic_event_count": len(previous_events),
            "added_synthetic_event_count": len(synthetic_events) - len(previous_events),
            "added_labeled_points": int(additional_mask.sum()),
        }
        audit["status"] = "PASS" if all(audit["checks"].values()) else "FAIL"
    if audit["status"] != "PASS":
        failed = [name for name, passed in audit["checks"].items() if not passed]
        raise RuntimeError(f"Quality audit failed: {failed}")

    fs = float(config["sampling_rate_hz"])
    split_id_map = {name: index for index, name in enumerate(config["split_ranges"].keys())}
    split_ids = np.empty(original.shape[0], dtype=np.uint8)
    for name, (start, stop) in config["split_ranges"].items():
        split_ids[int(start) : int(stop)] = split_id_map[name]

    augmented_path = output_dir / "augmented_waveforms.npz"
    labels_path = output_dir / "combined_labels.npz"
    np.savez_compressed(
        augmented_path,
        waveforms=augmented.astype(np.float32, copy=False),
        sample_index=np.arange(original.shape[0], dtype=np.int32),
        trace_ids=np.arange(1, original.shape[1] + 1, dtype=np.int16),
        sampling_rate_hz=np.array([fs], dtype=np.float64),
        source_sha256=np.array([config["source_sha256"]]),
        config_sha256=np.array([config_hash]),
    )
    np.savez_compressed(
        labels_path,
        binary_labels=combined_binary,
        class_labels=combined_classes,
        natural_reviewed_labels=natural,
        synthetic_class_labels=synthetic_classes,
        split_ids=split_ids,
        split_id_map_json=np.array([json.dumps(split_id_map, separators=(",", ":"))]),
        class_map_json=np.array([json.dumps(config["class_map"], separators=(",", ":"))]),
        sample_index=np.arange(original.shape[0], dtype=np.int32),
        trace_ids=np.arange(1, original.shape[1] + 1, dtype=np.int16),
        sampling_rate_hz=np.array([fs], dtype=np.float64),
        source_sha256=np.array([config["source_sha256"]]),
        config_sha256=np.array([config_hash]),
    )

    synthetic_events_path = output_dir / "synthetic_events.csv"
    combined_events_path = output_dir / "combined_events.csv"
    split_path = output_dir / "split_manifest.csv"
    summary_path = output_dir / "anomaly_type_summary.csv"
    binary_csv_path = output_dir / "combined_binary_labels.csv.gz"
    synthetic_events.to_csv(synthetic_events_path, index=False, encoding="utf-8-sig")
    combined_events.to_csv(combined_events_path, index=False, encoding="utf-8-sig")
    build_split_manifest(config, original.shape[0]).to_csv(split_path, index=False, encoding="utf-8-sig")
    build_type_summary(combined_classes, combined_events, config["class_map"]).to_csv(summary_path, index=False, encoding="utf-8-sig")
    save_binary_labels_csv(binary_csv_path, combined_binary, fs, config["split_ranges"])
    duration_stats_path = output_dir / "anomaly_duration_statistics.csv"
    split_stats_path = output_dir / "anomaly_split_statistics.csv"
    build_duration_statistics(combined_events).to_csv(duration_stats_path, index=False, encoding="utf-8-sig")
    build_duration_statistics(combined_events, by_split=True).to_csv(split_stats_path, index=False, encoding="utf-8-sig")

    from plot_injected_change_figure import draw_figure
    all_overview_path = output_dir / "all_anomaly_overview.png"
    draw_figure(config_path, output_dir, all_overview_path, dpi=300)

    audit_path = output_dir / "quality_audit.json"
    report_path = output_dir / "quality_report.md"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_quality_report(report_path, audit, config)

    manifest_path = output_dir / "augmentation_manifest.json"
    output_files = [
        augmented_path,
        labels_path,
        synthetic_events_path,
        combined_events_path,
        split_path,
        summary_path,
        binary_csv_path,
        audit_path,
        report_path,
        duration_stats_path,
        split_stats_path,
        all_overview_path,
    ]
    write_manifest(manifest_path, config, paths, output_files, original, audit, config_hash)
    if parent_manifest is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["parent_release"] = {
            "config_sha256": parent_manifest["config_sha256"],
            "output_files": parent_manifest["output_files"],
            "preservation_checks": extension_checks,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": audit["status"],
        "output_dir": str(output_dir),
        "synthetic_events": audit["counts"]["synthetic_events"],
        "synthetic_labeled_points": audit["counts"]["synthetic_labeled_points"],
        "combined_labeled_points": audit["counts"]["combined_labeled_points"],
        "manifest": str(manifest_path),
    }, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "anomaly_config.json")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / RESULTS_DIRNAME)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing results directory")
    parser.add_argument("--preserve-existing", action="store_true", help="Verify that extending the configuration preserves every existing event and waveform outside new intervals")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run(args.config.resolve(), args.output_dir.resolve(), args.overwrite, args.preserve_existing)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
