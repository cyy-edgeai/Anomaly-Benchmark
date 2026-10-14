#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training-time augmentation used by the paper experiments.

This module is a model-free extraction of the augmentation logic from the
experiment program.  It contains two deliberately separate stages:

1. ``augment_background_traces`` applies the offline real-background
   augmentation in trace-major layout: integer time shift with edge padding,
   amplitude scaling, Gaussian noise, and the same robust cross-trace cleanup.
2. ``make_self_supervised_view`` applies the online corruption used for every
   training window and returns both the corrupted window and its procedural
   point mask.

The online routine intentionally uses Python's global ``random`` stream for
event decisions and PyTorch's random stream for tensors, exactly as in the
experiment program.  Call ``seed_all`` before use when deterministic replay is
required.  PyTorch is imported lazily so the dataset-release validator keeps
its lightweight NumPy-only installation path.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableSequence, Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


EPS = 1.0e-9
DEFAULT_CONFIG_PATH = Path(__file__).with_name("training_augmentation_config.json")
ALL_PROXY_ANOMALY_TYPES = (
    "pulse",
    "burst",
    "step",
    "flatline",
    "drift",
    "dropout",
    "jitter",
    "clipping",
    "gain_jump",
    "strong_noise",
)


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Online self-supervised augmentation requires PyTorch. Install "
            "requirements-training-augmentation.txt in the experiment environment."
        ) from exc
    return torch


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load and minimally validate the published augmentation configuration."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {"randomness", "background_augmentation", "online_self_supervised"}
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing configuration section(s): {', '.join(missing)}")
    return config


def seed_all(seed: int) -> None:
    """Seed the Python and PyTorch streams used by the original implementation."""
    torch = _require_torch()
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(float(value) for value in values)
    q = clamp(q, 0.0, 1.0)
    position = q * (len(vals) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return vals[low]
    fraction = position - low
    return vals[low] * (1.0 - fraction) + vals[high] * fraction


def median(values: Sequence[float]) -> float:
    return quantile(values, 0.5)


def mad_scale(values: Sequence[float], center: Optional[float] = None) -> float:
    if not values:
        return 1.0
    if center is None:
        center = median(values)
    return max(1.4826 * median([abs(float(value) - center) for value in values]), EPS)


def row_center_scale(
    values: List[float], robust: bool = True, minimum_scale: float = 0.05
) -> Tuple[float, float]:
    if robust:
        center = median(values)
        scale = mad_scale(values, center)
    else:
        center = sum(values) / max(1, len(values))
        variance = sum((value - center) ** 2 for value in values) / max(1, len(values) - 1)
        scale = math.sqrt(max(variance, EPS))
    return center, max(scale, float(minimum_scale))


def time_shift(values: Sequence[float], shift: int) -> array:
    """Shift a trace by integer samples and replicate the boundary values."""
    n = len(values)
    output = array("f", [0.0]) * n
    for index in range(n):
        source = index - int(shift)
        if 0 <= source < n:
            output[index] = float(values[source])
        elif source < 0:
            output[index] = float(values[0])
        else:
            output[index] = float(values[-1])
    return output


def clean_robust_background(
    traces: MutableSequence[array],
    radius: int = 4,
    residual_clip: float = 0.25,
    passes: int = 4,
    minimum_scale: float = 0.05,
) -> None:
    """Apply the experiment program's in-place cross-trace robust cleanup."""
    n_traces = len(traces)
    n_samples = len(traces[0]) if traces else 0
    if n_traces < 3:
        return
    if any(len(trace) != n_samples for trace in traces):
        raise ValueError("All traces must have the same number of samples.")
    for _ in range(int(passes)):
        cleaned = [array("f", [0.0]) * n_samples for _ in range(n_traces)]
        for sample_index in range(n_samples):
            row = [float(traces[trace_index][sample_index]) for trace_index in range(n_traces)]
            global_center, global_scale = row_center_scale(
                row, minimum_scale=minimum_scale
            )
            for trace_index, value in enumerate(row):
                low = max(0, trace_index - int(radius))
                high = min(n_traces, trace_index + int(radius) + 1)
                neighbours = [row[index] for index in range(low, high) if index != trace_index]
                center, scale = (
                    row_center_scale(neighbours, minimum_scale=minimum_scale)
                    if neighbours
                    else (global_center, global_scale)
                )
                residual = clamp(
                    value - center,
                    -float(residual_clip) * scale,
                    float(residual_clip) * scale,
                )
                cleaned[trace_index][sample_index] = center + residual
        for trace_index in range(n_traces):
            traces[trace_index] = cleaned[trace_index]


@dataclass(frozen=True)
class BackgroundAugmentationRecord:
    trace_index: int
    shift_samples: int
    amplitude_scale: float
    gaussian_noise_mean: float
    gaussian_noise_std: float


def augment_background_traces(
    traces: Sequence[Sequence[float]],
    config: Mapping[str, Any],
    *,
    seed: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[List[array], List[BackgroundAugmentationRecord]]:
    """Apply the exact background augmentation loop to trace-major data.

    The returned traces use ``array('f')`` because the original program stores
    every assignment as IEEE-754 float32.  Pass an existing ``random.Random``
    instance when this stage must continue a caller-owned random stream.
    """
    if seed is not None and rng is not None:
        raise ValueError("Provide either seed or rng, not both.")
    settings = config["background_augmentation"]
    shift_min, shift_max = (int(value) for value in settings["time_shift_samples"])
    scale_min, scale_max = (float(value) for value in settings["amplitude_scale"])
    noise_mean = float(settings["gaussian_noise"]["mean"])
    noise_min, noise_max = (
        float(value) for value in settings["gaussian_noise"]["std_range"]
    )
    if noise_mean != 0.0:
        raise ValueError("The experiment implementation requires zero-mean background noise.")
    if rng is None:
        if seed is None:
            seed = int(config["randomness"]["default_seed"])
        rng = random.Random(int(seed))

    augmented: List[array] = []
    records: List[BackgroundAugmentationRecord] = []
    expected_length = len(traces[0]) if traces else 0
    for trace_index, trace in enumerate(traces):
        if len(trace) != expected_length:
            raise ValueError("All traces must have the same number of samples.")
        shift = rng.randint(shift_min, shift_max)
        shifted = time_shift(trace, shift)
        scale = rng.uniform(scale_min, scale_max)
        noise_std = rng.uniform(noise_min, noise_max)
        for sample_index in range(len(shifted)):
            shifted[sample_index] = (
                float(shifted[sample_index]) * scale + rng.gauss(noise_mean, noise_std)
            )
        augmented.append(shifted)
        records.append(
            BackgroundAugmentationRecord(
                trace_index=trace_index,
                shift_samples=shift,
                amplitude_scale=scale,
                gaussian_noise_mean=noise_mean,
                gaussian_noise_std=noise_std,
            )
        )

    cleanup = settings["robust_cross_trace_cleanup"]
    if bool(cleanup["enabled"]):
        clean_robust_background(
            augmented,
            radius=int(cleanup["radius"]),
            residual_clip=float(cleanup["residual_clip"]),
            passes=int(cleanup["passes"]),
            minimum_scale=float(cleanup["minimum_scale"]),
        )
    return augmented, records


def make_self_supervised_view(
    x: "torch.Tensor",
    config: Mapping[str, Any],
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Create online procedural corruptions from unlabeled real windows.

    ``x`` must have shape ``[batch, time]``.  Returned tensors have the same
    shape, device and dtype.  The function mirrors the experiment code's RNG
    call order; changing the anomaly-type order changes deterministic replay.
    """
    torch = _require_torch()
    settings = config["online_self_supervised"]
    noisy = x.clone()
    mask = torch.zeros_like(x)
    base_noise_std = float(settings["base_gaussian_noise_std"])
    if base_noise_std > 0:
        noisy = noisy + torch.randn_like(noisy) * base_noise_std
    batch_size, n = noisy.shape
    if n < int(settings["minimum_window_samples"]):
        return noisy, mask

    position_fraction = settings["event_position_fraction"]
    low = max(2, int(n * float(position_fraction[0])))
    high = max(low + 1, int(n * float(position_fraction[1])))
    kinds = list(settings["anomaly_types"])
    unknown = sorted(set(kinds).difference(ALL_PROXY_ANOMALY_TYPES))
    if unknown:
        raise ValueError(f"Unknown online anomaly type(s): {', '.join(unknown)}")
    if not kinds:
        kinds = list(ALL_PROXY_ANOMALY_TYPES)
    parameters = settings["anomaly_parameters"]
    event_min = int(settings["events_per_window"][0])
    event_max = int(settings["events_per_window"][1])
    bins_per_event = int(settings["position_bins_per_event"])

    for batch_index in range(batch_size):
        if random.random() > float(settings["augmentation_probability"]):
            continue
        event_count = random.randint(event_min, max(event_min, event_max))
        bins = list(range(event_count * bins_per_event))
        random.shuffle(bins)
        positions = []
        for bin_id in bins[:event_count]:
            denominator = max(1, event_count * bins_per_event)
            left = low + int((high - low) * bin_id / denominator)
            right = low + int((high - low) * (bin_id + 1) / denominator)
            positions.append(random.randint(left, max(left, right - 1)))
        positions.sort()

        for position in positions:
            kind = random.choice(kinds)
            # The original program draws sign for every kind, including kinds
            # that do not subsequently use it.  Preserve that RNG call.
            sign = -1.0 if random.random() < 0.5 else 1.0
            parameter = parameters[kind]
            length_range = parameter["length_samples"]
            length = random.randint(int(length_range[0]), int(length_range[1]))
            if kind in {"pulse", "burst", "step", "flatline", "drift"}:
                amplitude_range = parameter["amplitude_abs"]
                amplitude = sign * random.uniform(
                    float(amplitude_range[0]), float(amplitude_range[1])
                )
            elif kind == "dropout":
                amplitude = 0.0
            else:
                amplitude_range = parameter["amplitude"]
                amplitude = random.uniform(
                    float(amplitude_range[0]), float(amplitude_range[1])
                )

            end = min(n, position + length)
            if end <= position:
                continue
            segment_length = end - position
            if kind == "flatline":
                anchor = noisy[batch_index, max(0, position - 1)].detach() + amplitude
                jitter = max(
                    float(parameter["jitter_std_floor"]),
                    abs(amplitude) * float(parameter["jitter_std_amplitude_fraction"]),
                )
                noisy[batch_index, position:end] = anchor + torch.randn(
                    segment_length, device=x.device, dtype=x.dtype
                ) * jitter
            elif kind == "dropout":
                jitter = float(parameter["jitter_std"])
                anchor = torch.zeros((), device=x.device, dtype=x.dtype) + torch.randn(
                    (), device=x.device, dtype=x.dtype
                ) * jitter
                noisy[batch_index, position:end] = anchor + torch.randn(
                    segment_length, device=x.device, dtype=x.dtype
                ) * jitter
            elif kind == "jitter":
                noisy[batch_index, position:end] = noisy[batch_index, position:end] + torch.randn(
                    segment_length, device=x.device, dtype=x.dtype
                ) * amplitude
            elif kind == "clipping":
                center = noisy[batch_index, position:end].median().detach()
                noisy[batch_index, position:end] = torch.clamp(
                    noisy[batch_index, position:end], center - amplitude, center + amplitude
                )
            elif kind == "gain_jump":
                gain = (
                    amplitude
                    if random.random() < float(parameter["amplification_probability"])
                    else random.uniform(
                        float(parameter["attenuation_gain"][0]),
                        float(parameter["attenuation_gain"][1]),
                    )
                )
                noisy[batch_index, position:end] = noisy[batch_index, position:end] * gain
            elif kind == "strong_noise":
                white = torch.randn(segment_length, device=x.device, dtype=x.dtype) * amplitude
                if segment_length > 1:
                    white[1:] = (
                        float(parameter["previous_sample_weight"]) * white[:-1]
                        + float(parameter["current_sample_weight"]) * white[1:]
                    )
                noisy[batch_index, position:end] = noisy[batch_index, position:end] + white
            elif kind == "drift":
                ramp = torch.linspace(
                    float(parameter["ramp"][0]),
                    float(parameter["ramp"][1]),
                    segment_length,
                    device=x.device,
                    dtype=x.dtype,
                )
                noisy[batch_index, position:end] = noisy[batch_index, position:end] + amplitude * ramp
            elif kind == "burst":
                noisy[batch_index, position:end] = (
                    noisy[batch_index, position:end]
                    + amplitude
                    + torch.randn(segment_length, device=x.device, dtype=x.dtype)
                    * abs(amplitude)
                    * float(parameter["noise_std_amplitude_fraction"])
                )
            else:
                noisy[batch_index, position:end] = noisy[batch_index, position:end] + amplitude
            mask[batch_index, position:end] = 1.0
    return noisy, mask


def _self_test(config: Mapping[str, Any]) -> Dict[str, Any]:
    torch = _require_torch()
    seed = int(config["randomness"]["default_seed"])
    x = torch.linspace(-1.0, 1.0, 3 * 512, dtype=torch.float32).reshape(3, 512)
    seed_all(seed)
    first_view, first_mask = make_self_supervised_view(x, config)
    seed_all(seed)
    second_view, second_mask = make_self_supervised_view(x, config)
    if not torch.equal(first_view, second_view) or not torch.equal(first_mask, second_mask):
        raise AssertionError("Online augmentation is not deterministic after reseeding.")

    traces = [array("f", (float(index + trace) for index in range(32))) for trace in range(5)]
    first_background, first_records = augment_background_traces(traces, config, seed=seed)
    second_background, second_records = augment_background_traces(traces, config, seed=seed)
    if first_records != second_records or any(
        list(first) != list(second) for first, second in zip(first_background, second_background)
    ):
        raise AssertionError("Background augmentation is not deterministic after reseeding.")
    return {
        "status": "PASS",
        "seed": seed,
        "online_masked_points": int(first_mask.sum().item()),
        "online_shape": list(first_view.shape),
        "background_shape_trace_major": [len(first_background), len(first_background[0])],
        "torch_version": torch.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="verify deterministic replay for both augmentation stages",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="print the resolved JSON configuration",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.print_config:
        print(json.dumps(config, ensure_ascii=False, indent=2))
    if args.self_test:
        print(json.dumps(_self_test(config), ensure_ascii=False, indent=2))
    if not args.self_test and not args.print_config:
        parser.print_help()


if __name__ == "__main__":
    main()
