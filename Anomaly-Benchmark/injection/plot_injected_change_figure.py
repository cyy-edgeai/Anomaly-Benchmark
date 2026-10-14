#!/usr/bin/env python3
"""Draw a publication-ready overview of all natural and injected anomalies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 18,
    "axes.titlesize": 21,
    "axes.labelsize": 19,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 16,
})
COLORS = {
    1: "#FFFFFF",  # human-reviewed natural anomaly
    2: "#00B5EB",  # step
    3: "#FF7F0E",  # zero dropout
    4: "#20C997",  # flatline
    5: "#FF4D4D",  # drift
    6: "#B388FF",  # saturation clipping
    7: "#F4A261",  # gain jump
    8: "#FF66C4",  # high-frequency jitter
    9: "#F1C40F",  # broadband strong noise
}


def robust_energy_view(
    original: np.ndarray,
    augmented: np.ndarray,
    block: int,
) -> np.ndarray:
    """Reproduce the original figure's block-maximum robust-energy display."""
    center = np.median(original, axis=0, keepdims=True)
    mad = np.median(np.abs(original - center), axis=0, keepdims=True)
    scale = np.maximum(1.4826 * mad, 1.0)
    robust_z = ((augmented - center) / scale).astype(np.float32)
    block_count = robust_z.shape[0] // block
    energy = np.max(
        np.abs(robust_z[: block_count * block].reshape(block_count, block, robust_z.shape[1])),
        axis=1,
    ).T
    return np.arcsinh(energy)


def draw_figure(config_path: Path, results_dir: Path, output: Path, dpi: int) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = Path(config["source_csv"])
    if not source.is_absolute():
        source = (config_path.parent / source).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source CSV not found: {source}")

    original = pd.read_csv(source, header=None, dtype=np.float32).to_numpy(copy=False)
    augmented_archive = np.load(results_dir / "augmented_waveforms.npz", allow_pickle=False)
    labels_archive = np.load(results_dir / "combined_labels.npz", allow_pickle=False)
    augmented = augmented_archive["waveforms"]
    combined_classes = labels_archive["class_labels"]
    events = pd.read_csv(results_dir / "combined_events.csv")
    if original.shape != augmented.shape or combined_classes.shape != augmented.shape:
        raise ValueError("Source, augmented waveform, and label shapes do not match")

    fs = float(config["sampling_rate_hz"])
    block = 10
    duration_s = augmented.shape[0] / fs
    energy_view = robust_energy_view(original, augmented, block=block)

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 18,
        "axes.titlesize": 21,
        "axes.labelsize": 19,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 20,
    })
    fig = plt.figure(figsize=(18, 9.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.0], hspace=0.04)
    ax_wave = fig.add_subplot(grid[0])
    ax_count = fig.add_subplot(grid[1], sharex=None)

    ax_wave.imshow(
        energy_view,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=(0.0, energy_view.shape[1] * block / fs, 1, augmented.shape[1]),
        vmin=0.0,
        vmax=np.arcsinh(20.0),
        rasterized=True,
    )

    # A dark underlay keeps short intervals visible on bright waveform regions.
    for row in events.itertuples(index=False):
        traces = [int(value) for value in str(row.trace_ids).split(";")]
        class_id = int(row.class_id)
        color = COLORS[class_id]
        for trace in traces:
            ax_wave.hlines(trace, row.start_time_s, row.end_time_s_exclusive, color="#1A1A1A", linewidth=3.2, alpha=0.9, zorder=3)
            ax_wave.hlines(trace, row.start_time_s, row.end_time_s_exclusive, color=color, linewidth=1.8, alpha=1.0, zorder=4)
            midpoint = 0.5 * (row.start_time_s + row.end_time_s_exclusive)
            marker = "D" if class_id == 1 else "o"
            edgecolor = "#111111" if class_id == 1 else "white"
            size = 20 if class_id == 1 else 13
            ax_wave.scatter(midpoint, trace, s=size, marker=marker, facecolor=color, edgecolor=edgecolor, linewidth=0.55, zorder=5)

    class_map = {int(key): value for key, value in config["class_map"].items()}
    class_ids = sorted(key for key in class_map if key > 0)
    handles = [
        Line2D(
            [0], [0],
            color="#222222" if class_id == 1 else COLORS[class_id],
            marker="D" if class_id == 1 else "o",
            markerfacecolor=COLORS[class_id],
            markeredgecolor="#222222" if class_id == 1 else "white",
            linewidth=2.0,
            markersize=5,
            label=class_map[class_id],
        )
        for class_id in class_ids
    ]
    ax_wave.set_title("Trace-normalized augmented waveform context with all final anomaly labels")
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_ylabel("Trace (1-based)")
    ax_wave.set_xlim(0.0, duration_s)
    ax_wave.set_ylim(1, augmented.shape[1])
    ax_wave.set_xticks(np.arange(0, duration_s + 0.1, 100.0))
    ax_wave.set_yticks(np.arange(25, augmented.shape[1] + 1, 25))
    ax_wave.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.105), ncol=5, frameon=False, columnspacing=1.5)

    trace_ids = np.arange(1, augmented.shape[1] + 1)
    bottoms = np.zeros(augmented.shape[1], dtype=np.int64)
    for class_id in (key for key in class_ids if key > 1):
        counts = np.sum(combined_classes == class_id, axis=0, dtype=np.int64)
        ax_count.bar(
            trace_ids,
            counts,
            bottom=bottoms,
            width=0.86,
            color=COLORS[class_id],
            edgecolor="none",
            label=class_map[class_id],
        )
        bottoms += counts

    natural_counts = np.sum(combined_classes == 1, axis=0, dtype=np.int64)
    ax_natural = ax_count.twinx()
    nonzero = natural_counts > 0
    ax_natural.vlines(trace_ids[nonzero], 0, natural_counts[nonzero], color="#222222", linewidth=0.8, alpha=0.75, zorder=5)
    ax_natural.scatter(
        trace_ids[nonzero],
        natural_counts[nonzero],
        s=19,
        marker="D",
        facecolor="white",
        edgecolor="#222222",
        linewidth=0.7,
        zorder=6,
    )
    ax_natural.set_ylabel("Natural labeled samples")
    ax_natural.set_ylim(0, max(5.0, float(natural_counts.max()) * 1.14))
    ax_natural.grid(False)

    ax_count.set_title("All anomaly-label counts by trace (bars: injected; diamonds: reviewed natural)")
    ax_count.set_xlabel("Trace (1-based)")
    ax_count.set_ylabel("Injected labeled samples")
    ax_count.set_xlim(0.5, augmented.shape[1] + 0.5)
    ax_count.set_xticks(np.arange(0, augmented.shape[1] + 1, 25))
    ax_count.grid(axis="y", color="#D0D0D0", linewidth=0.6, alpha=0.65)
    ax_count.set_axisbelow(True)
    ax_count.text(
        0.995,
        0.94,
        f"{len(events)} events; {int(np.sum(combined_classes > 0)):,} labeled points\n"
        f"{int(np.sum(events['class_id'] == 1))} reviewed natural + {int(np.sum(events['class_id'] > 1))} injected",
        transform=ax_count.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        color="#333333",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "anomaly_config.json")
    parser.add_argument("--results-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "results" / "all_anomaly_overview.png")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    draw_figure(args.config.resolve(), args.results_dir.resolve(), args.output.resolve(), args.dpi)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
