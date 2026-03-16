"""Simulate and plot incremental capacity (IC) curves for five aging states.

The script creates synthetic constant-current discharge data, computes:
    Q = I * t
    IC = dQ/dV ≈ DeltaQ/DeltaV
and compares raw IC curves against moving-average filtered IC curves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Reproducibility and simulation controls
RANDOM_SEED = 42
CURRENT_A = 2.0
NOMINAL_CAPACITY_AH = 2.0
N_SAMPLES = 1600
V_MAX = 4.2
V_MIN = 2.7

# IC calculation controls
DELTA_V = 0.005

# Synthetic measurement noise
VOLTAGE_NOISE_STD = 0.0035
CURRENT_NOISE_STD = 0.015

# Moving average smoothing (MASF-style)
PRIMARY_FILTER_WINDOW = 11
FILTER_WINDOW_CANDIDATES = (5, 11, 21)

# label, capacity_scale, age_factor
AGING_STATES = [
    ("State 1 - Fresh", 1.00, 0.00),
    ("State 2 - Light Aging", 0.94, 0.25),
    ("State 3 - Mid Aging", 0.88, 0.50),
    ("State 4 - Heavy Aging", 0.82, 0.75),
    ("State 5 - End-of-Life Trend", 0.76, 1.00),
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    """Apply simple moving-average smoothing."""
    if window <= 1:
        return signal.copy()
    if window % 2 == 0:
        raise ValueError("Moving-average window must be an odd integer.")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(signal, kernel, mode="same")


def simulate_discharge_cycle(
    capacity_scale: float,
    age_factor: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic time, current, capacity and voltage for one discharge cycle."""
    capacity_ah = NOMINAL_CAPACITY_AH * capacity_scale
    discharge_time_s = (capacity_ah / CURRENT_A) * 3600.0

    time_s = np.linspace(0.0, discharge_time_s, N_SAMPLES)

    current_measured = CURRENT_A + rng.normal(0.0, CURRENT_NOISE_STD, size=N_SAMPLES)
    dt = np.diff(time_s, prepend=time_s[0])
    capacity_ah_trace = np.cumsum(current_measured * dt) / 3600.0
    capacity_ah_trace = np.clip(capacity_ah_trace, 0.0, capacity_ah)
    q_frac = capacity_ah_trace / max(capacity_ah, 1e-9)

    width_scale = 1.0 + 0.9 * age_factor
    knee1_center = 0.18 + 0.04 * age_factor
    knee2_center = 0.82 - 0.03 * age_factor

    knee1 = 0.35 * _sigmoid((q_frac - knee1_center) / (0.020 * width_scale))
    knee2 = 0.85 * _sigmoid((q_frac - knee2_center) / (0.030 * width_scale))
    slope = 0.22 * q_frac + 0.06 * q_frac**2
    ohmic_drop = CURRENT_A * (0.030 + 0.040 * age_factor)

    voltage_clean = V_MAX - slope - knee1 - knee2 - ohmic_drop
    voltage_clean = np.clip(voltage_clean, V_MIN, V_MAX)
    voltage_clean = np.minimum.accumulate(voltage_clean)

    voltage_measured = voltage_clean + rng.normal(0.0, VOLTAGE_NOISE_STD, size=N_SAMPLES)
    return time_s, current_measured, capacity_ah_trace, voltage_measured


def compute_ic_curve(
    voltage_v: np.ndarray,
    capacity_ah: np.ndarray,
    delta_v: float = DELTA_V,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute IC curve using fixed-voltage-interval finite differences.

    IC is reported as -dQ/dV so discharge curves are displayed with positive peaks.
    """
    mask = np.isfinite(voltage_v) & np.isfinite(capacity_ah)
    v = voltage_v[mask]
    q = capacity_ah[mask]

    if v.size < 5:
        return np.array([]), np.array([])

    v_monotonic = np.minimum.accumulate(v)

    v_asc = v_monotonic[::-1]
    q_asc = q[::-1]

    v_unique, unique_idx = np.unique(v_asc, return_index=True)
    q_unique = q_asc[unique_idx]

    v_low = max(V_MIN, float(v_unique.min()))
    v_high = min(V_MAX, float(v_unique.max()))
    if (v_high - v_low) < (2.0 * delta_v):
        return np.array([]), np.array([])

    v_grid = np.arange(v_low, v_high + delta_v, delta_v)
    q_grid = np.interp(v_grid, v_unique, q_unique)

    dq = np.diff(q_grid)
    dv = np.diff(v_grid)
    ic = -(dq / np.maximum(dv, 1e-12))
    v_mid = 0.5 * (v_grid[1:] + v_grid[:-1])
    return v_mid, ic


def build_state_result(
    label: str,
    capacity_scale: float,
    age_factor: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray | str | float]:
    """Create one state with discharge traces and IC curves."""
    time_s, current_a, capacity_ah, voltage_v = simulate_discharge_cycle(
        capacity_scale, age_factor, rng
    )
    voltage_ic, ic_raw = compute_ic_curve(voltage_v, capacity_ah)
    ic_filtered = moving_average(ic_raw, PRIMARY_FILTER_WINDOW) if ic_raw.size else ic_raw

    return {
        "label": label,
        "time_s": time_s,
        "current_a": current_a,
        "capacity_ah": capacity_ah,
        "voltage_v": voltage_v,
        "voltage_ic": voltage_ic,
        "ic_raw": ic_raw,
        "ic_filtered": ic_filtered,
        "capacity_end_ah": float(capacity_ah[-1]),
    }


def make_plots(results: list[dict[str, np.ndarray | str | float]], output_dir: Path) -> None:
    """Plot V-Q curves, raw IC curves, filtered IC curves, and filter-width sensitivity."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_vq, ax_vq = plt.subplots(figsize=(10, 5.5))
    for result in results:
        ax_vq.plot(
            result["capacity_ah"],
            result["voltage_v"],
            linewidth=1.8,
            label=result["label"],
        )
    ax_vq.set_title("Simulated Discharge V-Q Curves (5 Aging States)")
    ax_vq.set_xlabel("Capacity Q (Ah)")
    ax_vq.set_ylabel("Voltage V (V)")
    ax_vq.grid(alpha=0.3)
    ax_vq.legend(loc="best")
    fig_vq.tight_layout()
    fig_vq.savefig(output_dir / "simulated_vq_curves.png", dpi=220)

    fig_ic, axes_ic = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)

    for result in results:
        axes_ic[0].plot(
            result["voltage_ic"],
            result["ic_raw"],
            linewidth=1.3,
            alpha=0.9,
            label=result["label"],
        )
        axes_ic[1].plot(
            result["voltage_ic"],
            result["ic_filtered"],
            linewidth=2.0,
            alpha=0.95,
            label=result["label"],
        )

    axes_ic[0].set_title("Raw IC Curves (Noisy)")
    axes_ic[1].set_title(f"Filtered IC Curves (MA Window={PRIMARY_FILTER_WINDOW})")
    axes_ic[0].set_xlabel("Voltage V (V)")
    axes_ic[1].set_xlabel("Voltage V (V)")
    axes_ic[0].set_ylabel("Incremental Capacity IC = -dQ/dV (Ah/V)")

    for ax in axes_ic:
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig_ic.tight_layout()
    fig_ic.savefig(output_dir / "simulated_ic_raw_vs_filtered.png", dpi=220)

    mid_state = results[len(results) // 2]
    fig_filter, ax_filter = plt.subplots(figsize=(10, 5.5))
    ax_filter.plot(
        mid_state["voltage_ic"],
        mid_state["ic_raw"],
        color="gray",
        alpha=0.35,
        linewidth=1.2,
        label="Raw",
    )
    for window in FILTER_WINDOW_CANDIDATES:
        ax_filter.plot(
            mid_state["voltage_ic"],
            moving_average(mid_state["ic_raw"], window),
            linewidth=1.8,
            label=f"MA window = {window}",
        )

    ax_filter.set_title("Filter-Width Sensitivity (Mid-Aging State)")
    ax_filter.set_xlabel("Voltage V (V)")
    ax_filter.set_ylabel("Incremental Capacity IC = -dQ/dV (Ah/V)")
    ax_filter.grid(alpha=0.3)
    ax_filter.legend(loc="best")
    fig_filter.tight_layout()
    fig_filter.savefig(output_dir / "simulated_ic_filter_width_sensitivity.png", dpi=220)


def main(show_plots: bool = False) -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    output_dir = Path(__file__).resolve().parent / "demo_outputs"

    results = [
        build_state_result(label, cap_scale, age_factor, rng)
        for label, cap_scale, age_factor in AGING_STATES
    ]

    make_plots(results, output_dir)

    print("Generated simulated IC demo plots:")
    print(f"  - {output_dir / 'simulated_vq_curves.png'}")
    print(f"  - {output_dir / 'simulated_ic_raw_vs_filtered.png'}")
    print(f"  - {output_dir / 'simulated_ic_filter_width_sensitivity.png'}")
    print("\nEnd-of-discharge capacities (Ah):")
    for result in results:
        print(f"  - {result['label']}: {result['capacity_end_ah']:.3f}")

    if show_plots:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate and plot IC curves for five battery aging states."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving PNG files.",
    )
    args = parser.parse_args()
    main(show_plots=args.show)
