"""Plot incremental capacity (IC) curves directly from NASA battery MAT data.

This script reads one battery MAT file, extracts discharge cycles, computes
capacity from Q = I * t (using measured current/time), and plots:
1) V-Q curves for representative cycles,
2) raw IC curves,
3) filtered IC curves,
4) capacity trend across all discharge cycles.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io


DEFAULT_MAT_FILE = Path("1. BatteryAgingARC-FY08Q4/B0005.mat")
DEFAULT_NUM_CYCLES = 5
DEFAULT_DELTA_V = 0.005
DEFAULT_FILTER_WINDOW = 11


def moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    """Apply simple moving-average smoothing."""
    if window <= 1:
        return signal.copy()
    if window % 2 == 0:
        raise ValueError("Filter window must be an odd integer.")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(signal, kernel, mode="same")


def load_battery_struct(mat_file: Path, battery_key: str | None = None):
    """Load battery struct from a NASA MAT file."""
    mat = scipy.io.loadmat(mat_file, squeeze_me=True, struct_as_record=False)
    keys = [k for k in mat.keys() if not k.startswith("__")]

    if battery_key is None:
        stem = mat_file.stem
        if stem in mat:
            battery_key = stem
        elif len(keys) == 1:
            battery_key = keys[0]
        else:
            raise ValueError(
                "Could not infer battery key. Please pass --battery-key. "
                f"Available keys: {keys}"
            )

    if battery_key not in mat:
        raise KeyError(f"Battery key '{battery_key}' not found in MAT file.")
    return mat[battery_key], battery_key


def extract_discharge_cycles(battery_struct) -> list[dict[str, np.ndarray | float | int]]:
    """Extract discharge cycle arrays and derived capacity traces."""
    raw_cycles = np.atleast_1d(battery_struct.cycle)
    discharge_cycles: list[dict[str, np.ndarray | float | int]] = []

    discharge_idx = 0
    for global_idx, cycle in enumerate(raw_cycles):
        cycle_type = str(getattr(cycle, "type", "")).strip().lower()
        if cycle_type != "discharge":
            continue

        data = getattr(cycle, "data", None)
        if data is None:
            continue

        voltage_v = np.asarray(getattr(data, "Voltage_measured", np.array([])), dtype=float).ravel()
        current_a = np.asarray(getattr(data, "Current_measured", np.array([])), dtype=float).ravel()
        time_s = np.asarray(getattr(data, "Time", np.array([])), dtype=float).ravel()

        length = int(min(voltage_v.size, current_a.size, time_s.size))
        if length < 5:
            continue

        voltage_v = voltage_v[:length]
        current_a = current_a[:length]
        time_s = time_s[:length]

        order = np.argsort(time_s)
        voltage_v = voltage_v[order]
        current_a = current_a[order]
        time_s = time_s[order]

        dt = np.diff(time_s, prepend=time_s[0])
        dt = np.clip(dt, 0.0, None)

        # In this dataset, discharge current is typically negative.
        discharge_current_a = np.maximum(-current_a, 0.0)
        capacity_ah = np.cumsum(discharge_current_a * dt) / 3600.0

        capacity_field = float(getattr(data, "Capacity", np.nan))
        discharge_cycles.append(
            {
                "global_index": int(global_idx),
                "discharge_index": int(discharge_idx),
                "time_s": time_s,
                "voltage_v": voltage_v,
                "current_a": current_a,
                "capacity_ah": capacity_ah,
                "capacity_end_ah": float(capacity_ah[-1]),
                "capacity_field_ah": capacity_field,
            }
        )
        discharge_idx += 1

    return discharge_cycles


def pick_representative_indices(total_count: int, n_select: int) -> list[int]:
    """Pick evenly spaced cycle indices from start to end of life."""
    if total_count <= 0:
        return []

    n = max(1, min(n_select, total_count))
    raw_idx = [int(round(x)) for x in np.linspace(0, total_count - 1, n)]

    selected: list[int] = []
    for idx in raw_idx:
        if idx not in selected:
            selected.append(idx)

    if len(selected) < n:
        for idx in range(total_count):
            if idx not in selected:
                selected.append(idx)
            if len(selected) == n:
                break

    return selected


def compute_ic_curve(
    voltage_v: np.ndarray,
    capacity_ah: np.ndarray,
    delta_v: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute IC curve on a fixed voltage grid using finite differences.

    Reports IC as -dQ/dV so discharge peaks are positive.
    """
    valid = np.isfinite(voltage_v) & np.isfinite(capacity_ah)
    v = voltage_v[valid]
    q = capacity_ah[valid]

    if v.size < 5:
        return np.array([]), np.array([])

    v_monotonic = np.minimum.accumulate(v)

    v_asc = v_monotonic[::-1]
    q_asc = q[::-1]
    v_unique, idx_unique = np.unique(v_asc, return_index=True)
    q_unique = q_asc[idx_unique]

    if v_unique.size < 5:
        return np.array([]), np.array([])

    v_low = float(v_unique.min())
    v_high = float(v_unique.max())
    if (v_high - v_low) < (2.0 * delta_v):
        return np.array([]), np.array([])

    v_grid = np.arange(v_low, v_high + delta_v, delta_v)
    q_grid = np.interp(v_grid, v_unique, q_unique)

    dq = np.diff(q_grid)
    dv = np.diff(v_grid)
    ic = -(dq / np.maximum(dv, 1e-12))
    v_mid = 0.5 * (v_grid[1:] + v_grid[:-1])
    return v_mid, ic


def plot_results(
    battery_key: str,
    all_discharge_cycles: list[dict[str, np.ndarray | float | int]],
    selected_indices: list[int],
    delta_v: float,
    filter_window: int,
    output_dir: Path,
) -> None:
    """Create and save V-Q, IC, and capacity-trend plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_cycles = [all_discharge_cycles[i] for i in selected_indices]
    colors = plt.cm.plasma(np.linspace(0.15, 0.90, len(selected_cycles)))

    fig_vq, ax_vq = plt.subplots(figsize=(10, 5.5))
    for color, cycle in zip(colors, selected_cycles):
        label = (
            f"Discharge #{int(cycle['discharge_index']) + 1} "
            f"(Q_end={float(cycle['capacity_end_ah']):.3f} Ah)"
        )
        ax_vq.plot(cycle["capacity_ah"], cycle["voltage_v"], color=color, linewidth=1.8, label=label)
    ax_vq.set_title(f"{battery_key}: V-Q Curves for Representative Discharge Cycles")
    ax_vq.set_xlabel("Capacity Q (Ah)")
    ax_vq.set_ylabel("Voltage V (V)")
    ax_vq.grid(alpha=0.3)
    ax_vq.legend(loc="best", fontsize=8)
    fig_vq.tight_layout()
    fig_vq.savefig(output_dir / f"{battery_key}_vq_representative_cycles.png", dpi=220)

    fig_ic, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for color, cycle in zip(colors, selected_cycles):
        v_ic, ic_raw = compute_ic_curve(cycle["voltage_v"], cycle["capacity_ah"], delta_v)
        ic_filtered = moving_average(ic_raw, filter_window) if ic_raw.size else ic_raw

        label = f"Discharge #{int(cycle['discharge_index']) + 1}"
        axes[0].plot(v_ic, ic_raw, color=color, linewidth=1.3, alpha=0.9, label=label)
        axes[1].plot(v_ic, ic_filtered, color=color, linewidth=2.0, alpha=0.95, label=label)

    axes[0].set_title("Raw IC Curves")
    axes[1].set_title(f"Filtered IC Curves (MA window={filter_window})")
    axes[0].set_xlabel("Voltage V (V)")
    axes[1].set_xlabel("Voltage V (V)")
    axes[0].set_ylabel("Incremental Capacity IC = -dQ/dV (Ah/V)")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(loc="best", fontsize=8)
    fig_ic.tight_layout()
    fig_ic.savefig(output_dir / f"{battery_key}_ic_raw_vs_filtered.png", dpi=220)

    fig_cap, ax_cap = plt.subplots(figsize=(10, 4.8))
    discharge_numbers = np.arange(1, len(all_discharge_cycles) + 1)
    cap_integrated = np.array([float(c["capacity_end_ah"]) for c in all_discharge_cycles])
    cap_field = np.array([float(c["capacity_field_ah"]) for c in all_discharge_cycles])

    ax_cap.plot(discharge_numbers, cap_integrated, linewidth=1.6, color="#005f73", label="Integrated Q_end")
    if np.isfinite(cap_field).any():
        ax_cap.plot(discharge_numbers, cap_field, linewidth=1.2, color="#94a1b2", label="Capacity field")

    highlighted_x = [int(all_discharge_cycles[i]["discharge_index"]) + 1 for i in selected_indices]
    highlighted_y = [float(all_discharge_cycles[i]["capacity_end_ah"]) for i in selected_indices]
    ax_cap.scatter(highlighted_x, highlighted_y, color="#bb3e03", s=45, zorder=3, label="Selected cycles")

    ax_cap.set_title(f"{battery_key}: Discharge Capacity Trend")
    ax_cap.set_xlabel("Discharge cycle number")
    ax_cap.set_ylabel("Capacity (Ah)")
    ax_cap.grid(alpha=0.3)
    ax_cap.legend(loc="best")
    fig_cap.tight_layout()
    fig_cap.savefig(output_dir / f"{battery_key}_capacity_trend.png", dpi=220)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot IC curves from NASA battery MAT data.")
    parser.add_argument(
        "--mat-file",
        type=Path,
        default=DEFAULT_MAT_FILE,
        help="Path to a battery MAT file (e.g., 1. BatteryAgingARC-FY08Q4/B0005.mat).",
    )
    parser.add_argument(
        "--battery-key",
        type=str,
        default=None,
        help="Top-level key inside MAT file (defaults to filename stem when available).",
    )
    parser.add_argument(
        "--num-cycles",
        type=int,
        default=DEFAULT_NUM_CYCLES,
        help="Number of representative discharge cycles to plot.",
    )
    parser.add_argument(
        "--delta-v",
        type=float,
        default=DEFAULT_DELTA_V,
        help="Voltage interval for finite-difference IC computation.",
    )
    parser.add_argument(
        "--filter-window",
        type=int,
        default=DEFAULT_FILTER_WINDOW,
        help="Odd moving-average window size for IC smoothing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demo_outputs_real_data"),
        help="Directory for saved PNG plots.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.filter_window <= 0:
        raise ValueError("--filter-window must be a positive odd integer.")
    if args.filter_window % 2 == 0:
        raise ValueError("--filter-window must be odd.")
    if args.delta_v <= 0.0:
        raise ValueError("--delta-v must be positive.")
    if args.num_cycles <= 0:
        raise ValueError("--num-cycles must be positive.")

    battery_struct, battery_key = load_battery_struct(args.mat_file, args.battery_key)
    discharge_cycles = extract_discharge_cycles(battery_struct)
    if not discharge_cycles:
        raise RuntimeError("No discharge cycles found in the selected MAT file.")

    selected = pick_representative_indices(len(discharge_cycles), args.num_cycles)
    plot_results(
        battery_key=battery_key,
        all_discharge_cycles=discharge_cycles,
        selected_indices=selected,
        delta_v=args.delta_v,
        filter_window=args.filter_window,
        output_dir=args.output_dir,
    )

    print(f"Battery key: {battery_key}")
    print(f"Total discharge cycles found: {len(discharge_cycles)}")
    print("Selected representative discharge cycles:")
    for idx in selected:
        cycle = discharge_cycles[idx]
        print(
            f"  - Discharge #{int(cycle['discharge_index']) + 1}: "
            f"Q_end={float(cycle['capacity_end_ah']):.3f} Ah"
        )

    print("\nSaved plots:")
    print(f"  - {args.output_dir / f'{battery_key}_vq_representative_cycles.png'}")
    print(f"  - {args.output_dir / f'{battery_key}_ic_raw_vs_filtered.png'}")
    print(f"  - {args.output_dir / f'{battery_key}_capacity_trend.png'}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
