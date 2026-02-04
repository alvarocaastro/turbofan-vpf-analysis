"""Advanced plotting functions for VPF vs FPF comparison."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from turbofan_vpf.viz.constants import (
    ALPHA_CURVE,
    ALPHA_GRID,
    ALPHA_POINT,
    COLOR_FPF,
    COLOR_POLAR,
    COLOR_VPF,
    EDGE_WIDTH,
    FIG_SIZE_STANDARD,
    FIG_SIZE_TALL,
    FIG_SIZE_WIDE,
    FIGURE_SAVE_KWARGS,
    FONT_SIZE_LABEL,
    FONT_SIZE_LEGEND,
    FONT_SIZE_SUBTITLE,
    FONT_SIZE_TITLE,
    FONT_SIZE_VALUE,
    LINE_WIDTH,
    MARKER_FPF,
    MARKER_SIZE,
    MARKER_VPF,
)
from turbofan_vpf.utils.paths import ensure_dir_exists

# Type aliases
PolarData = tuple[np.ndarray, np.ndarray, np.ndarray]
PhaseResult = dict[str, Any]
PhaseResults = list[PhaseResult]


def _save_figure(output_path: Path) -> None:
    """Save figure to file with standard settings.

    Args:
        output_path: Path to save the figure
    """
    ensure_dir_exists(output_path.parent)
    plt.savefig(output_path, **FIGURE_SAVE_KWARGS)
    plt.close()


def _get_phase_colors(num_phases: int) -> np.ndarray:
    """Get color array for phases.

    Args:
        num_phases: Number of phases

    Returns:
        Array of colors
    """
    return plt.cm.tab10(np.linspace(0, 1, num_phases))


def _plot_fpf_vpf_points(
    ax: plt.Axes,
    phase_results: PhaseResults,
    colors: np.ndarray,
    x_fpf: np.ndarray,
    y_fpf: np.ndarray,
    x_vpf: np.ndarray,
    y_vpf: np.ndarray,
) -> None:
    """Plot FPF and VPF points for phases.

    Args:
        ax: Matplotlib axes
        phase_results: List of phase result dictionaries
        colors: Color array for phases
        x_fpf: X coordinates for FPF points
        y_fpf: Y coordinates for FPF points
        x_vpf: X coordinates for VPF points
        y_vpf: Y coordinates for VPF points
    """
    for i, result in enumerate(phase_results):
        phase_name = result["phase"]
        color = colors[i]

        # FPF point
        ax.scatter(
            x_fpf[i],
            y_fpf[i],
            marker=MARKER_FPF,
            s=MARKER_SIZE,
            color=color,
            edgecolors=COLOR_POLAR,
            linewidth=EDGE_WIDTH,
            label=f"{phase_name} (FPF)" if i == 0 else "",
            alpha=ALPHA_POINT,
            zorder=3,
        )

        # VPF point
        ax.scatter(
            x_vpf[i],
            y_vpf[i],
            marker=MARKER_VPF,
            s=MARKER_SIZE,
            color=color,
            edgecolors=COLOR_POLAR,
            linewidth=EDGE_WIDTH,
            label=f"{phase_name} (VPF)" if i == 0 else "",
            alpha=ALPHA_POINT,
            zorder=3,
        )


def plot_polar_with_phases_v2(
    polar: PolarData,
    phase_results: PhaseResults,
    output_path: Path,
) -> None:
    """Plot polar CL-CD with points for each phase (FPF vs VPF).

    Improved version that interpolates CL values. The polar curve shown is the
    base (incompressible) polar. Operating points are compressibility-corrected
    for their respective phase Mach numbers.

    Args:
        polar: Tuple of (alpha_deg, cl, cd) arrays (base/incompressible polar)
        phase_results: List of dictionaries with phase comparison results (values are compressibility-corrected)
        output_path: Path to save the figure
    """
    alpha, cl, cd = polar

    fig, ax = plt.subplots(figsize=FIG_SIZE_TALL)

    # Plot base polar curve (incompressible)
    ax.plot(cd, cl, COLOR_POLAR, linewidth=LINE_WIDTH, label="Polar (base)", alpha=ALPHA_CURVE, zorder=1)

    # Interpolate CL for FPF and VPF points
    num_phases = len(phase_results)
    colors = _get_phase_colors(num_phases)

    cd_fpf = np.array([r["cd_fpf"] for r in phase_results])
    cd_vpf = np.array([r["cd_vpf"] for r in phase_results])
    cl_fpf = np.array([np.interp(r["alpha_fpf"], alpha, cl) for r in phase_results])
    cl_vpf = np.array([np.interp(r["alpha_vpf"], alpha, cl) for r in phase_results])

    _plot_fpf_vpf_points(ax, phase_results, colors, cd_fpf, cl_fpf, cd_vpf, cl_vpf)

    ax.set_xlabel("Drag Coefficient, $C_d$", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Lift Coefficient, $C_l$", fontsize=FONT_SIZE_LABEL)
    ax.set_title("Polar CL-CD: FPF vs VPF by Flight Phase", fontsize=FONT_SIZE_TITLE, fontweight="bold")
    ax.grid(True, alpha=ALPHA_GRID)
    ax.legend(loc="best", fontsize=FONT_SIZE_LEGEND, ncol=2)

    plt.tight_layout()
    _save_figure(output_path)


def plot_cd_vs_alpha(
    polar: PolarData,
    phase_results: PhaseResults,
    output_path: Path,
) -> None:
    """Plot CD vs alpha with optimal alpha markers.

    The polar curve shown is the base (incompressible) polar. Operating points
    are compressibility-corrected for their respective phase Mach numbers.

    Args:
        polar: Tuple of (alpha_deg, cl, cd) arrays (base/incompressible polar)
        phase_results: List of dictionaries with phase comparison results (values are compressibility-corrected)
        output_path: Path to save the figure
    """
    alpha, _, cd = polar

    fig, ax = plt.subplots(figsize=FIG_SIZE_STANDARD)

    # Plot base CD curve (incompressible)
    ax.plot(alpha, cd, COLOR_POLAR, linewidth=LINE_WIDTH, label="Polar CD (base)", alpha=ALPHA_CURVE)

    # Mark optimal alpha (VPF) and FPF alpha for each phase
    num_phases = len(phase_results)
    colors = _get_phase_colors(num_phases)

    alpha_fpf = np.array([r["alpha_fpf"] for r in phase_results])
    alpha_vpf = np.array([r["alpha_vpf"] for r in phase_results])
    cd_fpf = np.array([r["cd_fpf"] for r in phase_results])
    cd_vpf = np.array([r["cd_vpf"] for r in phase_results])

    _plot_fpf_vpf_points(ax, phase_results, colors, alpha_fpf, cd_fpf, alpha_vpf, cd_vpf)

    ax.set_xlabel("Angle of Attack, $\\alpha$ [deg]", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Drag Coefficient, $C_d$", fontsize=FONT_SIZE_LABEL)
    ax.set_title("CD vs Alpha: FPF vs VPF Operating Points", fontsize=FONT_SIZE_TITLE, fontweight="bold")
    ax.grid(True, alpha=ALPHA_GRID)
    ax.legend(loc="best", fontsize=FONT_SIZE_LEGEND, ncol=2)

    plt.tight_layout()
    _save_figure(output_path)


def _add_bar_value_labels(ax: plt.Axes, bars: Any, fmt: str = "{:.4f}", offset: float = 0.0) -> None:
    """Add value labels on bar chart.

    Args:
        ax: Matplotlib axes
        bars: Bar chart object
        fmt: Format string for values
        offset: Vertical offset for labels
    """
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=FONT_SIZE_VALUE,
        )


def plot_cd_comparison_bars(
    phase_results: PhaseResults,
    output_path: Path,
) -> None:
    """Plot bar chart comparing delta_cd and ratio_cd by phase.

    Args:
        phase_results: List of dictionaries with phase comparison results
        output_path: Path to save the figure
    """
    df = pd.DataFrame(phase_results)
    phases = df["phase"].values
    delta_cd = df["delta_cd"].values
    ratio_cd = df["ratio_cd"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE_WIDE)

    # Delta CD bars
    bars1 = ax1.bar(phases, delta_cd, color=COLOR_FPF, alpha=ALPHA_POINT, edgecolor=COLOR_POLAR)
    ax1.axhline(y=0, color=COLOR_POLAR, linestyle="--", linewidth=1)
    ax1.set_ylabel("$\\Delta C_d$ (VPF - FPF)", fontsize=FONT_SIZE_LABEL)
    ax1.set_xlabel("Flight Phase", fontsize=FONT_SIZE_LABEL)
    ax1.set_title("CD Difference by Phase", fontsize=FONT_SIZE_SUBTITLE, fontweight="bold")
    ax1.grid(True, alpha=ALPHA_GRID, axis="y")
    ax1.tick_params(axis="x", rotation=45)

    _add_bar_value_labels(ax1, bars1, fmt="{:.4f}", offset=0.0001 if delta_cd.min() >= 0 else -0.0003)

    # Ratio CD bars
    bars2 = ax2.bar(phases, ratio_cd, color=COLOR_VPF, alpha=ALPHA_POINT, edgecolor=COLOR_POLAR)
    ax2.axhline(y=1.0, color=COLOR_POLAR, linestyle="--", linewidth=1, label="Reference (FPF)")
    ax2.set_ylabel("CD Ratio (VPF / FPF)", fontsize=FONT_SIZE_LABEL)
    ax2.set_xlabel("Flight Phase", fontsize=FONT_SIZE_LABEL)
    ax2.set_title("CD Ratio by Phase", fontsize=FONT_SIZE_SUBTITLE, fontweight="bold")
    ax2.grid(True, alpha=ALPHA_GRID, axis="y")
    ax2.tick_params(axis="x", rotation=45)
    ax2.legend()

    _add_bar_value_labels(ax2, bars2, fmt="{:.3f}", offset=0.02)

    plt.tight_layout()
    _save_figure(output_path)


def plot_ld_comparison(
    phase_results: PhaseResults,
    output_path: Path,
) -> None:
    """Plot L/D comparison by phase (FPF vs VPF).

    Args:
        phase_results: List of dictionaries with phase comparison results
        output_path: Path to save the figure
    """
    df = pd.DataFrame(phase_results)
    phases = df["phase"].values
    ld_fpf = df["ld_fpf"].values
    ld_vpf = df["ld_vpf"].values

    x = np.arange(len(phases))
    width = 0.35

    fig, ax = plt.subplots(figsize=FIG_SIZE_STANDARD)

    bars1 = ax.bar(
        x - width / 2,
        ld_fpf,
        width,
        label="FPF",
        color=COLOR_FPF,
        alpha=ALPHA_POINT,
        edgecolor=COLOR_POLAR,
    )
    bars2 = ax.bar(
        x + width / 2,
        ld_vpf,
        width,
        label="VPF",
        color=COLOR_VPF,
        alpha=ALPHA_POINT,
        edgecolor=COLOR_POLAR,
    )

    ax.set_ylabel("Lift-to-Drag Ratio, $L/D$", fontsize=FONT_SIZE_LABEL)
    ax.set_xlabel("Flight Phase", fontsize=FONT_SIZE_LABEL)
    ax.set_title("L/D Comparison: FPF vs VPF by Flight Phase", fontsize=FONT_SIZE_TITLE, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(phases, rotation=45, ha="right")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=ALPHA_GRID, axis="y")

    # Add value labels on bars
    for bars in [bars1, bars2]:
        _add_bar_value_labels(ax, bars, fmt="{:.1f}", offset=0.5)

    plt.tight_layout()
    _save_figure(output_path)
