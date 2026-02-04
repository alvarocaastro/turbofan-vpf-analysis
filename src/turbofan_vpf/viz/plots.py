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


def _format_phase_label(phase_name: str, mach: float) -> str:
    """Format phase label with Mach number.

    Args:
        phase_name: Phase name
        mach: Mach number

    Returns:
        Formatted label string
    """
    return f"{phase_name.capitalize()} (M={mach:.2f})"


def _plot_fpf_vpf_points_enhanced(
    ax: plt.Axes,
    phase_results: PhaseResults,
    colors: np.ndarray,
    x_fpf: np.ndarray,
    y_fpf: np.ndarray,
    x_vpf: np.ndarray,
    y_vpf: np.ndarray,
    show_all_labels: bool = False,
) -> None:
    """Plot FPF and VPF points for phases with enhanced labels.

    Args:
        ax: Matplotlib axes
        phase_results: List of phase result dictionaries
        colors: Color array for phases
        x_fpf: X coordinates for FPF points
        y_fpf: Y coordinates for FPF points
        x_vpf: X coordinates for VPF points
        y_vpf: Y coordinates for VPF points
        show_all_labels: If True, show labels for all phases
    """
    # Create legend entries for FPF and VPF markers (shown once)
    fpf_label_shown = False
    vpf_label_shown = False

    for i, result in enumerate(phase_results):
        phase_name = result["phase"]
        mach = result.get("mach", 0.0)
        color = colors[i]
        phase_label = _format_phase_label(phase_name, mach)

        # FPF point
        ax.scatter(
            x_fpf[i],
            y_fpf[i],
            marker=MARKER_FPF,
            s=MARKER_SIZE,
            color=color,
            edgecolors=COLOR_POLAR,
            linewidth=EDGE_WIDTH,
            label="FPF" if not fpf_label_shown else "",
            alpha=ALPHA_POINT,
            zorder=3,
        )
        if not fpf_label_shown:
            fpf_label_shown = True

        # VPF point
        ax.scatter(
            x_vpf[i],
            y_vpf[i],
            marker=MARKER_VPF,
            s=MARKER_SIZE,
            color=color,
            edgecolors=COLOR_POLAR,
            linewidth=EDGE_WIDTH,
            label="VPF (optimized)" if not vpf_label_shown else "",
            alpha=ALPHA_POINT,
            zorder=3,
        )
        if not vpf_label_shown:
            vpf_label_shown = True

        # Add phase annotation near points
        if show_all_labels:
            mid_x = (x_fpf[i] + x_vpf[i]) / 2
            mid_y = (y_fpf[i] + y_vpf[i]) / 2
            ax.annotate(
                phase_label,
                xy=(mid_x, mid_y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=FONT_SIZE_VALUE,
                alpha=0.8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="gray"),
            )


def plot_polar_with_phases_v2(
    polar: PolarData,
    phase_results: PhaseResults,
    output_path: Path,
) -> None:
    """Plot polar CL-CD with points for each phase (FPF vs VPF).

    Improved version with professional styling and clear technical labels.
    The polar curve shown is the base (incompressible) polar. Operating points
    are compressibility-corrected for their respective phase Mach numbers.

    Args:
        polar: Tuple of (alpha_deg, cl, cd) arrays (base/incompressible polar)
        phase_results: List of dictionaries with phase comparison results (values are compressibility-corrected)
        output_path: Path to save the figure
    """
    alpha, cl, cd = polar

    fig, ax = plt.subplots(figsize=FIG_SIZE_TALL)
    fig.patch.set_facecolor("white")

    # Plot base polar curve (incompressible)
    ax.plot(
        cd,
        cl,
        COLOR_POLAR,
        linewidth=LINE_WIDTH,
        label="Airfoil Polar (incompressible, base)",
        alpha=ALPHA_CURVE,
        zorder=1,
    )

    # Interpolate CL for FPF and VPF points
    num_phases = len(phase_results)
    colors = _get_phase_colors(num_phases)

    cd_fpf = np.array([r["cd_fpf"] for r in phase_results])
    cd_vpf = np.array([r["cd_vpf"] for r in phase_results])
    cl_fpf = np.array([np.interp(r["alpha_fpf"], alpha, cl) for r in phase_results])
    cl_vpf = np.array([np.interp(r["alpha_vpf"], alpha, cl) for r in phase_results])

    _plot_fpf_vpf_points_enhanced(ax, phase_results, colors, cd_fpf, cl_fpf, cd_vpf, cl_vpf)

    # Add phase annotations with Mach numbers
    for i, result in enumerate(phase_results):
        phase_name = result["phase"]
        mach = result.get("mach", 0.0)
        phase_label = _format_phase_label(phase_name, mach)
        mid_cd = (cd_fpf[i] + cd_vpf[i]) / 2
        mid_cl = (cl_fpf[i] + cl_vpf[i]) / 2
        ax.annotate(
            phase_label,
            xy=(mid_cd, mid_cl),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=FONT_SIZE_VALUE,
            alpha=0.85,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor=colors[i], linewidth=1.5),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color=colors[i], alpha=0.6, lw=1),
        )

    ax.set_xlabel("Drag Coefficient, $C_d$ [-]", fontsize=FONT_SIZE_LABEL, fontweight="normal")
    ax.set_ylabel("Lift Coefficient, $C_l$ [-]", fontsize=FONT_SIZE_LABEL, fontweight="normal")
    ax.set_title(
        "Aerodynamic Polar: Operating Points Comparison\nFPF vs VPF by Flight Phase",
        fontsize=FONT_SIZE_TITLE,
        fontweight="bold",
        pad=20,
    )
    ax.grid(True, alpha=ALPHA_GRID, linestyle="--", linewidth=0.5)
    ax.legend(loc="upper left", fontsize=FONT_SIZE_LEGEND, framealpha=0.9, fancybox=True, shadow=True)

    # Add note about compressibility
    ax.text(
        0.02,
        0.98,
        "Note: Operating points are compressibility-corrected\nusing Prandtl-Glauert theory for each phase Mach number",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.7, edgecolor="gray"),
    )

    plt.tight_layout()
    _save_figure(output_path)


def plot_cd_vs_alpha(
    polar: PolarData,
    phase_results: PhaseResults,
    output_path: Path,
) -> None:
    """Plot CD vs alpha with optimal alpha markers.

    Professional version with enhanced technical labels and annotations.
    The polar curve shown is the base (incompressible) polar. Operating points
    are compressibility-corrected for their respective phase Mach numbers.

    Args:
        polar: Tuple of (alpha_deg, cl, cd) arrays (base/incompressible polar)
        phase_results: List of dictionaries with phase comparison results (values are compressibility-corrected)
        output_path: Path to save the figure
    """
    alpha, _, cd = polar

    fig, ax = plt.subplots(figsize=FIG_SIZE_STANDARD)
    fig.patch.set_facecolor("white")

    # Plot base CD curve (incompressible)
    ax.plot(
        alpha,
        cd,
        COLOR_POLAR,
        linewidth=LINE_WIDTH,
        label="Drag Polar (incompressible, base)",
        alpha=ALPHA_CURVE,
        zorder=1,
    )

    # Mark optimal alpha (VPF) and FPF alpha for each phase
    num_phases = len(phase_results)
    colors = _get_phase_colors(num_phases)

    alpha_fpf = np.array([r["alpha_fpf"] for r in phase_results])
    alpha_vpf = np.array([r["alpha_vpf"] for r in phase_results])
    cd_fpf = np.array([r["cd_fpf"] for r in phase_results])
    cd_vpf = np.array([r["cd_vpf"] for r in phase_results])

    _plot_fpf_vpf_points_enhanced(ax, phase_results, colors, alpha_fpf, cd_fpf, alpha_vpf, cd_vpf)

    # Add phase annotations
    for i, result in enumerate(phase_results):
        phase_name = result["phase"]
        mach = result.get("mach", 0.0)
        phase_label = _format_phase_label(phase_name, mach)
        mid_alpha = (alpha_fpf[i] + alpha_vpf[i]) / 2
        mid_cd = (cd_fpf[i] + cd_vpf[i]) / 2
        ax.annotate(
            phase_label,
            xy=(mid_alpha, mid_cd),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=FONT_SIZE_VALUE,
            alpha=0.85,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor=colors[i], linewidth=1.5),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color=colors[i], alpha=0.6, lw=1),
        )

    ax.set_xlabel("Angle of Attack, $\\alpha$ [deg]", fontsize=FONT_SIZE_LABEL, fontweight="normal")
    ax.set_ylabel("Drag Coefficient, $C_d$ [-]", fontsize=FONT_SIZE_LABEL, fontweight="normal")
    ax.set_title(
        "Drag Coefficient vs Angle of Attack\nOperating Points: FPF vs VPF Configurations",
        fontsize=FONT_SIZE_TITLE,
        fontweight="bold",
        pad=20,
    )
    ax.grid(True, alpha=ALPHA_GRID, linestyle="--", linewidth=0.5)
    ax.legend(loc="upper left", fontsize=FONT_SIZE_LEGEND, framealpha=0.9, fancybox=True, shadow=True)

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
            fontweight="bold",
        )


def plot_cd_comparison_bars(
    phase_results: PhaseResults,
    output_path: Path,
) -> None:
    """Plot bar chart comparing delta_cd and ratio_cd by phase.

    Professional version with enhanced technical presentation.

    Args:
        phase_results: List of dictionaries with phase comparison results
        output_path: Path to save the figure
    """
    df = pd.DataFrame(phase_results)
    phases = df["phase"].values
    mach_numbers = df["mach"].values
    delta_cd = df["delta_cd"].values
    ratio_cd = df["ratio_cd"].values

    # Format phase labels with Mach numbers
    phase_labels = [_format_phase_label(phase, mach) for phase, mach in zip(phases, mach_numbers)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE_WIDE)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Drag Coefficient Comparison: VPF vs FPF",
        fontsize=FONT_SIZE_TITLE + 2,
        fontweight="bold",
        y=1.02,
    )

    # Delta CD bars
    bars1 = ax1.bar(phase_labels, delta_cd, color=COLOR_FPF, alpha=ALPHA_POINT, edgecolor=COLOR_POLAR, linewidth=1.5)
    ax1.axhline(y=0, color=COLOR_POLAR, linestyle="--", linewidth=1.5, label="Reference (VPF = FPF)")
    ax1.set_ylabel("Drag Coefficient Difference\n$\\Delta C_d = C_{d,VPF} - C_{d,FPF}$ [-]", fontsize=FONT_SIZE_LABEL)
    ax1.set_xlabel("Flight Phase", fontsize=FONT_SIZE_LABEL)
    ax1.set_title("Absolute Difference", fontsize=FONT_SIZE_SUBTITLE, fontweight="bold")
    ax1.grid(True, alpha=ALPHA_GRID, axis="y", linestyle="--", linewidth=0.5)
    ax1.tick_params(axis="x", rotation=45)
    ax1.legend(loc="best", fontsize=FONT_SIZE_LEGEND - 1, framealpha=0.9)

    _add_bar_value_labels(ax1, bars1, fmt="{:.4f}", offset=0.0001 if delta_cd.min() >= 0 else -0.0003)

    # Ratio CD bars
    bars2 = ax2.bar(phase_labels, ratio_cd, color=COLOR_VPF, alpha=ALPHA_POINT, edgecolor=COLOR_POLAR, linewidth=1.5)
    ax2.axhline(
        y=1.0,
        color=COLOR_POLAR,
        linestyle="--",
        linewidth=1.5,
        label="Reference (VPF = FPF, ratio = 1.0)",
    )
    ax2.set_ylabel("Drag Coefficient Ratio\n$C_{d,VPF} / C_{d,FPF}$ [-]", fontsize=FONT_SIZE_LABEL)
    ax2.set_xlabel("Flight Phase", fontsize=FONT_SIZE_LABEL)
    ax2.set_title("Relative Ratio", fontsize=FONT_SIZE_SUBTITLE, fontweight="bold")
    ax2.grid(True, alpha=ALPHA_GRID, axis="y", linestyle="--", linewidth=0.5)
    ax2.tick_params(axis="x", rotation=45)
    ax2.legend(loc="best", fontsize=FONT_SIZE_LEGEND - 1, framealpha=0.9)

    _add_bar_value_labels(ax2, bars2, fmt="{:.3f}", offset=0.02)

    plt.tight_layout()
    _save_figure(output_path)


def plot_ld_comparison(
    phase_results: PhaseResults,
    output_path: Path,
) -> None:
    """Plot L/D comparison by phase (FPF vs VPF).

    Professional version with enhanced technical presentation.

    Args:
        phase_results: List of dictionaries with phase comparison results
        output_path: Path to save the figure
    """
    df = pd.DataFrame(phase_results)
    phases = df["phase"].values
    mach_numbers = df["mach"].values
    ld_fpf = df["ld_fpf"].values
    ld_vpf = df["ld_vpf"].values

    # Format phase labels with Mach numbers
    phase_labels = [_format_phase_label(phase, mach) for phase, mach in zip(phases, mach_numbers)]

    x = np.arange(len(phases))
    width = 0.35

    fig, ax = plt.subplots(figsize=FIG_SIZE_STANDARD)
    fig.patch.set_facecolor("white")

    bars1 = ax.bar(
        x - width / 2,
        ld_fpf,
        width,
        label="Fixed Pitch Fan (FPF)",
        color=COLOR_FPF,
        alpha=ALPHA_POINT,
        edgecolor=COLOR_POLAR,
        linewidth=1.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        ld_vpf,
        width,
        label="Variable Pitch Fan (VPF, optimized)",
        color=COLOR_VPF,
        alpha=ALPHA_POINT,
        edgecolor=COLOR_POLAR,
        linewidth=1.5,
    )

    ax.set_ylabel("Lift-to-Drag Ratio, $L/D = C_l / C_d$ [-]", fontsize=FONT_SIZE_LABEL)
    ax.set_xlabel("Flight Phase", fontsize=FONT_SIZE_LABEL)
    ax.set_title(
        "Aerodynamic Efficiency Comparison\nLift-to-Drag Ratio: FPF vs VPF",
        fontsize=FONT_SIZE_TITLE,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels, rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.legend(loc="upper left", fontsize=FONT_SIZE_LEGEND, framealpha=0.9, fancybox=True, shadow=True)
    ax.grid(True, alpha=ALPHA_GRID, axis="y", linestyle="--", linewidth=0.5)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        _add_bar_value_labels(ax, bars, fmt="{:.1f}", offset=0.5)

    # Add improvement percentage annotations
    for i, (ld_f, ld_v) in enumerate(zip(ld_fpf, ld_vpf)):
        improvement = ((ld_v - ld_f) / ld_f * 100) if ld_f > 0 else 0
        ax.text(
            x[i],
            max(ld_f, ld_v) + 2,
            f"+{improvement:.1f}%",
            ha="center",
            fontsize=FONT_SIZE_VALUE - 1,
            fontweight="bold",
            color=COLOR_VPF,
        )

    plt.tight_layout()
    _save_figure(output_path)
