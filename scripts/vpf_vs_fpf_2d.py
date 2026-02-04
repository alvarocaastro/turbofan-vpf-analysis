"""2D aerodynamic comparison of VPF vs FPF across flight phases.

This script loads flight phases and polar data, computes performance metrics
for both Fixed Pitch Fan (FPF) and Variable Pitch Fan (VPF) configurations,
and generates a comparison table and plots.
"""

import sys
from pathlib import Path

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from turbofan_vpf.aero.incidence_control import compare_phase
from turbofan_vpf.aero.polars import load_polar_csv
from turbofan_vpf.domain.flight_phases import FlightPhase, get_default_phases
from turbofan_vpf.utils.paths import (
    ensure_dir_exists,
    get_figure_path,
    get_polar_file,
    get_results_csv,
)
from turbofan_vpf.viz.plots import (
    plot_cd_comparison_bars,
    plot_cd_vs_alpha,
    plot_ld_comparison,
    plot_polar_with_phases_v2,
)

# Constants
VPF_TARGET = "max_ld"  # VPF optimization target
SEPARATOR_LENGTH = 70


def print_separator() -> None:
    """Print a separator line."""
    print("=" * SEPARATOR_LENGTH)


def load_and_validate_polar(polar_path: Path) -> tuple:
    """Load and validate polar data.

    Args:
        polar_path: Path to polar CSV file

    Returns:
        Tuple of (alpha, cl, cd) arrays

    Raises:
        SystemExit: If file not found or loading fails
    """
    print(f"Loading polar data from: {polar_path}")
    if not polar_path.exists():
        print(f"ERROR: Polar file not found: {polar_path}")
        sys.exit(1)

    try:
        alpha, cl, cd = load_polar_csv(polar_path)
        polar = (alpha, cl, cd)
        print(f"  Loaded {len(alpha)} data points")
        print(f"  Alpha range: {alpha.min():.2f}° to {alpha.max():.2f}°")
        print()
        return polar
    except Exception as e:
        print(f"ERROR: Failed to load polar data: {e}")
        sys.exit(1)


def compute_phase_metrics(
    polar: tuple,
    phases: list[FlightPhase],
    vpf_target: str,
) -> list[dict[str, float]]:
    """Compute metrics for each flight phase.

    Args:
        polar: Tuple of (alpha_deg, cl, cd) arrays
        phases: List of flight phases
        vpf_target: VPF optimization target

    Returns:
        List of dictionaries with phase comparison results
    """
    print("Computing metrics for each flight phase...")
    results = []

    for phase in phases:
        try:
            comparison = compare_phase(polar, phase, vpf_target=vpf_target)
            results.append(
                {
                    "phase": phase.name,
                    "altitude_m": phase.altitude_m,
                    "mach": phase.mach,
                    "alpha_fpf": comparison["fpf_alpha"],
                    "alpha_vpf": comparison["vpf_alpha"],
                    "cd_fpf": comparison["fpf_cd"],
                    "cd_vpf": comparison["vpf_cd"],
                    "delta_cd": comparison["delta_cd"],
                    "ratio_cd": comparison["cd_ratio"],
                    "ld_fpf": comparison["fpf_ld"],
                    "ld_vpf": comparison["vpf_ld"],
                }
            )
            print(
                f"  [OK] {phase.name:10s} - CD reduction: {comparison['cd_reduction_pct']:6.2f}%"
            )
        except Exception as e:
            print(f"  [ERROR] {phase.name:10s} - ERROR: {e}")
            continue

    print()
    return results


def save_results_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Save results DataFrame to CSV.

    Args:
        df: DataFrame with results
        output_path: Path to save CSV file
    """
    ensure_dir_exists(output_path.parent)
    print(f"Saving results to: {output_path}")
    df.to_csv(output_path, index=False, float_format="%.6f")
    print("  [OK] Saved successfully")
    print()


def generate_plots(
    polar: tuple,
    phase_results: list[dict[str, float]],
) -> None:
    """Generate all comparison plots.

    Args:
        polar: Tuple of (alpha_deg, cl, cd) arrays
        phase_results: List of phase result dictionaries
    """
    print("Generating plots...")

    plot_functions = [
        (plot_polar_with_phases_v2, "polar_cl_cd_phases.png", "Polar CL-CD plot"),
        (plot_cd_vs_alpha, "cd_vs_alpha.png", "CD vs alpha plot"),
        (plot_cd_comparison_bars, "cd_comparison_bars.png", "CD comparison bars"),
        (plot_ld_comparison, "ld_comparison.png", "L/D comparison plot"),
    ]

    for plot_func, filename, description in plot_functions:
        try:
            output_path = get_figure_path(filename)
            if plot_func == plot_cd_comparison_bars or plot_func == plot_ld_comparison:
                plot_func(phase_results, output_path)
            else:
                plot_func(polar, phase_results, output_path)
            print(f"  [OK] {description} saved")
        except Exception as e:
            print(f"  [ERROR] Failed to generate {description}: {e}")

    print()


def print_results_table(df: pd.DataFrame) -> None:
    """Print formatted results table.

    Args:
        df: DataFrame with results
    """
    print_separator()
    print("RESULTS SUMMARY")
    print_separator()
    print()

    # Format table for display
    header = (
        f"{'Phase':<12} {'Alpha FPF':>10} {'Alpha VPF':>10} {'CD FPF':>10} "
        f"{'CD VPF':>10} {'Delta CD':>10} {'Ratio CD':>10} {'LD FPF':>10} {'LD VPF':>10}"
    )
    print(header)
    print("-" * 100)

    for _, row in df.iterrows():
        print(
            f"{row['phase']:<12} "
            f"{row['alpha_fpf']:>10.3f} "
            f"{row['alpha_vpf']:>10.3f} "
            f"{row['cd_fpf']:>10.6f} "
            f"{row['cd_vpf']:>10.6f} "
            f"{row['delta_cd']:>10.6f} "
            f"{row['ratio_cd']:>10.4f} "
            f"{row['ld_fpf']:>10.2f} "
            f"{row['ld_vpf']:>10.2f}"
        )

    print()
    print_separator()


def print_statistics(df: pd.DataFrame) -> None:
    """Print summary statistics.

    Args:
        df: DataFrame with results
    """
    print("\nSTATISTICS:")
    print(f"  Average CD reduction: {df['cd_fpf'].mean() - df['cd_vpf'].mean():.6f}")
    print(f"  Average CD ratio: {df['ratio_cd'].mean():.4f}")
    print(f"  Average LD improvement: {df['ld_vpf'].mean() - df['ld_fpf'].mean():.2f}")
    print(f"  Average LD ratio: {(df['ld_vpf'] / df['ld_fpf']).mean():.4f}")

    # Find best and worst phases
    best_cd_reduction = df.loc[df["delta_cd"].idxmin(), "phase"]
    worst_cd_reduction = df.loc[df["delta_cd"].idxmax(), "phase"]
    best_ld_improvement = df.loc[(df["ld_vpf"] / df["ld_fpf"]).idxmax(), "phase"]

    print(f"\n  Best CD reduction: {best_cd_reduction}")
    print(f"  Worst CD reduction: {worst_cd_reduction}")
    print(f"  Best LD improvement: {best_ld_improvement}")


def main() -> None:
    """Main execution function."""
    print_separator()
    print("VPF vs FPF 2D Aerodynamic Comparison")
    print_separator()
    print()

    # Load flight phases
    print("Loading flight phases...")
    phases = get_default_phases()
    print(f"  Loaded {len(phases)} flight phases")
    print()

    # Load polar data
    polar_path = get_polar_file("example.csv")
    polar = load_and_validate_polar(polar_path)

    # Compute metrics for each phase
    results = compute_phase_metrics(polar, phases, VPF_TARGET)

    # Create DataFrame and save
    df = pd.DataFrame(results)
    output_csv_path = get_results_csv("results_by_phase.csv")
    save_results_csv(df, output_csv_path)

    # Generate plots
    generate_plots(polar, results)

    # Print summary
    print_results_table(df)
    print_statistics(df)

    print()
    print_separator()
    print("Analysis complete!")
    print_separator()


if __name__ == "__main__":
    main()
