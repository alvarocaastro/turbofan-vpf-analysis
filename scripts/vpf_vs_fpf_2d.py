"""2D aerodynamic comparison of VPF vs FPF across flight phases.

This script implements the FPF vs VPF model:
- FPF: Fixed angle = alpha_min_cd(cruise), constant across all phases
- VPF: Variable angle = alpha_min_cd(phase) for each phase
"""

import sys
from pathlib import Path

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from turbofan_vpf.aero.compressibility import apply_compressibility_to_polar
from turbofan_vpf.aero.incidence_control import compute_phase_results
from turbofan_vpf.aero.optima import find_alpha_min_cd
from turbofan_vpf.aero.polars import load_polar_csv
from turbofan_vpf.domain.flight_phases import FlightPhase, get_default_phases
from turbofan_vpf.utils.paths import (
    ensure_dir_exists,
    get_figure_path,
    get_polar_file,
    get_results_csv,
)
from turbofan_vpf.viz.plots import (
    plot_cd_vs_alpha,
    plot_delta_alpha_pitch,
    plot_delta_cd_bars,
    plot_polar_with_phases_v2,
)

# Constants
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


def compute_all_phase_results(
    polar_base: tuple,
    phases: list[FlightPhase],
) -> list[dict[str, float]]:
    """Compute results for all flight phases.

    Args:
        polar_base: Tuple of (alpha_deg, cl, cd) arrays (base polar)
        phases: List of flight phases

    Returns:
        List of dictionaries with phase comparison results
    """
    print("Computing metrics for each flight phase...")

    # Step 1: Find cruise phase and calculate alpha_fpf = alpha_min_cd(cruise)
    cruise_phase = next((p for p in phases if p.name == "cruise"), phases[0])
    print(f"  Determining FPF fixed angle from cruise phase (M={cruise_phase.mach:.2f})...")

    # Apply compressibility correction for cruise
    polar_cruise = apply_compressibility_to_polar(polar_base, cruise_phase.mach, phase_name=cruise_phase.name)

    # Calculate alpha_fpf = alpha_min_cd(cruise)
    alpha_fpf = find_alpha_min_cd(polar_cruise)
    print(f"  FPF fixed angle of attack: {alpha_fpf:.2f}° (constant across all phases)")
    print()

    # Step 2: Compute results for each phase
    results = []

    for phase in phases:
        try:
            result = compute_phase_results(polar_base, phase, alpha_fpf)
            results.append(result)
            print(
                f"  [OK] {phase.name:10s} - alpha_min_cd: {result['alpha_min_cd']:6.2f}°, "
                f"delta_alpha_pitch: {result['delta_alpha_pitch']:6.2f}°, "
                f"delta_CD: {result['delta_cd']:7.4f}"
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

    # Select and order columns as specified
    columns = [
        "phase",
        "mach",
        "alpha_min_cd",
        "alpha_fpf",
        "delta_alpha_pitch",
        "cd_at_alpha_min",
        "cd_fpf",
        "delta_cd",
        "ratio_cd",
    ]
    df_output = df[columns].copy()
    df_output.to_csv(output_path, index=False, float_format="%.6f")
    print("  [OK] Saved successfully")
    print()


def generate_plots(
    polar_base: tuple,
    phase_results: list[dict[str, float]],
) -> None:
    """Generate all comparison plots.

    Args:
        polar_base: Tuple of (alpha_deg, cl, cd) arrays (base polar)
        phase_results: List of phase result dictionaries
    """
    print("Generating plots...")

    try:
        # A) CD vs alpha by phase
        plot_cd_vs_alpha(polar_base, phase_results, get_figure_path("cd_vs_alpha.png"))
        print("  [OK] CD vs alpha plot saved")

        # B) Delta CD bars (benefit of VPF)
        plot_delta_cd_bars(phase_results, get_figure_path("delta_cd_bars.png"))
        print("  [OK] Delta CD bars saved")

        # C) Delta alpha pitch bars
        plot_delta_alpha_pitch(phase_results, get_figure_path("delta_alpha_pitch.png"))
        print("  [OK] Delta alpha pitch bars saved")

        # D) Polar CL-CD (optional)
        plot_polar_with_phases_v2(polar_base, phase_results, get_figure_path("polar_cl_cd_phases.png"))
        print("  [OK] Polar CL-CD plot saved")
    except Exception as e:
        print(f"  [ERROR] Failed to generate plots: {e}")

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
        f"{'Phase':<12} {'Mach':>6} {'alpha_min_cd':>12} {'alpha_fpf':>10} "
        f"{'delta_alpha':>12} {'CD_at_min':>12} {'CD_fpf':>10} {'delta_CD':>10} {'ratio_CD':>10}"
    )
    print(header)
    print("-" * 110)

    for _, row in df.iterrows():
        print(
            f"{row['phase']:<12} "
            f"{row['mach']:>6.2f} "
            f"{row['alpha_min_cd']:>12.2f} "
            f"{row['alpha_fpf']:>10.2f} "
            f"{row['delta_alpha_pitch']:>12.2f} "
            f"{row['cd_at_alpha_min']:>12.6f} "
            f"{row['cd_fpf']:>10.6f} "
            f"{row['delta_cd']:>10.6f} "
            f"{row['ratio_cd']:>10.4f}"
        )

    print()
    print_separator()


def print_statistics(df: pd.DataFrame) -> None:
    """Print summary statistics.

    Args:
        df: DataFrame with results
    """
    print("\nSTATISTICS:")
    print(f"  Average delta_CD (VPF benefit): {df['delta_cd'].mean():.6f}")
    print(f"  Average CD ratio: {df['ratio_cd'].mean():.4f}")
    print(f"  Average delta_alpha_pitch: {df['delta_alpha_pitch'].mean():.2f}°")
    print(f"  Max delta_alpha_pitch: {df['delta_alpha_pitch'].abs().max():.2f}°")

    # Find best and worst phases
    best_phase = df.loc[df["delta_cd"].idxmax(), "phase"]
    worst_phase = df.loc[df["delta_cd"].idxmin(), "phase"]

    print(f"\n  Best VPF benefit (delta_CD): {best_phase}")
    print(f"  Worst VPF benefit (delta_CD): {worst_phase}")


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
    polar_base = load_and_validate_polar(polar_path)

    # Compute results for all phases
    results = compute_all_phase_results(polar_base, phases)

    # Create DataFrame and save
    df = pd.DataFrame(results)
    output_csv_path = get_results_csv("results_by_phase.csv")
    save_results_csv(df, output_csv_path)

    # Generate plots
    generate_plots(polar_base, results)

    # Print summary
    print_results_table(df)
    print_statistics(df)

    print()
    print_separator()
    print("Analysis complete!")
    print_separator()


if __name__ == "__main__":
    main()
