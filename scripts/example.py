"""Example script demonstrating package usage."""

import numpy as np
from pathlib import Path

from turbofan_vpf.aero.calculations import calculate_lift_coefficient
from turbofan_vpf.viz.plotting import plot_lift_curve


def main() -> None:
    """Run example analysis."""
    # Generate sample data
    alpha = np.linspace(-5, 15, 50)
    cl = calculate_lift_coefficient(alpha=alpha, cl_alpha=0.1, alpha_0=0.0)

    # Plot results
    ax = plot_lift_curve(alpha, cl, label="Example Airfoil")
    output_path = Path("outputs/figures/example_lift_curve.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
