# turbofan-vpf-analysis

TFG project: 2D aerodynamic comparison of Variable Pitch Fan (VPF) vs Fixed Pitch Fan (FPF) using Python and airfoil polars.

## Objective

This project performs a 2D aerodynamic analysis comparing the performance of Variable Pitch Fan (VPF) and Fixed Pitch Fan (FPF) configurations across typical commercial aircraft flight phases. The analysis evaluates drag coefficients (CD) and lift-to-drag ratios (L/D) to quantify the potential benefits of variable pitch control in turbofan engines.

## Method

The comparison is based on 2D airfoil polar data and incidence angle control strategies:

- **Fixed Pitch Fan (FPF)**: Operates at a fixed geometric angle of attack that varies linearly with Mach number according to the law: `α_FPF = α₀ + k_Mach × (Mach - Mach_ref)`, where α₀ = 4.0° at Mach_ref = 0.5, and k_Mach = 2.0 deg/Mach.

- **Variable Pitch Fan (VPF)**: Adjusts its pitch angle to optimize performance at each flight condition. The analysis considers two optimization targets: maximum lift-to-drag ratio (max L/D) or minimum drag coefficient (min CD).

For each flight phase, the method:
1. Loads the base (incompressible) airfoil polar data
2. Applies compressibility corrections using Prandtl-Glauert theory for the phase Mach number
3. Determines the operating angle of attack for both FPF and VPF configurations using the corrected polar
4. Interpolates aerodynamic coefficients (CL, CD) from the corrected polar
5. Computes performance metrics (CD, L/D) for comparison
6. Generates quantitative comparisons (deltas, ratios, percentages)

## Assumptions

The analysis uses standard flight phases with representative Mach numbers and altitudes:

| Phase     | Altitude [m] | Mach Number |
|-----------|--------------|-------------|
| Takeoff   | 0            | 0.25        |
| Climb     | 5,000        | 0.60        |
| Cruise    | 11,000       | 0.80        |
| Descent   | 7,000        | 0.65        |
| Approach  | 500          | 0.30        |

Additional assumptions:
- 2D analysis using airfoil polar data (no 3D effects)
- Simplified ISA atmosphere model for atmospheric properties
- Linear interpolation of aerodynamic coefficients
- FPF incidence angle varies linearly with Mach number
- Compressibility corrections applied using Prandtl-Glauert theory

## Installation

```bash
# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

## How to Run

### Main Analysis Script

Run the complete VPF vs FPF comparison:

```bash
python scripts/vpf_vs_fpf_2d.py
```

This script will:
1. Load flight phases and polar data from `data/polars/example.csv`
2. Compute performance metrics for all phases
3. Generate comparison plots in `outputs/figures/`
4. Save results table to `outputs/results_by_phase.csv`
5. Display summary statistics in the console

### Output Files

- **`outputs/results_by_phase.csv`**: Detailed comparison table with all metrics per phase
- **`outputs/figures/`**: Four visualization plots (see Figures Produced section)

## Figures Produced

The analysis generates four comparison plots:

1. **`polar_cl_cd_phases.png`**: Polar diagram (CL vs CD) showing the airfoil curve with operating points for FPF (circles) and VPF (triangles) at each flight phase. Visualizes where each configuration operates on the polar.

2. **`cd_vs_alpha.png`**: Drag coefficient versus angle of attack, marking FPF and VPF operating points for each phase. Shows the angle of attack selection strategy and resulting drag levels.

3. **`cd_comparison_bars.png`**: Bar charts comparing CD difference (VPF - FPF) and CD ratio (VPF/FPF) across phases. Quantifies the drag penalty or benefit of VPF relative to FPF.

4. **`ld_comparison.png`**: Side-by-side bar chart of lift-to-drag ratios for FPF and VPF by phase. Highlights the efficiency improvement achieved by variable pitch control.

## Compressibility Model & Limitations

The analysis applies **Prandtl-Glauert compressibility corrections** to account for subsonic compressibility effects. The corrections are applied automatically to all aerodynamic coefficients:

- **Lift coefficient**: `CL_corrected = CL_incompressible / β`, where `β = √(1 - M²)`
- **Drag coefficient**: `CD_corrected = CD_incompressible / β²`

### Model Limitations

**Important**: This model has significant limitations and should not be used for transonic or supersonic analysis:

1. **No wave drag**: The model does not account for wave drag or shock effects that occur near and above Mach 0.7.

2. **Subsonic validity**: The Prandtl-Glauert correction is valid only for subsonic flow (M < 0.7). Results become increasingly inaccurate as Mach number approaches 0.7.

3. **No transonic effects**: The model does not capture transonic phenomena such as:
   - Shock wave formation
   - Drag divergence
   - Critical Mach number effects
   - Local supersonic regions

4. **Maximum valid Mach**: The default configuration sets `MACH_MAX_VALID = 0.7`. Warnings are issued when flight phases exceed this value (e.g., cruise at M = 0.80).

**For accurate transonic analysis**, more sophisticated methods (e.g., transonic small disturbance theory, computational fluid dynamics) would be required. This 2D analysis is intended for conceptual comparison and should not be used for detailed design or performance predictions in the transonic regime.

## Project Structure

```
.
├── src/turbofan_vpf/    # Main package
│   ├── domain/          # Domain models and data structures
│   ├── aero/            # Aerodynamic calculations
│   ├── viz/             # Visualization utilities
│   └── utils/           # Utility functions
├── scripts/             # Analysis scripts
├── data/polars/         # Airfoil polar data files
├── outputs/figures/     # Generated plots and figures
└── tests/               # Unit tests
```

## Development

```bash
# Format code
black src/ tests/ scripts/

# Lint code
ruff check src/ tests/ scripts/

# Run tests
pytest
```
