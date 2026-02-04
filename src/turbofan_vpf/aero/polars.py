"""Airfoil polar data loading and interpolation."""

import warnings
from pathlib import Path
from typing import Tuple

import numpy as np


def load_polar_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load airfoil polar data from CSV file.

    Expected CSV format with header row containing: alpha_deg, cl, cd

    Args:
        path: Path to the CSV file

    Returns:
        Tuple of (alpha_deg, cl, cd) arrays where:
        - alpha_deg: Angle of attack in degrees
        - cl: Lift coefficient
        - cd: Drag coefficient

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file format is invalid or missing required columns
    """
    if not path.exists():
        raise FileNotFoundError(f"Polar file not found: {path}")

    try:
        # Read header and data
        with open(path, "r", encoding="utf-8") as f:
            header_line = f.readline().strip()
            headers = [h.strip().lower() for h in header_line.split(",")]

            # Check required columns
            required_cols = ["alpha_deg", "cl", "cd"]
            for col in required_cols:
                if col not in headers:
                    raise ValueError(
                        f"CSV file must contain '{col}' column. Found: {headers}"
                    )

            # Get column indices
            alpha_idx = headers.index("alpha_deg")
            cl_idx = headers.index("cl")
            cd_idx = headers.index("cd")

            # Read data rows
            data_rows = []
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    values = [float(v.strip()) for v in line.split(",")]
                    data_rows.append(values)

        if not data_rows:
            raise ValueError("No data rows found in CSV file")

        # Convert to numpy arrays
        data_array = np.array(data_rows)
        alpha_deg = data_array[:, alpha_idx]
        cl = data_array[:, cl_idx]
        cd = data_array[:, cd_idx]

        # Remove any NaN values
        valid_mask = ~(np.isnan(alpha_deg) | np.isnan(cl) | np.isnan(cd))
        alpha_deg = alpha_deg[valid_mask]
        cl = cl[valid_mask]
        cd = cd[valid_mask]

        if len(alpha_deg) == 0:
            raise ValueError("No valid data found in CSV file")

        # Set as current polar data for interpolation
        set_polar_data(alpha_deg, cl, cd)

        return alpha_deg, cl, cd

    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        raise ValueError(f"Error reading polar CSV file: {e}") from e


# Module-level storage for loaded polar data
_loaded_alpha: np.ndarray | None = None
_loaded_cl: np.ndarray | None = None
_loaded_cd: np.ndarray | None = None


def interp_cl_cd(alpha_deg: float) -> Tuple[float, float]:
    """Interpolate lift and drag coefficients for given angle of attack.

    Uses linear interpolation on the most recently loaded polar data.
    Emits warnings if alpha_deg is outside the data range.

    Args:
        alpha_deg: Angle of attack in degrees

    Returns:
        Tuple of (cl, cd) interpolated coefficients

    Raises:
        RuntimeError: If no polar data has been loaded yet or data is empty
    """
    global _loaded_alpha, _loaded_cl, _loaded_cd

    if (
        _loaded_alpha is None
        or _loaded_cl is None
        or _loaded_cd is None
        or len(_loaded_alpha) == 0
    ):
        raise RuntimeError(
            "No polar data loaded. Call load_polar_csv() first to load data."
        )

    alpha_min = _loaded_alpha.min()
    alpha_max = _loaded_alpha.max()

    if alpha_deg < alpha_min:
        warnings.warn(
            f"Alpha {alpha_deg:.2f}° is below minimum {alpha_min:.2f}°. "
            f"Extrapolating using boundary value.",
            UserWarning,
            stacklevel=2,
        )
        alpha_deg = alpha_min
    elif alpha_deg > alpha_max:
        warnings.warn(
            f"Alpha {alpha_deg:.2f}° is above maximum {alpha_max:.2f}°. "
            f"Extrapolating using boundary value.",
            UserWarning,
            stacklevel=2,
        )
        alpha_deg = alpha_max

    cl = np.interp(alpha_deg, _loaded_alpha, _loaded_cl)
    cd = np.interp(alpha_deg, _loaded_alpha, _loaded_cd)

    return float(cl), float(cd)


def set_polar_data(
    alpha_deg: np.ndarray, cl: np.ndarray, cd: np.ndarray
) -> None:
    """Set polar data for interpolation.

    This function allows setting polar data programmatically without loading from CSV.

    Args:
        alpha_deg: Angle of attack array in degrees
        cl: Lift coefficient array
        cd: Drag coefficient array

    Raises:
        ValueError: If arrays have different lengths or are empty
    """
    global _loaded_alpha, _loaded_cl, _loaded_cd

    alpha_deg = np.asarray(alpha_deg)
    cl = np.asarray(cl)
    cd = np.asarray(cd)

    if len(alpha_deg) == 0:
        # Set to None to indicate no data
        _loaded_alpha = None
        _loaded_cl = None
        _loaded_cd = None
        return

    if len(alpha_deg) != len(cl) or len(alpha_deg) != len(cd):
        raise ValueError("All arrays must have the same length")

    _loaded_alpha = alpha_deg
    _loaded_cl = cl
    _loaded_cd = cd


# Auto-load data when module is imported if load_polar_csv is called
def _reset_polar_data() -> None:
    """Reset loaded polar data (for testing)."""
    global _loaded_alpha, _loaded_cl, _loaded_cd
    _loaded_alpha = None
    _loaded_cl = None
    _loaded_cd = None
