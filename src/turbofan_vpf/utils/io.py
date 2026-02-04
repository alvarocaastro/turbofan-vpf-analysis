"""Input/output utility functions."""

import numpy as np
from pathlib import Path
from typing import Tuple


def load_polar_data(filepath: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load airfoil polar data from a CSV file.

    Expected format: alpha, cl, cd (comma-separated)

    Args:
        filepath: Path to the CSV file

    Returns:
        Tuple of (alpha, cl, cd) arrays

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file format is invalid
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Polar file not found: {filepath}")

    try:
        data = np.loadtxt(filepath, delimiter=",", skiprows=1)
        if data.shape[1] < 3:
            raise ValueError("File must contain at least 3 columns: alpha, cl, cd")

        alpha = data[:, 0]
        cl = data[:, 1]
        cd = data[:, 2]

        return alpha, cl, cd
    except Exception as e:
        raise ValueError(f"Error reading polar file: {e}") from e


def save_polar_data(
    filepath: Path,
    alpha: np.ndarray,
    cl: np.ndarray,
    cd: np.ndarray,
) -> None:
    """Save airfoil polar data to a CSV file.

    Args:
        filepath: Path to save the CSV file
        alpha: Angle of attack array in degrees
        cl: Lift coefficient array
        cd: Drag coefficient array
    """
    data = np.column_stack([alpha, cl, cd])
    header = "alpha,cl,cd"
    np.savetxt(filepath, data, delimiter=",", header=header, comments="")
