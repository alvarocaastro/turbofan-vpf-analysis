"""Path utilities for project directories and files."""

from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """Get the project root directory.

    Assumes this file is in src/turbofan_vpf/utils/ and project root is 3 levels up.

    Returns:
        Path to project root directory
    """
    return Path(__file__).parent.parent.parent.parent


def get_data_dir() -> Path:
    """Get the data directory path.

    Returns:
        Path to data directory
    """
    return get_project_root() / "data"


def get_polars_dir() -> Path:
    """Get the polars data directory path.

    Returns:
        Path to polars directory
    """
    return get_data_dir() / "polars"


def get_outputs_dir() -> Path:
    """Get the outputs directory path.

    Returns:
        Path to outputs directory
    """
    return get_project_root() / "outputs"


def get_figures_dir() -> Path:
    """Get the figures output directory path.

    Returns:
        Path to figures directory
    """
    return get_outputs_dir() / "figures"


def get_polar_file(filename: str = "example.csv") -> Path:
    """Get path to a polar data file.

    Args:
        filename: Name of the polar file (default: "example.csv")

    Returns:
        Path to the polar file
    """
    return get_polars_dir() / filename


def get_results_csv(filename: str = "results_by_phase.csv") -> Path:
    """Get path to a results CSV file.

    Args:
        filename: Name of the results file (default: "results_by_phase.csv")

    Returns:
        Path to the results CSV file
    """
    return get_outputs_dir() / filename


def get_figure_path(filename: str) -> Path:
    """Get path to a figure file.

    Args:
        filename: Name of the figure file (e.g., "plot.png")

    Returns:
        Path to the figure file
    """
    return get_figures_dir() / filename


def ensure_dir_exists(path: Path) -> None:
    """Ensure that a directory exists, creating it if necessary.

    Args:
        path: Path to the directory
    """
    path.mkdir(parents=True, exist_ok=True)
