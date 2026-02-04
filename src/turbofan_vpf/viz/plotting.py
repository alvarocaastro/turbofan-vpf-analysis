"""Plotting and visualization functions."""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional


def plot_polar(
    alpha: np.ndarray,
    cl: np.ndarray,
    cd: np.ndarray,
    label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot airfoil polar (cl vs cd).

    Args:
        alpha: Angle of attack array in degrees
        cl: Lift coefficient array
        cd: Drag coefficient array
        label: Label for the plot legend
        ax: Matplotlib axes to plot on. If None, creates new figure

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(cd, cl, label=label, marker="o", markersize=4)
    ax.set_xlabel("Drag Coefficient, $C_d$")
    ax.set_ylabel("Lift Coefficient, $C_l$")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return ax


def plot_lift_curve(
    alpha: np.ndarray,
    cl: np.ndarray,
    label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot lift curve (cl vs alpha).

    Args:
        alpha: Angle of attack array in degrees
        cl: Lift coefficient array
        label: Label for the plot legend
        ax: Matplotlib axes to plot on. If None, creates new figure

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(alpha, cl, label=label, marker="o", markersize=4)
    ax.set_xlabel("Angle of Attack, $\\alpha$ [deg]")
    ax.set_ylabel("Lift Coefficient, $C_l$")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return ax
