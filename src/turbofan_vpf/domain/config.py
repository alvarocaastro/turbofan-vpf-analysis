"""Global configuration for compressibility corrections and model parameters."""

import warnings
from dataclasses import dataclass


@dataclass(frozen=True)
class CompressibilityConfig:
    """Configuration for compressibility corrections.

    Attributes:
        use_compressibility: Whether to apply compressibility corrections
        model: Compressibility model to use ("PG" for Prandtl-Glauert)
        mach_max_valid: Maximum Mach number for which corrections are valid
    """

    use_compressibility: bool = True
    model: str = "PG"  # Prandtl-Glauert
    mach_max_valid: float = 0.7

    def validate_mach(self, mach: float, phase_name: str = "") -> None:
        """Validate Mach number against maximum valid value.

        Args:
            mach: Mach number to validate
            phase_name: Optional phase name for warning message

        Raises:
            UserWarning: If Mach number exceeds maximum valid value
        """
        if mach > self.mach_max_valid:
            phase_str = f" for phase '{phase_name}'" if phase_name else ""
            warnings.warn(
                f"Mach number {mach:.2f}{phase_str} exceeds maximum valid value "
                f"({self.mach_max_valid:.2f}) for {self.model} compressibility model. "
                "Results may be inaccurate due to transonic effects.",
                UserWarning,
                stacklevel=3,
            )


# Global configuration instance
_config = CompressibilityConfig()


def get_config() -> CompressibilityConfig:
    """Get the global compressibility configuration.

    Returns:
        Current compressibility configuration
    """
    return _config


def set_config(config: CompressibilityConfig) -> None:
    """Set the global compressibility configuration.

    Args:
        config: New compressibility configuration
    """
    global _config
    _config = config
