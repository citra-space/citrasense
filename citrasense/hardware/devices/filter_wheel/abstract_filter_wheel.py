"""Abstract filter wheel device interface."""

from abc import abstractmethod

from citrasense.hardware.devices.abstract_hardware_device import AbstractHardwareDevice


class AbstractFilterWheel(AbstractHardwareDevice):
    """Abstract base class for filter wheel devices.

    Provides a common interface for controlling motorized filter wheels.
    """

    @abstractmethod
    def set_filter_position(self, position: int) -> bool:
        """Move to specified filter position.

        Args:
            position: Filter position (0-indexed)

        Returns:
            True if move initiated successfully, False otherwise
        """
        pass

    @abstractmethod
    def get_filter_position(self) -> int | None:
        """Get current filter position.

        Returns:
            Current filter position (0-indexed), or None if unavailable
        """
        pass

    @abstractmethod
    def is_moving(self) -> bool:
        """Check if filter wheel is currently moving.

        Returns:
            True if moving, False if stationary
        """
        pass

    @abstractmethod
    def get_filter_count(self) -> int:
        """Get the number of filter positions.

        Returns:
            Number of available filter positions
        """
        pass

    @abstractmethod
    def get_filter_names(self) -> list[str]:
        """Get the names of all filters.

        Returns:
            List of filter names for each position
        """
        pass

    @abstractmethod
    def set_filter_names(self, names: list[str]) -> bool:
        """Set the names for all filter positions.

        Args:
            names: List of filter names (must match filter count)

        Returns:
            True if names set successfully, False otherwise
        """
        pass
