"""Abstract focuser device interface."""

from abc import abstractmethod

from citrasense.hardware.devices.abstract_hardware_device import AbstractHardwareDevice


class AbstractFocuser(AbstractHardwareDevice):
    """Abstract base class for focuser devices.

    Provides a common interface for controlling motorized focusers.
    """

    @abstractmethod
    def move_absolute(self, position: int) -> bool:
        """Move to absolute focuser position.

        Args:
            position: Target position in steps

        Returns:
            True if move initiated successfully, False otherwise
        """
        pass

    @abstractmethod
    def move_relative(self, steps: int) -> bool:
        """Move focuser by relative number of steps.

        Args:
            steps: Number of steps to move (positive=outward, negative=inward)

        Returns:
            True if move initiated successfully, False otherwise
        """
        pass

    @abstractmethod
    def get_position(self) -> int | None:
        """Get current focuser position.

        Returns:
            Current position in steps, or None if unavailable
        """
        pass

    @abstractmethod
    def is_moving(self) -> bool:
        """Check if focuser is currently moving.

        Returns:
            True if moving, False if stationary
        """
        pass

    @abstractmethod
    def abort_move(self):
        """Stop the current focuser movement."""
        pass

    @abstractmethod
    def get_max_position(self) -> int | None:
        """Get the maximum focuser position.

        Returns:
            Maximum position in steps, or None if unlimited/unknown
        """
        pass

    @abstractmethod
    def get_temperature(self) -> float | None:
        """Get focuser temperature reading if available.

        Returns:
            Temperature in degrees Celsius, or None if not available
        """
        pass
