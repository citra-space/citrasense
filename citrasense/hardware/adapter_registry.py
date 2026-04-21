"""Hardware adapter registry.

This module provides a centralized registry for all hardware adapters.
To add a new adapter, simply add an entry to the REGISTERED_ADAPTERS dict below.

Each adapter entry should include:
- module: The full module path to import
- class_name: The class name within that module
- description: A human-readable description of the adapter

Third-party adapters can be added by modifying this registry.
"""

import importlib

from citrasense.hardware.abstract_astro_hardware_adapter import AbstractAstroHardwareAdapter

# Central registry of all available hardware adapters
REGISTERED_ADAPTERS: dict[str, dict[str, str]] = {
    "indi": {
        "module": "citrasense.hardware.indi.indi_adapter",
        "class_name": "IndiAdapter",
        "description": "INDI Protocol - Universal astronomy device control",
    },
    "nina": {
        "module": "citrasense.hardware.nina.nina_adapter",
        "class_name": "NinaAdvancedHttpAdapter",
        "description": "N.I.N.A. Advanced HTTP API - Windows-based astronomy imaging",
    },
    "kstars": {
        "module": "citrasense.hardware.kstars.kstars_adapter",
        "class_name": "KStarsDBusAdapter",
        "description": "KStars/Ekos via D-Bus - Linux astronomy suite",
    },
    "direct": {
        "module": "citrasense.hardware.direct.direct_adapter",
        "class_name": "DirectHardwareAdapter",
        "description": "Direct Hardware Control - Composable device adapters for cameras, mounts, etc.",
    },
    "dummy": {
        "module": "citrasense.hardware.dummy_adapter",
        "class_name": "DummyAdapter",
        "description": "Dummy Adapter - Fake hardware for testing without real devices",
    },
}


def get_adapter_class(adapter_name: str) -> type[AbstractAstroHardwareAdapter]:
    """Get the adapter class for the given adapter name.

    Args:
        adapter_name: The name of the adapter (e.g., "indi", "nina", "kstars")

    Returns:
        The adapter class

    Raises:
        ValueError: If the adapter name is not registered
        ImportError: If the adapter module cannot be imported (e.g., missing dependencies)
    """
    if adapter_name not in REGISTERED_ADAPTERS:
        available = ", ".join(f"'{name}'" for name in REGISTERED_ADAPTERS.keys())
        raise ValueError(f"Unknown hardware adapter type: '{adapter_name}'. " f"Valid options are: {available}")

    adapter_info = REGISTERED_ADAPTERS[adapter_name]
    module = importlib.import_module(adapter_info["module"])
    adapter_class = getattr(module, adapter_info["class_name"])

    return adapter_class


def list_adapters() -> dict[str, dict[str, str]]:
    """Get a dictionary of all registered adapters with their descriptions.

    Returns:
        Dict mapping adapter names to their info (description, module, class_name)
    """
    return {
        name: {
            "description": info["description"],
            "module": info["module"],
            "class_name": info["class_name"],
        }
        for name, info in REGISTERED_ADAPTERS.items()
    }


def get_adapter_schema(adapter_name: str, **kwargs) -> list:
    """Get the configuration schema for a specific adapter.

    Args:
        adapter_name: The name of the adapter
        **kwargs: Additional arguments to pass to the adapter's get_settings_schema method
                  (e.g., current settings for dynamic schema generation)

    Returns:
        The adapter's settings schema

    Raises:
        ValueError: If the adapter name is not registered
        ImportError: If the adapter module cannot be imported
    """
    adapter_class = get_adapter_class(adapter_name)
    # Call classmethod, passing kwargs for dynamic schemas
    return adapter_class.get_settings_schema(**kwargs)
