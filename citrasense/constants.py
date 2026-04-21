"""Application-wide constants for CitraSense.

This module contains all shared constants used across different parts of the application.
Centralizing these values prevents duplication and circular import issues.
"""

# ============================================================================
# API ENDPOINTS
# ============================================================================
PROD_API_HOST = "api.citra.space"
DEV_API_HOST = "dev.api.citra.space"
DEFAULT_API_PORT = 443

# ============================================================================
# WEB APP URLs
# ============================================================================
PROD_APP_URL = "https://app.citra.space"
DEV_APP_URL = "https://dev.app.citra.space"

# ============================================================================
# WEB SERVER
# ============================================================================
DEFAULT_WEB_PORT = 24872  # "CITRA" on phone keypad

# ============================================================================
# AUTOFOCUS TARGET PRESETS
# Bright stars suitable as autofocus reference targets.
# RA/Dec in degrees (J2000). Ordered roughly by year-round utility.
# ============================================================================
AUTOFOCUS_TARGET_PRESETS = {
    "current": {
        "name": "Current position",
        "designation": "No slew",
        "ra": None,
        "dec": None,
        "mag": None,
        "description": "Focus where the telescope is currently pointing (no slew)",
    },
    "mirach": {
        "name": "Mirach",
        "designation": "Beta Andromedae",
        "ra": (1 + 9 / 60.0 + 43.92 / 3600.0) * 15,  # 17.43 deg
        "dec": 35 + 37 / 60.0 + 14.0 / 3600.0,  # +35.62 deg
        "mag": 2.05,
        "description": "Bright fall/winter star, good default",
    },
    "polaris": {
        "name": "Polaris",
        "designation": "Alpha Ursae Minoris",
        "ra": (2 + 31 / 60.0 + 49.09 / 3600.0) * 15,  # 37.95 deg
        "dec": 89 + 15 / 60.0 + 50.8 / 3600.0,  # +89.26 deg
        "mag": 1.98,
        "description": "Always visible from northern latitudes",
    },
    "vega": {
        "name": "Vega",
        "designation": "Alpha Lyrae",
        "ra": (18 + 36 / 60.0 + 56.34 / 3600.0) * 15,  # 279.23 deg
        "dec": 38 + 47 / 60.0 + 1.3 / 3600.0,  # +38.78 deg
        "mag": 0.03,
        "description": "Very bright summer star",
    },
    "capella": {
        "name": "Capella",
        "designation": "Alpha Aurigae",
        "ra": (5 + 16 / 60.0 + 41.36 / 3600.0) * 15,  # 79.17 deg
        "dec": 45 + 59 / 60.0 + 52.8 / 3600.0,  # +46.00 deg
        "mag": 0.08,
        "description": "Bright winter circumpolar star",
    },
    "arcturus": {
        "name": "Arcturus",
        "designation": "Alpha Bootis",
        "ra": (14 + 15 / 60.0 + 39.67 / 3600.0) * 15,  # 213.92 deg
        "dec": 19 + 10 / 60.0 + 56.7 / 3600.0,  # +19.18 deg
        "mag": -0.05,
        "description": "Bright spring/summer star",
    },
    "sirius": {
        "name": "Sirius",
        "designation": "Alpha Canis Majoris",
        "ra": (6 + 45 / 60.0 + 8.92 / 3600.0) * 15,  # 101.29 deg
        "dec": -(16 + 42 / 60.0 + 58.0 / 3600.0),  # -16.72 deg
        "mag": -1.46,
        "description": "Brightest star, good for southern sites",
    },
    "regulus": {
        "name": "Regulus",
        "designation": "Alpha Leonis",
        "ra": (10 + 8 / 60.0 + 22.31 / 3600.0) * 15,  # 152.09 deg
        "dec": 11 + 58 / 60.0 + 1.9 / 3600.0,  # +11.97 deg
        "mag": 1.40,
        "description": "Bright spring star near ecliptic",
    },
    "canopus": {
        "name": "Canopus",
        "designation": "Alpha Carinae",
        "ra": (6 + 23 / 60.0 + 57.11 / 3600.0) * 15,  # 95.99 deg
        "dec": -(52 + 41 / 60.0 + 44.4 / 3600.0),  # -52.70 deg
        "mag": -0.74,
        "description": "Second brightest star, deep southern sky",
    },
    "achernar": {
        "name": "Achernar",
        "designation": "Alpha Eridani",
        "ra": (1 + 37 / 60.0 + 42.85 / 3600.0) * 15,  # 24.43 deg
        "dec": -(57 + 14 / 60.0 + 12.3 / 3600.0),  # -57.24 deg
        "mag": 0.46,
        "description": "Bright southern circumpolar star",
    },
    "fomalhaut": {
        "name": "Fomalhaut",
        "designation": "Alpha Piscis Austrini",
        "ra": (22 + 57 / 60.0 + 39.05 / 3600.0) * 15,  # 344.41 deg
        "dec": -(29 + 37 / 60.0 + 20.1 / 3600.0),  # -29.62 deg
        "mag": 1.16,
        "description": "Bright autumn star, visible from both hemispheres",
    },
}
