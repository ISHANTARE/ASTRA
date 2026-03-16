"""
ASTRA Core Configuration
Simulation constants, CelesTrak URLs, and application settings.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    # --- Application ---
    APP_NAME: str = "ASTRA - Autonomous Space Traffic Risk Analyzer"
    DEBUG: bool = True

    # --- Database ---
    DATABASE_URL: str = "sqlite+aiosqlite:///./astra.db"

    # --- Simulation Parameters (from doc 08) ---
    PREDICTION_WINDOW_HOURS: int = 24
    TIME_RESOLUTION_MINUTES: int = 5
    TOTAL_SIMULATION_STEPS: int = 288  # 24h * 60min / 5min

    # --- CelesTrak TLE Sources ---
    CELESTRAK_BASE_URL: str = "https://celestrak.org/NORAD/elements/gp.php"
    CELESTRAK_GROUPS: list[str] = [
        "active",           # Active satellites (~9k)
        "analyst",          # Analyst objects
        "cosmos-1408-debris",  # Debris fragments
        "fengyun-1c-debris",   # Debris fragments
        "iridium-33-debris",   # Debris fragments
        "cosmos-2251-debris",  # Debris fragments
    ]

    # --- Filtering Thresholds ---
    ALTITUDE_BAND_WIDTH_KM: float = 100.0   # Width of altitude bands for spatial grid
    ALTITUDE_FILTER_MARGIN_KM: float = 50.0  # Margin for altitude overlap check
    INCLINATION_THRESHOLD_DEG: float = 2.0   # Max inclination difference for plane filter
    ANGULAR_SECTOR_SIZE_DEG: float = 10.0    # Size of angular sectors for grid indexing

    # --- Risk Classification Thresholds (from doc 03) ---
    RISK_EXTREME_KM: float = 0.1    # < 100m
    RISK_HIGH_KM: float = 1.0       # < 1 km
    RISK_CONJUNCTION_KM: float = 10.0  # < 10 km

    # --- Orbit Region Altitude Boundaries (from doc 03) ---
    LEO_MIN_KM: float = 160.0
    LEO_MAX_KM: float = 2000.0
    MEO_MIN_KM: float = 2000.0
    MEO_MAX_KM: float = 35000.0
    GEO_ALTITUDE_KM: float = 35786.0
    GEO_TOLERANCE_KM: float = 200.0  # ± tolerance for GEO classification

    # --- Earth Constants ---
    EARTH_RADIUS_KM: float = 6371.0
    EARTH_MU: float = 398600.4418  # km³/s² gravitational parameter

    # --- CORS ---
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
