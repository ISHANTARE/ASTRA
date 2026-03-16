"""
ASTRA Data Models (Pydantic Schemas)
Defines all request/response models and internal data structures.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


# ---------- Enums ----------

class ObjectType(str, Enum):
    SATELLITE = "satellite"
    DEBRIS = "debris"
    ROCKET_BODY = "rocket_body"
    UNKNOWN = "unknown"


class OrbitRegion(str, Enum):
    LEO = "LEO"
    MEO = "MEO"
    GEO = "GEO"
    HEO = "HEO"
    ALL = "ALL"


class RiskClassification(str, Enum):
    EXTREME = "Extreme"
    HIGH_RISK = "High Risk"
    CONJUNCTION = "Conjunction"
    SAFE = "Safe"


# ---------- Orbital Object ----------

class OrbitalObjectBase(BaseModel):
    """Core orbital object fields."""
    norad_id: int
    name: str
    object_type: ObjectType = ObjectType.UNKNOWN
    orbit_region: OrbitRegion = OrbitRegion.LEO
    tle_line1: str
    tle_line2: str


class OrbitalObjectDetail(OrbitalObjectBase):
    """Extended orbital object with derived orbital parameters."""
    inclination_deg: float = 0.0
    eccentricity: float = 0.0
    period_min: float = 0.0
    apogee_km: float = 0.0
    perigee_km: float = 0.0
    mean_motion: float = 0.0
    raan_deg: float = 0.0
    arg_perigee_deg: float = 0.0
    mean_anomaly_deg: float = 0.0
    epoch: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class OrbitalObjectResponse(BaseModel):
    """API response for a single orbital object."""
    id: int = Field(alias="norad_id")
    name: str
    type: ObjectType = Field(alias="object_type")
    tle_line1: str
    tle_line2: str

    class Config:
        populate_by_name = True


# ---------- Conjunction Event ----------

class ConjunctionEvent(BaseModel):
    """Represents a detected close approach between two objects."""
    object_1: str
    object_2: str
    object_1_id: int
    object_2_id: int
    closest_distance_km: float
    time_of_closest_approach_utc: datetime
    relative_velocity_km_s: float
    risk_classification: RiskClassification


# ---------- API Request/Response Models ----------

class PredictionRequest(BaseModel):
    """Request body for POST /api/predict/approaches."""
    region: OrbitRegion = OrbitRegion.LEO
    altitude_range: list[float] = Field(default=[400, 600], min_length=2, max_length=2)
    prediction_window_hours: int = 24
    time_resolution_mins: int = 5


class PredictionResponse(BaseModel):
    """Response for POST /api/predict/approaches."""
    objects_analyzed: int
    candidate_pairs: int
    events: list[ConjunctionEvent]
    computation_time_seconds: float


class DatasetRefreshResponse(BaseModel):
    """Response for GET /api/dataset/refresh."""
    status: str
    objects_updated: int
    timestamp: datetime


class CongestionBand(BaseModel):
    """Traffic density for a single altitude band."""
    altitude_min_km: float
    altitude_max_km: float
    total_objects: int
    satellites: int
    debris: int
    rocket_bodies: int


class CongestionResponse(BaseModel):
    """Response for GET /api/analytics/congestion."""
    bands: list[CongestionBand]
    total_objects: int
