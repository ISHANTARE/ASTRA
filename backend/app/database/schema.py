"""
ASTRA Database Schema
SQLAlchemy ORM model for the orbital_objects table.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Enum as SAEnum
from sqlalchemy.sql import func

from app.database.connection import Base


class OrbitalObjectDB(Base):
    """ORM model representing a tracked space object."""

    __tablename__ = "orbital_objects"

    norad_id = Column(Integer, primary_key=True, index=True)
    name = Column(String(128), nullable=False)
    object_type = Column(String(20), nullable=False, default="unknown")
    orbit_region = Column(String(10), nullable=False, default="LEO")

    # TLE data
    tle_line1 = Column(String(70), nullable=False)
    tle_line2 = Column(String(70), nullable=False)

    # Orbital elements (derived from TLE)
    inclination_deg = Column(Float, default=0.0)
    eccentricity = Column(Float, default=0.0)
    period_min = Column(Float, default=0.0)
    apogee_km = Column(Float, default=0.0)
    perigee_km = Column(Float, default=0.0)
    mean_motion = Column(Float, default=0.0)
    raan_deg = Column(Float, default=0.0)
    arg_perigee_deg = Column(Float, default=0.0)
    mean_anomaly_deg = Column(Float, default=0.0)

    # Metadata
    epoch = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<OrbitalObject {self.norad_id}: {self.name}>"
