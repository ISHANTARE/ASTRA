"""
ASTRA API Routes
FastAPI route handlers — no heavy computation, only orchestration of backend services.
"""

import time
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from app.core.config import settings
from app.models.schemas import (
    OrbitalObjectDetail,
    OrbitalObjectResponse,
    PredictionRequest,
    PredictionResponse,
    DatasetRefreshResponse,
    CongestionResponse,
    OrbitRegion,
    ObjectType,
)
from app.data.tle_parser import fetch_all_tles
from app.database.connection import async_session
from app.database.schema import OrbitalObjectDB
from app.orbit.propagator import propagate_batch, generate_simulation_times, propagate_to_time
from app.filtering.pipeline import run_filtering_pipeline
from app.analysis.conjunction import analyze_candidates
from app.analysis.congestion import compute_congestion

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


# ---------- Helper: DB ↔ Dict conversion ----------

def db_obj_to_dict(obj: OrbitalObjectDB) -> dict:
    """Convert a database ORM object to a plain dict."""
    return {
        "norad_id": obj.norad_id,
        "name": obj.name,
        "object_type": obj.object_type,
        "orbit_region": obj.orbit_region,
        "tle_line1": obj.tle_line1,
        "tle_line2": obj.tle_line2,
        "inclination_deg": obj.inclination_deg,
        "eccentricity": obj.eccentricity,
        "period_min": obj.period_min,
        "apogee_km": obj.apogee_km,
        "perigee_km": obj.perigee_km,
        "mean_motion": obj.mean_motion,
        "raan_deg": obj.raan_deg,
        "arg_perigee_deg": obj.arg_perigee_deg,
        "mean_anomaly_deg": obj.mean_anomaly_deg,
    }


# =============================================================================
# GET /api/dataset/refresh
# =============================================================================

@router.get("/dataset/refresh", response_model=DatasetRefreshResponse)
async def refresh_dataset():
    """
    Pull latest TLE data from CelesTrak and store in the database.
    Endpoint per doc 04: GET /api/dataset/refresh
    """
    logger.info("Starting dataset refresh from CelesTrak...")

    try:
        objects = await fetch_all_tles()

        if not objects:
            raise HTTPException(status_code=502, detail="Failed to fetch TLE data from CelesTrak")

        # Upsert into database
        async with async_session() as session:
            count = 0
            for obj in objects:
                existing = await session.get(OrbitalObjectDB, obj.norad_id)
                if existing:
                    # Update existing
                    existing.name = obj.name
                    existing.object_type = obj.object_type.value
                    existing.orbit_region = obj.orbit_region.value
                    existing.tle_line1 = obj.tle_line1
                    existing.tle_line2 = obj.tle_line2
                    existing.inclination_deg = obj.inclination_deg
                    existing.eccentricity = obj.eccentricity
                    existing.period_min = obj.period_min
                    existing.apogee_km = obj.apogee_km
                    existing.perigee_km = obj.perigee_km
                    existing.mean_motion = obj.mean_motion
                    existing.raan_deg = obj.raan_deg
                    existing.arg_perigee_deg = obj.arg_perigee_deg
                    existing.mean_anomaly_deg = obj.mean_anomaly_deg
                    existing.epoch = obj.epoch
                else:
                    # Insert new
                    db_obj = OrbitalObjectDB(
                        norad_id=obj.norad_id,
                        name=obj.name,
                        object_type=obj.object_type.value,
                        orbit_region=obj.orbit_region.value,
                        tle_line1=obj.tle_line1,
                        tle_line2=obj.tle_line2,
                        inclination_deg=obj.inclination_deg,
                        eccentricity=obj.eccentricity,
                        period_min=obj.period_min,
                        apogee_km=obj.apogee_km,
                        perigee_km=obj.perigee_km,
                        mean_motion=obj.mean_motion,
                        raan_deg=obj.raan_deg,
                        arg_perigee_deg=obj.arg_perigee_deg,
                        mean_anomaly_deg=obj.mean_anomaly_deg,
                        epoch=obj.epoch,
                    )
                    session.add(db_obj)
                count += 1

            await session.commit()

        logger.info(f"Dataset refresh complete: {count} objects updated")
        return DatasetRefreshResponse(
            status="success",
            objects_updated=count,
            timestamp=datetime.now(timezone.utc),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset refresh failed: {e}")
        raise HTTPException(status_code=500, detail=f"Dataset refresh failed: {str(e)}")


# =============================================================================
# GET /api/objects
# =============================================================================

@router.get("/objects")
async def get_objects(
    region: Optional[OrbitRegion] = Query(None, description="Filter by orbit region"),
    type: Optional[ObjectType] = Query(None, description="Filter by object type"),
    limit: int = Query(1000, ge=1, le=50000, description="Max objects to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    """
    Return orbital objects based on filter parameters.
    Endpoint per doc 04: GET /api/objects
    """
    async with async_session() as session:
        query = select(OrbitalObjectDB)

        if region and region != OrbitRegion.ALL:
            query = query.where(OrbitalObjectDB.orbit_region == region.value)
        if type:
            query = query.where(OrbitalObjectDB.object_type == type.value)

        query = query.offset(offset).limit(limit)
        result = await session.execute(query)
        db_objects = result.scalars().all()

    objects = []
    for obj in db_objects:
        objects.append({
            "id": obj.norad_id,
            "name": obj.name,
            "type": obj.object_type,
            "orbit_region": obj.orbit_region,
            "tle_line1": obj.tle_line1,
            "tle_line2": obj.tle_line2,
            "inclination_deg": obj.inclination_deg,
            "eccentricity": obj.eccentricity,
            "apogee_km": obj.apogee_km,
            "perigee_km": obj.perigee_km,
        })

    return objects


# =============================================================================
# GET /api/positions
# =============================================================================

@router.get("/positions")
async def get_positions(
    time_step: int = Query(0, description="Simulation time step (0 to 288 for 24h at 5min)"),
    region: Optional[OrbitRegion] = Query(None, description="Filter by orbit region"),
    type: Optional[ObjectType] = Query(None, description="Filter by object type"),
):
    """
    Return explicit [x,y,z] arrays for all requested objects at a specific time step.
    This fulfills doc 08 architecture requirements.
    
    Returns:
        JSON with:
        - positions: flat array of [x, y, z] values for Three.js InstancedMesh
        - ids: parallel array of norad_ids to map the positions back to the objects
    """
    start_time_perf = time.time()
    
    async with async_session() as session:
        query = select(OrbitalObjectDB.norad_id, OrbitalObjectDB.tle_line1, OrbitalObjectDB.tle_line2)

        if region and region != OrbitRegion.ALL:
            query = query.where(OrbitalObjectDB.orbit_region == region.value)
        if type:
            query = query.where(OrbitalObjectDB.object_type == type.value)

        result = await session.execute(query)
        db_objects = result.all()

    if not db_objects:
        return {"positions": [], "ids": []}

    # Convert to expected dict format
    objects = [
        {
            "norad_id": row.norad_id,
            "tle_line1": row.tle_line1,
            "tle_line2": row.tle_line2,
        }
        for row in db_objects
    ]

    # Propagate to specific time step
    flat_positions, ids_list = propagate_to_time(objects, time_step)

    elapsed = round((time.time() - start_time_perf) * 1000, 2)
    logger.info(f"Computed positions for {len(objects)} objects at step {time_step} in {elapsed}ms")

    return {
        "positions": flat_positions.tolist(),
        "ids": ids_list,
    }


# =============================================================================
# GET /api/object/{id}
# =============================================================================

@router.get("/object/{norad_id}")
async def get_object(norad_id: int):
    """
    Return detailed data for a specific orbital object.
    Endpoint per doc 04: GET /api/object/{id}
    """
    async with async_session() as session:
        obj = await session.get(OrbitalObjectDB, norad_id)

    if not obj:
        raise HTTPException(status_code=404, detail=f"Object {norad_id} not found")

    return {
        "id": obj.norad_id,
        "name": obj.name,
        "type": obj.object_type,
        "orbit_region": obj.orbit_region,
        "tle_line1": obj.tle_line1,
        "tle_line2": obj.tle_line2,
        "inclination_deg": obj.inclination_deg,
        "eccentricity": obj.eccentricity,
        "period_min": obj.period_min,
        "apogee_km": obj.apogee_km,
        "perigee_km": obj.perigee_km,
        "mean_motion": obj.mean_motion,
        "raan_deg": obj.raan_deg,
        "arg_perigee_deg": obj.arg_perigee_deg,
        "mean_anomaly_deg": obj.mean_anomaly_deg,
        "epoch": obj.epoch.isoformat() if obj.epoch else None,
        "updated_at": obj.updated_at.isoformat() if obj.updated_at else None,
    }


# =============================================================================
# POST /api/predict/approaches
# =============================================================================

@router.post("/predict/approaches", response_model=PredictionResponse)
async def predict_approaches(request: PredictionRequest):
    """
    Run orbital prediction and close approach detection.
    Endpoint per doc 04: POST /api/predict/approaches

    Pipeline:
    1. Fetch objects from DB
    2. Run multi-stage filtering pipeline
    3. Precompute trajectories for filtered objects (SGP4)
    4. Analyze candidate pairs for conjunctions
    5. Return events
    """
    start_time_perf = time.time()
    logger.info(
        f"Prediction request: region={request.region}, "
        f"altitude={request.altitude_range}, window={request.prediction_window_hours}h"
    )

    # Step 1: Fetch all objects from database
    async with async_session() as session:
        result = await session.execute(select(OrbitalObjectDB))
        db_objects = result.scalars().all()

    if not db_objects:
        raise HTTPException(
            status_code=404,
            detail="No orbital objects in database. Run /api/dataset/refresh first."
        )

    # Convert to dicts
    objects = [db_obj_to_dict(obj) for obj in db_objects]
    object_names = {obj["norad_id"]: obj["name"] for obj in objects}

    # Step 2: Run multi-stage filtering pipeline
    filtered_objects, candidate_pairs = run_filtering_pipeline(
        objects=objects,
        region=request.region,
        altitude_range=tuple(request.altitude_range),
    )

    if not candidate_pairs:
        return PredictionResponse(
            objects_analyzed=len(filtered_objects),
            candidate_pairs=0,
            events=[],
            computation_time_seconds=round(time.time() - start_time_perf, 2),
        )

    # Step 3: Generate simulation times and precompute trajectories
    sim_times, jd_array, fr_array = generate_simulation_times(
        window_hours=request.prediction_window_hours,
        resolution_minutes=request.time_resolution_mins,
    )

    # Only propagate filtered objects (not all 30k)
    objects_to_propagate = [
        {"norad_id": obj["norad_id"], "tle_line1": obj["tle_line1"], "tle_line2": obj["tle_line2"]}
        for obj in filtered_objects
    ]
    trajectories = propagate_batch(objects_to_propagate, jd_array, fr_array)

    # Step 4: Analyze candidate pairs using precomputed trajectories
    events = analyze_candidates(
        candidate_pairs=candidate_pairs,
        trajectories=trajectories,
        simulation_times=sim_times,
        object_names=object_names,
    )

    elapsed = round(time.time() - start_time_perf, 2)
    logger.info(
        f"Prediction complete: {len(filtered_objects)} objects analyzed, "
        f"{len(candidate_pairs)} pairs checked, {len(events)} events found in {elapsed}s"
    )

    return PredictionResponse(
        objects_analyzed=len(filtered_objects),
        candidate_pairs=len(candidate_pairs),
        events=events,
        computation_time_seconds=elapsed,
    )


# =============================================================================
# GET /api/analytics/congestion
# =============================================================================

@router.get("/analytics/congestion", response_model=CongestionResponse)
async def get_congestion(
    region: Optional[OrbitRegion] = Query(None, description="Filter by orbit region"),
    max_altitude_km: float = Query(2000.0, description="Max altitude to analyze"),
):
    """
    Return spatial density calculations grouped by altitude bands.
    Endpoint per doc 04: GET /api/analytics/congestion
    """
    async with async_session() as session:
        query = select(OrbitalObjectDB)
        if region and region != OrbitRegion.ALL:
            query = query.where(OrbitalObjectDB.orbit_region == region.value)
        result = await session.execute(query)
        db_objects = result.scalars().all()

    objects = [db_obj_to_dict(obj) for obj in db_objects]
    return compute_congestion(objects, max_altitude_km=max_altitude_km)
