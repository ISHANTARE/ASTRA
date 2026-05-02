import pytest
import numpy as np

from astra.covariance import (
    compute_collision_probability,
    compute_collision_probability_mc,
    estimate_covariance,
    rotate_covariance_rtn_to_eci,
)


def test_estimate_covariance():
    cov = estimate_covariance(time_since_epoch_days=1.0)
    assert cov.shape == (3, 3)
    assert np.all(np.diag(cov) > 0)


def test_compute_collision_probability_zero_miss():
    miss_vec = np.zeros(3)
    rel_vel = np.array([0.0, 7.0, 0.0])
    cov_a = np.eye(3) * 10.0
    cov_b = np.eye(3) * 5.0

    p = compute_collision_probability(
        miss_vec, rel_vel, cov_a, cov_b, radius_a_km=0.005, radius_b_km=0.005
    )
    assert p > 0.0
    assert p <= 1.0


def test_compute_collision_probability_far_miss():
    miss_vec = np.array([1000.0, 0.0, 0.0])
    rel_vel = np.array([0.0, 7.0, 0.0])
    cov_a = np.eye(3) * 1.0
    cov_b = np.eye(3) * 1.0

    p = compute_collision_probability(
        miss_vec, rel_vel, cov_a, cov_b, radius_a_km=0.005, radius_b_km=0.005
    )
    assert p < 1e-10


def test_compute_collision_probability_deterministic():
    miss_vec = np.array([0.005, 0.0, 0.0])  # 5 meters
    rel_vel = np.array([0.0, 7.0, 0.0])
    cov_a = np.zeros((3, 3))
    cov_b = np.zeros((3, 3))

    # 0 variance, miss < 10m combined radius
    p = compute_collision_probability(
        miss_vec, rel_vel, cov_a, cov_b, radius_a_km=0.005, radius_b_km=0.005
    )
    assert p == 1.0

    miss_vec_far = np.array([0.015, 0.0, 0.0])  # 15 meters
    p2 = compute_collision_probability(
        miss_vec_far, rel_vel, cov_a, cov_b, radius_a_km=0.005, radius_b_km=0.005
    )
    assert p2 == 0.0


def test_compute_collision_probability_accepts_6x6_covariance():
    miss_vec = np.array([0.05, 0.0, 0.0])
    rel_vel = np.array([0.0, 0.0, 10.0])
    cov = np.diag([0.05**2, 0.05**2, 0.05**2, 1e-8, 1e-8, 1e-8])

    p = compute_collision_probability(
        miss_vec, rel_vel, cov, cov, radius_a_km=0.005, radius_b_km=0.005
    )

    assert np.isfinite(p)
    assert 0.0 <= p <= 1.0


def test_compute_collision_probability_rejects_mixed_covariance_shapes():
    with pytest.raises(ValueError, match="matching dimensions"):
        compute_collision_probability(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 7.0, 0.0]),
            np.eye(3),
            np.eye(6),
        )


def test_compute_collision_probability_rejects_negative_radius():
    with pytest.raises(ValueError, match="non-negative"):
        compute_collision_probability(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 7.0, 0.0]),
            np.eye(3),
            np.eye(3),
            radius_a_km=-0.001,
        )


def test_compute_collision_probability_mc_rejects_invalid_sample_count():
    with pytest.raises(ValueError, match="n_samples"):
        compute_collision_probability_mc(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 7.0, 0.0]),
            np.eye(6),
            np.eye(6),
            n_samples=0,
        )


def test_rotate_covariance_rtn_to_eci_rejects_bad_shape():
    with pytest.raises(ValueError, match="cov_rtn"):
        rotate_covariance_rtn_to_eci(
            np.eye(6),
            np.array([7000.0, 0.0, 0.0]),
            np.array([0.0, 7.5, 0.0]),
        )
