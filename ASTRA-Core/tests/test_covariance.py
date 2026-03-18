import pytest
import numpy as np

from astra.covariance import compute_collision_probability, estimate_covariance

def test_estimate_covariance():
    cov = estimate_covariance(time_since_epoch_days=1.0)
    assert cov.shape == (3, 3)
    assert np.all(np.diag(cov) > 0)
    
def test_compute_collision_probability_zero_miss():
    miss_vec = np.zeros(3)
    rel_vel = np.array([0.0, 7.0, 0.0])
    cov_a = np.eye(3) * 10.0
    cov_b = np.eye(3) * 5.0
    
    p = compute_collision_probability(miss_vec, rel_vel, cov_a, cov_b, 0.010)
    assert p > 0.0
    assert p <= 1.0

def test_compute_collision_probability_far_miss():
    miss_vec = np.array([1000.0, 0.0, 0.0])
    rel_vel = np.array([0.0, 7.0, 0.0])
    cov_a = np.eye(3) * 1.0
    cov_b = np.eye(3) * 1.0
    
    p = compute_collision_probability(miss_vec, rel_vel, cov_a, cov_b, 0.010)
    assert p < 1e-10

def test_compute_collision_probability_deterministic():
    miss_vec = np.array([0.005, 0.0, 0.0])  # 5 meters
    rel_vel = np.array([0.0, 7.0, 0.0])
    cov_a = np.zeros((3, 3))
    cov_b = np.zeros((3, 3))
    
    # 0 variance, miss < 10m combined radius
    p = compute_collision_probability(miss_vec, rel_vel, cov_a, cov_b, 0.010)
    assert p == 1.0
    
    miss_vec_far = np.array([0.015, 0.0, 0.0]) # 15 meters
    p2 = compute_collision_probability(miss_vec_far, rel_vel, cov_a, cov_b, 0.010)
    assert p2 == 0.0
