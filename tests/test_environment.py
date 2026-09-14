"""Vertical-slice tests: environment (green now) + sensors (your task).

The sensor tests use the reference/measurement structure that TRIAD will need:
a sensor reading is the true reference vector rotated into the body frame, plus
noise. The determination slice (TRIAD/QUEST) will invert exactly this.
"""
import numpy as np

from adcs_sim.quaternion import from_axis_angle, apply, normalize
from adcs_sim.environment import (
    R_EARTH, magnetic_field_eci, sun_vector_eci, in_eclipse,
    sun_sensor, magnetometer,
)


# ---------- environment (implemented) ----------
def test_dipole_magnitude_is_physical_at_leo():
    r = np.array([R_EARTH + 500.0, 0, 0])
    B = magnetic_field_eci(r)
    assert 15000 < np.linalg.norm(B) < 50000        # nT, sane LEO range


def test_dipole_falls_off_as_r_cubed():
    B1 = np.linalg.norm(magnetic_field_eci([R_EARTH + 500, 0, 0]))
    B2 = np.linalg.norm(magnetic_field_eci([2*(R_EARTH + 500), 0, 0]))
    assert np.isclose(B1 / B2, 8.0, rtol=1e-6)      # doubling r -> 1/8 field


def test_sun_vector_is_unit_and_tracks_ecliptic():
    s0 = sun_vector_eci(0.0)
    assert np.isclose(np.linalg.norm(s0), 1.0)
    assert np.allclose(s0, [1, 0, 0], atol=1e-9)    # equinox: Sun on +X (vernal eq.)
    s90 = sun_vector_eci(365.25 / 4)                # quarter year later
    assert s90[2] > 0.3                             # lifted out of equator by obliquity


def test_eclipse_geometry():
    sun = np.array([1.0, 0, 0])
    assert in_eclipse([-(R_EARTH + 500), 0, 0], sun)     # directly behind Earth
    assert not in_eclipse([R_EARTH + 500, 0, 0], sun)    # sunward side
    assert not in_eclipse([0, R_EARTH + 500, 0], sun)    # beside Earth, lit


# ---------- sun sensor (your task) ----------
def test_sun_sensor_noiseless_matches_apply():
    q = from_axis_angle([0.3, -0.5, 0.2], 0.9)
    s_eci = normalize([0.6, 0.7, -0.4])
    meas = sun_sensor(q, s_eci, sigma_deg=0.0)
    assert np.allclose(meas, apply(q, s_eci), atol=1e-12)


def test_sun_sensor_returns_unit_vector():
    q = from_axis_angle([1, 0, 0], 0.5)
    meas = sun_sensor(q, normalize([1, 1, 1]), sigma_deg=2.0,
                      rng=np.random.default_rng(0))
    assert np.isclose(np.linalg.norm(meas), 1.0, atol=1e-9)


def test_sun_sensor_noise_is_the_right_size():
    """Average angular error over many samples should be ~sigma_deg."""
    q = from_axis_angle([0.2, 0.4, 0.5], 1.1)
    s_eci = normalize([0.3, -0.6, 0.7])
    truth = apply(q, s_eci)
    rng = np.random.default_rng(1)
    errs = []
    for _ in range(4000):
        m = sun_sensor(q, s_eci, sigma_deg=1.5, rng=rng)
        errs.append(np.degrees(np.arccos(np.clip(np.dot(m, truth), -1, 1))))
    assert 0.8 < np.mean(errs) < 2.5                # right ballpark, not exact


# ---------- magnetometer (your task) ----------
def test_magnetometer_noiseless_matches_apply():
    q = from_axis_angle([0.1, 0.2, -0.3], 0.7)
    B_eci = magnetic_field_eci([R_EARTH + 600, 1000, -2000])
    meas = magnetometer(q, B_eci, sigma_nT=0.0)
    assert np.allclose(meas, apply(q, B_eci), atol=1e-9)


def test_magnetometer_preserves_magnitude_noiseless():
    """A magnetometer measures magnitude too -- rotation must not change |B|."""
    q = from_axis_angle([0.5, -0.2, 0.8], 2.0)
    B_eci = magnetic_field_eci([R_EARTH + 400, 0, 3000])
    meas = magnetometer(q, B_eci, sigma_nT=0.0)
    assert np.isclose(np.linalg.norm(meas), np.linalg.norm(B_eci), atol=1e-6)


def test_magnetometer_bias_is_applied():
    q = np.array([1.0, 0, 0, 0])                    # identity: body == eci
    B_eci = np.array([100.0, 0, 0])
    bias = np.array([50.0, -30.0, 10.0])
    meas = magnetometer(q, B_eci, sigma_nT=0.0, bias_nT=bias)
    assert np.allclose(meas, B_eci + bias, atol=1e-9)


def test_magnetometer_noise_size():
    q = from_axis_angle([1, 1, 1], 1.0)
    B_eci = np.array([20000.0, 5000, -10000])
    rng = np.random.default_rng(2)
    samples = np.array([magnetometer(q, B_eci, sigma_nT=300.0, rng=rng)
                        for _ in range(3000)])
    per_axis_std = samples.std(axis=0)
    assert np.all(np.abs(per_axis_std - 300.0) < 40.0)
