"""DET1 (TRIAD) mastery tests. When all pass, determination is closed.

Red until you implement triad_frame.
"""
import numpy as np
import pytest

from adcs_sim.quaternion import from_axis_angle, to_dcm, apply, normalize
from adcs_sim.determination import triad_frame, triad_attitude, triad_quaternion

Q_TRUE = from_axis_angle([0.3, -0.6, 0.2], np.radians(50))
A_TRUE = to_dcm(Q_TRUE)
S_R = normalize([1.0, 0.2, 0.1])          # Sun in ECI
B_R = normalize([-0.3, 0.9, 0.25])        # field in ECI


def _measure(q, ref, sigma_deg, rng):
    v = apply(q, ref)
    if sigma_deg > 0:
        v = v + rng.normal(0.0, np.radians(sigma_deg), size=3)
    return normalize(v)


def _att_err_deg(A):
    R = A @ A_TRUE.T
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


# ---------- triad_frame basics ----------
def test_triad_frame_is_orthonormal():
    M = triad_frame([1.0, 0.2, 0.1], [-0.3, 0.9, 0.25])
    assert np.allclose(M.T @ M, np.eye(3), atol=1e-12)      # orthonormal
    assert np.isclose(np.linalg.det(M), 1.0)               # proper (right-handed)


def test_triad_frame_first_column_is_primary():
    p = normalize([0.4, -0.5, 0.7])
    M = triad_frame(p, [1.0, 0.0, 0.0])
    assert np.allclose(M[:, 0], p)                          # u = normalized primary


def test_triad_frame_rejects_parallel_vectors():
    with pytest.raises(ValueError):
        triad_frame([1.0, 0, 0], [2.0, 0, 0])              # parallel -> undefined


# ---------- THE GATE: noiseless recovery ----------
def test_TRIAD_recovers_attitude_noiseless():
    """Exact truth data must return the true attitude to machine precision."""
    S_B = apply(Q_TRUE, S_R)
    B_B = apply(Q_TRUE, B_R)
    A = triad_attitude(S_B, B_B, S_R, B_R)
    assert np.allclose(A, A_TRUE, atol=1e-12)


def test_TRIAD_quaternion_recovers_truth_noiseless():
    S_B = apply(Q_TRUE, S_R)
    B_B = apply(Q_TRUE, B_R)
    q = triad_quaternion(S_B, B_B, S_R, B_R)
    assert np.allclose(q, Q_TRUE, atol=1e-10) or np.allclose(q, -Q_TRUE, atol=1e-10)


# ---------- noisy behaviour ----------
def test_TRIAD_noisy_within_a_few_degrees():
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(2000):
        S_B = _measure(Q_TRUE, S_R, 0.5, rng)      # accurate sun sensor
        B_B = _measure(Q_TRUE, B_R, 3.0, rng)      # coarse magnetometer
        errs.append(_att_err_deg(triad_attitude(S_B, B_B, S_R, B_R)))
    assert np.mean(errs) < 5.0                      # a few degrees, not exact


def test_primary_choice_matters_under_noise():
    """Putting the ACCURATE sensor primary should beat putting the coarse one primary."""
    rng = np.random.default_rng(1)
    err_sun_primary, err_mag_primary = [], []
    for _ in range(2000):
        S_B = _measure(Q_TRUE, S_R, 0.5, rng)
        B_B = _measure(Q_TRUE, B_R, 3.0, rng)
        err_sun_primary.append(_att_err_deg(triad_attitude(S_B, B_B, S_R, B_R)))
        err_mag_primary.append(_att_err_deg(triad_attitude(B_B, S_B, B_R, S_R)))
    assert np.mean(err_sun_primary) < np.mean(err_mag_primary)


# ---------- the discard property (DET1 criterion) ----------
def test_secondary_parallel_component_is_discarded():
    """Adding any multiple of the primary to the secondary must not change A.

    This is the 'TRIAD discards the secondary's component along the primary'
    property, made into a test.
    """
    S_B = apply(Q_TRUE, S_R)
    B_B = apply(Q_TRUE, B_R)
    A1 = triad_attitude(S_B, B_B, S_R, B_R)
    # contaminate the secondary with a big parallel component, in BOTH frames
    B_B2 = B_B + 5.0 * normalize(S_B)
    B_R2 = B_R + 5.0 * normalize(S_R)
    A2 = triad_attitude(S_B, B_B2, S_R, B_R2)
    assert np.allclose(A1, A2, atol=1e-12)
