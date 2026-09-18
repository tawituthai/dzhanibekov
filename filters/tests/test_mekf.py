"""Stage 1 MEKF tests (attitude-only, perfect gyro).

The gate: the filter converges from a large (20 deg) initial attitude error to
near-truth using noisy sun+magnetometer measurements. Red until small_dq,
predict, and update_vector are implemented.
"""
import numpy as np

from adcs_sim.quaternion import from_axis_angle, normalize, apply, multiply, conjugate, q_dot
from adcs_sim.kinematics import rk4_step
from filters.mekf import MEKF, small_dq, skew


OMEGA = np.array([0.03, -0.02, 0.05])          # rad/s, constant body rate
S_REF = normalize([1.0, 0.2, 0.1])             # sun in inertial
B_REF = normalize([-0.3, 0.9, 0.25])           # field in inertial
DT = 0.1


def _measure(q_true, ref, sigma, rng):
    v = apply(q_true, ref)
    if sigma > 0:
        v = v + rng.normal(0, sigma, 3)
    return v / np.linalg.norm(v)


# ---------- small_dq basics ----------
def test_small_dq_is_unit():
    q = small_dq([0.02, -0.01, 0.03])
    assert np.isclose(np.linalg.norm(q), 1.0)


def test_small_dq_zero_is_identity():
    assert np.allclose(small_dq([0, 0, 0]), [1, 0, 0, 0])


# ---------- predict alone: no measurements => tracks truth exactly (perfect gyro) ----------
def test_predict_matches_truth_no_measurements():
    q_true = from_axis_angle([0.3, -0.6, 0.2], np.radians(40))
    f = MEKF(q0=q_true.copy(), P0=np.eye(3)*1e-4, Q=np.eye(3)*1e-8)
    qt = q_true.copy()
    for k in range(500):
        t = k*DT
        qt = normalize(rk4_step(lambda q, u: q_dot(q, OMEGA), qt, t, DT))
        f.predict(OMEGA, DT)
    assert f.error_deg(qt) < 1e-6      # perfect gyro, same init -> stays locked


# ---------- THE GATE: converge from a large initial error ----------
def _run(sigma_deg, seed, T=200.0, meas_every=5, q0_err_deg=20.0):
    q_true = from_axis_angle([0.3, -0.6, 0.2], np.radians(40))
    q_hat = multiply(from_axis_angle([0, 1, 0], np.radians(q0_err_deg)), q_true.copy())
    sigma = np.radians(sigma_deg)
    f = MEKF(q0=q_hat, P0=np.eye(3)*np.radians(30)**2, Q=np.eye(3)*np.radians(0.02)**2)
    R = sigma**2 * np.eye(3) if sigma > 0 else np.eye(3)*1e-8
    rng = np.random.default_rng(seed)
    qt = q_true.copy()
    errs = []
    for k in range(int(T/DT)):
        t = k*DT
        qt = normalize(rk4_step(lambda q, u: q_dot(q, OMEGA), qt, t, DT))
        f.predict(OMEGA, DT)
        if k % meas_every == 0:
            for ref, sg in ((S_REF, sigma), (B_REF, sigma)):
                f.update_vector(_measure(qt, ref, sg, rng), ref, R)
        errs.append(f.error_deg(qt))
    return np.array(errs)


def test_MEKF_noiseless_converges_to_zero():
    errs = _run(sigma_deg=0.0, seed=0)
    assert errs[0] > 10.0                       # started badly wrong
    assert errs[-1] < 0.01                      # noiseless -> essentially exact


def test_MEKF_converges_under_noise():
    errs = _run(sigma_deg=0.5, seed=0)
    assert errs[0] > 10.0                       # 20 deg initial error
    assert np.mean(errs[-100:]) < 1.5           # pulled in to sub-degree-ish


def test_MEKF_covariance_shrinks():
    q_true = from_axis_angle([0.1, 0.2, -0.3], 0.7)
    f = MEKF(q0=q_true.copy(), P0=np.eye(3)*np.radians(30)**2, Q=np.eye(3)*1e-8)
    trP0 = np.trace(f.P)
    rng = np.random.default_rng(1)
    qt = q_true.copy()
    for k in range(200):
        t = k*DT
        qt = normalize(rk4_step(lambda q, u: q_dot(q, OMEGA), qt, t, DT))
        f.predict(OMEGA, DT)
        if k % 5 == 0:
            f.update_vector(_measure(qt, S_REF, np.radians(0.5), rng), S_REF,
                            np.radians(0.5)**2*np.eye(3))
    assert np.trace(f.P) < trP0
