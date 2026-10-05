"""Node C1 -- B-dot magnetic detumbling on a real orbit.

THE GATE is test_BDOT_DETUMBLES_AND_SETTLES_AT_THE_FLOOR: a 6U tumbling at
~10 deg/s in a 500 km SSO, magnetometer only, must get below 0.5 deg/s within
one orbit and then settle NEAR TWICE THE ORBITAL RATE -- not zero. That floor
is the R_dot term of M&C Eq. 7.52 (M&C p. 309; Lovera Prop. 3), and B-cross,
fed by the true rate, has none.

The 20-deg cases are EOS Orbit's planned inclination band.
"""
import numpy as np
import pytest

from adcs_sim.control import (estimate_bdot, bdot_law, bcross_law, bdot_gain,
                              simulate_detumble)
from adcs_sim.actuators import NT_TO_T
from adcs_sim.environment import magnetic_field_eci, OMEGA_EARTH
from adcs_sim.orbit import mean_motion, elements_to_rv
from adcs_sim.quaternion import from_axis_angle, apply, q_dot, normalize
from adcs_sim.kinematics import rk4_step

J = np.diag([0.10, 0.12, 0.05])           # 6U-ish, kg m^2
A_KM = 6371.0 + 500.0
N = mean_motion(A_KM)
T_ORB = 2*np.pi / N
Q0 = from_axis_angle([0.3, 0.5, -0.2], 0.4)
W0 = np.radians([5.0, -6.0, 5.5])         # ~9.6 deg/s tumble
TWO_N_DEG = np.degrees(2*N)               # field-rotation rate, ~0.127 deg/s


def _orbit(inc_deg):
    return dict(a=A_KM, e=0.0, i=np.radians(inc_deg), Omega=0.3, omega=0.0, M0=0.0)


def _gain(inc_deg):
    return bdot_gain(T_ORB, np.radians(min(inc_deg, 180 - inc_deg)), J.diagonal().min())


def _rate_deg(run):
    return np.degrees(np.linalg.norm(run["w"], axis=1))


def _time_below(run, thr_deg=0.5):
    w = _rate_deg(run)
    return run["t"][np.argmax(w < thr_deg)] if (w < thr_deg).any() else np.inf


_CACHE = {}
def _run(law, inc, hours=4.0, t0=0.0, w0=W0, bias_dps=0.0):
    key = (law, inc, hours, t0, tuple(w0), bias_dps)
    if key not in _CACHE:
        bias = np.radians(bias_dps) * np.array([1.0, -1.0, 1.0]) / np.sqrt(3)
        _CACHE[key] = simulate_detumble(law, Q0, w0, J, _orbit(inc), hours*3600, _gain(inc),
                                        t0=t0, gyro_bias=bias)
    return _CACHE[key]


def _floor(run):
    return _rate_deg(run)[-int(T_ORB):].mean()          # mean rate over the last orbit


# ---------- your two functions ----------
def test_estimate_bdot_is_a_finite_difference():
    assert np.allclose(estimate_bdot([100.0, -50.0, 20.0], [90.0, -40.0, 20.0], 2.0), [5.0, -5.0, 0.0])


def test_bdot_law_opposes_the_change_and_works_in_tesla():
    B = np.array([20000.0, 0.0, 0.0])          # nT
    Bdot = np.array([0.0, 300.0, 0.0])         # nT/s
    m = bdot_law(Bdot, B, k=2e-4)
    # -k * 300e-9 / (20000e-9)^2 = -2e-4 * 750 = -0.15 A m^2, along -y
    assert np.allclose(m, [0.0, -0.15, 0.0])


def test_bdot_equals_bcross_when_the_field_is_fixed_in_space():
    """With R_dot = 0 (M&C 7.52), B_dot = -w x B exactly, so B-dot IS B-cross."""
    w = np.radians([3.0, -5.0, 8.0])
    R = np.array([15000.0, -20000.0, 25000.0])
    q, Ts = from_axis_angle([0.3, 0.5, -0.2], 0.4), 0.01
    B0 = apply(q, R)
    B1 = apply(normalize(rk4_step(lambda x, t: q_dot(x, w), q, 0.0, Ts)), R)
    m_dot = bdot_law(estimate_bdot(B1, B0, Ts), B0, k=2e-4)
    m_cross = bcross_law(w, B0, k=2e-4)
    assert np.linalg.norm(m_dot - m_cross) < 0.01 * np.linalg.norm(m_cross)   # O(Ts) finite-difference error


# ---------- given pieces ----------
def test_gain_from_mc_eq_7_55():
    # (4 pi / 5668 s)(1 + sin 82.6 deg)(0.05) = 2.21e-4 kg m^2/s
    assert np.isclose(_gain(97.4), 2.21e-4, rtol=0.01)


def test_field_model_rotates_with_the_earth():
    r = [A_KM, 0.0, 0.0]
    T_sid = 2*np.pi / OMEGA_EARTH
    B_now, B_day = magnetic_field_eci(r, 0.0), magnetic_field_eci(r, T_sid)
    B_half = magnetic_field_eci(r, T_sid / 2)
    assert np.allclose(B_now, B_day, rtol=1e-6)          # one sidereal day: back where it was
    assert not np.allclose(B_now, B_half, rtol=1e-2)     # half a day: different


# ---------- THE GATE ----------
@pytest.mark.slow
def test_BDOT_DETUMBLES_AND_SETTLES_AT_THE_FLOOR():
    run = _run("bdot", 97.4)
    w = _rate_deg(run)
    assert w[0] > 9.0
    assert _time_below(run) < T_ORB                       # < 0.5 deg/s within one orbit
    floor = w[-int(T_ORB):].mean()                        # mean over the last orbit
    assert 0.5*TWO_N_DEG < floor < 2.0*TWO_N_DEG, f"floor {floor:.3f} deg/s, 2n = {TWO_N_DEG:.3f}"


# ---------- B-dot vs B-cross ----------
@pytest.mark.slow
def test_bdot_and_bcross_detumble_equally_fast():
    """Same orbit, same gain: the laws differ in WHERE they end, not how fast
    they get below 0.5 deg/s (here 0.41 vs 0.39 orbits)."""
    t_dot, t_cross = _time_below(_run("bdot", 97.4)), _time_below(_run("bcross", 97.4))
    assert 0.8 < t_dot / t_cross < 1.25


@pytest.mark.slow
def test_bcross_with_a_perfect_gyro_has_no_floor():
    """True rate instead of B_dot: no R_dot term, so no 2n floor."""
    assert _floor(_run("bcross", 97.4)) < 0.02 < 0.2 * _floor(_run("bdot", 97.4))


@pytest.mark.slow
def test_bcross_floor_is_the_gyro_bias():
    """B-cross nulls the MEASURED rate, so the true rate settles at the bias.
    A fresh, uncalibrated MEMS gyro (0.5 deg/s) makes B-cross WORSE than
    B-dot -- why flight software detumbles with B-dot first and switches to
    B-cross only after the bias is estimated (node E3)."""
    f_small = _floor(_run("bcross", 97.4, bias_dps=0.05))
    assert 0.5*0.05 < f_small < 1.5*0.05
    assert _floor(_run("bcross", 97.4, bias_dps=0.5)) > _floor(_run("bdot", 97.4))


@pytest.mark.slow
def test_low_inclination_detumbles_but_slower():
    """EOS Orbit's band: the field direction varies less, so the worst axis
    has less authority (Lovera Assumption 1). Still detumbles, several times
    slower -- and how much slower depends on the deployment epoch (t0)."""
    t_sso = _time_below(_run("bdot", 97.4))
    t_20 = _time_below(_run("bdot", 20.0, t0=9*3600))
    assert np.isfinite(t_20)
    assert t_20 > 2.0 * t_sso


@pytest.mark.slow
def test_starting_with_momentum_along_the_field_still_detumbles():
    """The 'unlucky' start: h exactly along B at t = 0. In a constant field this
    would be permanent (A2 gate); in orbit the field moves and it recovers."""
    r0, _ = elements_to_rv(A_KM, 0.0, np.radians(97.4), 0.3, 0.0, 0.0)
    b = magnetic_field_eci(r0, 0.0)
    b /= np.linalg.norm(b)
    w_aligned = np.linalg.solve(J, apply(Q0, b) * np.linalg.norm(J @ W0))
    run = _run("bdot", 97.4, hours=2.0, w0=w_aligned)
    h0 = run["h_eci"][0]
    assert np.degrees(np.arccos(np.dot(h0 / np.linalg.norm(h0), b))) < 0.1
    assert _time_below(run) < 2*3600
