"""F5 orbit tests. When all pass (with F4 already green), Foundations is complete.

Red until you implement solve_kepler, perifocal_to_eci, and lvlh_frame.
"""
import numpy as np

from adcs_sim.orbit import (
    MU_EARTH, mean_motion, solve_kepler, perifocal_to_eci,
    elements_to_rv, propagate_orbit, lvlh_frame,
)
from adcs_sim.environment import magnetic_field_eci, in_eclipse

# a reference LEO (ISS-like)
A, E, I = 7000.0, 0.01, np.radians(51.6)
OM, W, M0 = np.radians(40.0), np.radians(60.0), np.radians(30.0)


# ---------- Kepler ----------
def test_kepler_satisfies_equation():
    for M in np.linspace(0, 2*np.pi, 20):
        for e in (0.0, 0.01, 0.3, 0.7):
            E = solve_kepler(M, e)
            assert abs((E - e*np.sin(E)) - M) < 1e-10       # M = E - e sinE


def test_kepler_circular_is_identity():
    for M in np.linspace(0, 2*np.pi, 10):
        assert np.isclose(solve_kepler(M, 0.0), M)          # e=0 -> E=M


# ---------- perifocal -> ECI DCM ----------
def test_NO_is_proper_rotation():
    C = perifocal_to_eci(OM, I, W)
    assert np.allclose(C @ C.T, np.eye(3), atol=1e-12)      # orthonormal
    assert np.isclose(np.linalg.det(C), 1.0)               # proper


# ---------- THE GATE: vis-viva energy consistency ----------
def test_VISVIVA_energy_matches():
    """r,v from the conversion must satisfy v^2/2 - mu/r = -mu/2a (S&J Problem 8.6c spirit)."""
    r, v = elements_to_rv(A, E, I, OM, W, M0)
    eps_state = v @ v / 2 - MU_EARTH / np.linalg.norm(r)
    eps_orbit = -MU_EARTH / (2 * A)
    assert np.isclose(eps_state, eps_orbit, rtol=1e-9)


def test_radius_within_perigee_apogee():
    r, _ = elements_to_rv(A, E, I, OM, W, M0)
    rp, ra = A*(1 - E), A*(1 + E)
    assert rp - 1e-6 <= np.linalg.norm(r) <= ra + 1e-6


# ---------- propagation ----------
def test_orbit_returns_after_one_period():
    n = mean_motion(A)
    P = 2*np.pi / n
    _, rh, _ = propagate_orbit(A, E, I, OM, W, M0, P, P/1000)
    assert np.linalg.norm(rh[-1] - rh[0]) < 1e-6            # closed orbit


def test_energy_conserved_along_propagation():
    _, rh, vh = propagate_orbit(A, E, I, OM, W, M0, 6000.0, 10.0)
    eps = np.array([v@v/2 - MU_EARTH/np.linalg.norm(r) for r, v in zip(rh, vh)])
    assert np.max(np.abs(eps - eps[0])) < 1e-6             # energy constant over the orbit


# ---------- LVLH frame ----------
def test_lvlh_is_proper_rotation():
    r, v = elements_to_rv(A, E, I, OM, W, M0)
    L = lvlh_frame(r, v)
    assert np.allclose(L @ L.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(L), 1.0)


def test_lvlh_nadir_points_down():
    """The nadir axis (3rd row) must point opposite to the position vector."""
    r, v = elements_to_rv(A, E, I, OM, W, M0)
    L = lvlh_frame(r, v)
    nadir = L[2]                                            # o3 row, in ECI
    assert np.allclose(nadir, -r/np.linalg.norm(r), atol=1e-9)


# ---------- wiring into the environment (the F5 payoff) ----------
def test_field_evaluated_along_orbit():
    """The whole point: r(t) feeds the environment models over a real orbit."""
    _, rh, _ = propagate_orbit(A, E, I, OM, W, M0, 3000.0, 100.0)
    mags = [np.linalg.norm(magnetic_field_eci(r)) for r in rh]
    assert all(15000 < m < 60000 for m in mags)            # physical field along the whole orbit
    assert np.std(mags) > 0                                 # field actually varies as position changes
