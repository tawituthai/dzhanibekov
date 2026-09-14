"""F4 mastery tests. When all pass, F4 closes.

All are red until you implement omega_dot (and principal_axes), because the
whole plant runs through Euler's equations.
"""
import numpy as np

from adcs_sim.quaternion import normalize
from adcs_sim.dynamics import (
    omega_dot, state_dot, propagate_state, principal_axes,
    angular_momentum_body, kinetic_energy,
)

J_TRI = np.diag([1.0, 2.0, 3.0])          # three distinct principal moments


# ---------- Euler's equations, basic behaviour ----------
def test_no_torque_no_spin_stays_still():
    assert np.allclose(omega_dot([0, 0, 0], J_TRI, [0, 0, 0]), np.zeros(3))


def test_torque_about_principal_axis_is_pure_acceleration():
    """Spin-free body, torque along b1: omega_dot = tau / I11, nothing else."""
    wd = omega_dot([0, 0, 0], J_TRI, [3.0, 0, 0])
    assert np.allclose(wd, [3.0 / 1.0, 0, 0])


def test_spin_about_principal_axis_is_torque_free_equilibrium():
    """omega aligned with a principal axis => omega x J omega = 0 => no change."""
    for axis in range(3):
        w = np.zeros(3)
        w[axis] = 4.0
        assert np.allclose(omega_dot(w, J_TRI, [0, 0, 0]), np.zeros(3), atol=1e-14)


def test_cross_coupling_sign():
    """Off-axis spin must produce the S&J (4.33) coupling, with the right sign.

        I11 w1_dot = -(I33 - I22) w2 w3
    """
    w = np.array([0.0, 2.0, 5.0])
    wd = omega_dot(w, J_TRI, [0, 0, 0])
    expected_w1dot = -(3.0 - 2.0) * w[1] * w[2] / 1.0
    assert np.isclose(wd[0], expected_w1dot)


# ---------- conservation, torque-free ----------
def test_angular_momentum_magnitude_conserved():
    _, _, wh = propagate_state([1, 0, 0, 0], [0.6, -0.4, 1.5], J_TRI, 40.0, 1e-3)
    H = np.linalg.norm([angular_momentum_body(w, J_TRI) for w in wh], axis=1)
    assert np.max(np.abs(H - H[0])) < 1e-9


def test_kinetic_energy_conserved():
    _, _, wh = propagate_state([1, 0, 0, 0], [0.6, -0.4, 1.5], J_TRI, 40.0, 1e-3)
    T = np.array([kinetic_energy(w, J_TRI) for w in wh])
    assert np.max(np.abs(T - T[0])) < 1e-9


def test_quaternion_stays_unit_through_coupled_propagation():
    _, qh, _ = propagate_state([1, 0, 0, 0], [0.6, -0.4, 1.5], J_TRI, 40.0, 1e-3)
    assert np.max(np.abs(np.linalg.norm(qh, axis=1) - 1.0)) < 1e-12


# ---------- validation against the closed-form solution ----------
def test_axially_symmetric_matches_SJ_4_42():
    """S&J Eqs. (4.34)-(4.42): closed form for an axisymmetric torque-free body.

        w1(t) = w10 cos(wp t) - w20 sin(wp t)
        w2(t) = w20 cos(wp t) + w10 sin(wp t)
        w3(t) = w30            with  wp = (I33/IT - 1) w30
    """
    IT, I3 = 2.0, 5.0
    J = np.diag([IT, IT, I3])
    w0 = np.array([0.6, -0.4, 1.5])
    T = 7.0
    _, _, wh = propagate_state([1, 0, 0, 0], w0, J, T, 1e-4)
    wp = (I3 / IT - 1.0) * w0[2]
    exact = np.array([w0[0]*np.cos(wp*T) - w0[1]*np.sin(wp*T),
                      w0[1]*np.cos(wp*T) + w0[0]*np.sin(wp*T),
                      w0[2]])
    assert np.allclose(wh[-1], exact, atol=1e-10)


# ---------- THE GATE: intermediate-axis (tennis racket) instability ----------
def test_TENNIS_RACKET_intermediate_axis_flips():
    """THE F4 GATE.

    Spin about the INTERMEDIATE inertia axis with a tiny perturbation. The
    Dzhanibekov effect must emerge from the equations alone: omega_2 reverses
    sign entirely (+5 -> -5). A wrong cross-product order or sign will NOT
    produce this.
    """
    _, _, wh = propagate_state([1, 0, 0, 0], [1e-3, 5.0, 1e-3], J_TRI, 60.0, 1e-3)
    w2 = wh[:, 1]
    assert w2.min() < -4.9 and w2.max() > 4.9      # full reversal, both ways
    assert np.sum(np.diff(np.sign(w2)) != 0) >= 2   # flips repeatedly


def test_stable_spin_about_max_and_min_inertia_axes():
    """Same perturbation about the largest / smallest inertia axis must NOT flip."""
    for axis, rate in [(0, 5.0), (2, 5.0)]:
        w0 = np.full(3, 1e-3)
        w0[axis] = rate
        _, _, wh = propagate_state([1, 0, 0, 0], w0, J_TRI, 60.0, 1e-3)
        swing = np.max(wh.max(axis=0) - wh.min(axis=0))
        assert swing < 0.05, f"axis {axis} should be stable, saw swing {swing}"


# ---------- principal axes ----------
def test_principal_axes_diagonalizes():
    J = np.array([[3.0, 1.0, 1.0],
                  [1.0, 5.0, 2.0],
                  [1.0, 2.0, 4.0]])          # S&J Example 4.2
    I_p, C = principal_axes(J)
    assert np.allclose(C @ J @ C.T, np.diag(I_p), atol=1e-12)
    assert np.allclose(np.sort(I_p), I_p)     # ascending


def test_principal_axes_is_proper_rotation():
    """Rows unit length, mutually orthogonal, right-handed (det = +1)."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        A = rng.normal(size=(3, 3))
        J = A @ A.T + 3*np.eye(3)             # symmetric positive definite
        _, C = principal_axes(J)
        assert np.allclose(C @ C.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(C), 1.0)


def test_dynamics_invariant_under_principal_rotation():
    """Physics must not care which body frame you wrote J in.

    Propagating in the original frame and in the principal frame must give the
    same angular-momentum magnitude history.
    """
    J = np.array([[3.0, 1.0, 1.0],
                  [1.0, 5.0, 2.0],
                  [1.0, 2.0, 4.0]])
    I_p, C = principal_axes(J)
    w0 = np.array([0.5, -0.3, 0.9])

    _, _, wh_b = propagate_state([1, 0, 0, 0], w0, J, 20.0, 1e-3)
    _, _, wh_p = propagate_state([1, 0, 0, 0], C @ w0, np.diag(I_p), 20.0, 1e-3)

    H_b = np.linalg.norm([J @ w for w in wh_b], axis=1)
    H_p = np.linalg.norm([np.diag(I_p) @ w for w in wh_p], axis=1)
    assert np.max(np.abs(H_b - H_p)) < 1e-8
