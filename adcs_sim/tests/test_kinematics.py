"""F3 mastery tests. When all pass, F3 closes.

Green now (rk4_step, propagate):   test_rk4_matches_notes_example, test_rk4_fourth_order,
    test_propagate_constant_omega_GATE, test_norm_stays_unit, test_zero_omega_holds_attitude
Red until you implement B_matrix / q_dot_B / omega_from_qdot.
"""
import numpy as np

from adcs_sim.quaternion import from_axis_angle, multiply, normalize, q_dot
from adcs_sim.kinematics import (
    rk4_step, propagate, B_matrix, q_dot_B, omega_from_qdot,
)


# ---------- the generic integrator, checked against your notes' worked example ----------
def test_rk4_matches_notes_example():
    """y' = y - t^2 + 1, y(0)=0.5, h=0.5 -- the Burden & Faires example in your notes."""
    f = lambda y, t: y - t**2 + 1.0
    y, t, h = 0.5, 0.0, 0.5
    expected = [1.425130208333333, 2.639602661132812,
                4.006818970044454, 5.301605229265987]
    for w_expected in expected:
        y = rk4_step(f, y, t, h)
        t += h
        assert abs(y - w_expected) < 1e-12
    # and within 0.004 of the exact solution y = (t+1)^2 - 0.5 e^t after only 4 steps
    assert abs(y - ((t + 1)**2 - 0.5*np.exp(t))) < 4e-3


def test_rk4_fourth_order():
    """Halving dt must cut the error ~16x (O(dt^4)) on the quaternion kinematics."""
    q0 = normalize([0.3, -0.2, 0.5, 0.1])
    w = np.array([0.4, -0.25, 0.7])
    T = 3.0
    exact = multiply(q0, from_axis_angle(w/np.linalg.norm(w), np.linalg.norm(w)*T))
    errs = []
    for dt in [0.2, 0.1, 0.05]:
        _, qh = propagate(q0, w, T, dt)
        errs.append(np.max(np.abs(qh[-1] - exact)))
    for a, b in zip(errs[:-1], errs[1:]):
        assert 12.0 < a/b < 20.0        # ~16x per halving


# ---------- propagation ----------
def test_propagate_constant_omega_GATE():
    """THE F3 GATE: constant body rate must land on the closed-form answer.

    For constant omega, integrating q_dot = 1/2 q (x) omega for time T gives
    exactly a rotation of |omega|*T about the axis omega_hat:
        q(T) = q0 (x) from_axis_angle(omega_hat, |omega|*T)
    """
    q0 = normalize([0.3, -0.2, 0.5, 0.1])
    w = np.array([0.4, -0.25, 0.7])
    T = 3.0
    _, qh = propagate(q0, w, T, 1e-3)
    exact = multiply(q0, from_axis_angle(w/np.linalg.norm(w), np.linalg.norm(w)*T))
    assert np.allclose(qh[-1], exact, atol=1e-10)


def test_norm_stays_unit():
    """Renormalization must hold |q| = 1 over a long, time-varying propagation."""
    q0 = np.array([1.0, 0, 0, 0])
    omega = lambda t: np.array([0.5*np.sin(0.3*t), 0.2, -0.4*np.cos(t)])
    _, qh = propagate(q0, omega, 200.0, 0.01)
    norms = np.linalg.norm(qh, axis=1)
    assert np.max(np.abs(norms - 1.0)) < 1e-12


def test_zero_omega_holds_attitude():
    q0 = normalize([0.2, 0.4, -0.1, 0.6])
    _, qh = propagate(q0, [0.0, 0.0, 0.0], 10.0, 0.01)
    assert np.allclose(qh[-1], q0, atol=1e-14)


# ---------- [B(q)]: your task ----------
def test_B_matrix_shape_and_entries():
    q = normalize([0.3, -0.2, 0.5, 0.1])
    q0, q1, q2, q3 = q
    B = B_matrix(q)
    assert B.shape == (4, 3)
    assert np.allclose(B, [[-q1, -q2, -q3],
                           [ q0, -q3,  q2],
                           [ q3,  q0, -q1],
                           [-q2,  q1,  q0]])


def test_B_columns_orthonormal():
    """[B]^T [B] = I_3 -- the fact that makes the inverse map trivial."""
    rng = np.random.default_rng(0)
    for _ in range(100):
        q = normalize(rng.normal(size=4))
        assert np.allclose(B_matrix(q).T @ B_matrix(q), np.eye(3))


def test_SJ_identity_3107():
    """S&J Eq. (3.107): [B(q)]^T q = 0  -- q_dot is always perpendicular to q."""
    rng = np.random.default_rng(1)
    for _ in range(100):
        q = normalize(rng.normal(size=4))
        assert np.allclose(B_matrix(q).T @ q, np.zeros(3), atol=1e-14)


def test_q_dot_B_agrees_with_hamilton_form():
    """S&J 3.105 matrix form == quaternion.q_dot Hamilton form."""
    rng = np.random.default_rng(2)
    for _ in range(200):
        q = normalize(rng.normal(size=4))
        w = rng.normal(size=3)
        assert np.allclose(q_dot_B(q, w), q_dot(q, w))


def test_omega_from_qdot_roundtrip():
    """omega -> q_dot -> omega must return the original body rate."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        q = normalize(rng.normal(size=4))
        w = rng.normal(size=3)
        assert np.allclose(omega_from_qdot(q, q_dot(q, w)), w)
