"""F2 mastery tests. These ARE the gate criteria — when they all pass, F2 closes.

Green now (functions I implemented):     conventions, multiply, conjugate,
    from_axis_angle, to_dcm, apply, q_dot.
Red until you implement from_dcm:        test_from_dcm_roundtrip,
    test_from_dcm_180deg, test_F1_gate_sun_vector.
"""
import numpy as np
from adcs_sim.quaternion import (
    normalize, conjugate, multiply, from_axis_angle, to_dcm, apply, q_dot, from_dcm,
)

I3 = np.eye(3)


def Rot1(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def LN_matrix(inc):
    """F1 LVLH-from-ECI DCM at the ascending node, inclination inc."""
    c, s = np.cos(inc), np.sin(inc)
    return np.array([[0, c, s], [0, s, -c], [-1.0, 0, 0]])


# ---------- basics ----------
def test_identity_quaternion_is_identity_dcm():
    assert np.allclose(to_dcm([1, 0, 0, 0]), I3)


def test_from_axis_angle_x90_matches_Rot1():
    q = from_axis_angle([1, 0, 0], np.pi / 2)
    assert np.allclose(q, [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])
    assert np.allclose(to_dcm(q), Rot1(np.pi / 2))


def test_conjugate_is_inverse():
    q = normalize([0.3, -0.2, 0.5, 0.1])
    assert np.allclose(multiply(q, conjugate(q)), [1, 0, 0, 0])


def test_apply_matches_dcm():
    q = from_axis_angle([0.2, 0.5, -0.3], 1.1)
    v = np.array([1.0, -2.0, 0.5])
    assert np.allclose(apply(q, v), to_dcm(q) @ v)


# ---------- the properties we spent lessons on ----------
def test_double_cover_A_of_q_equals_A_of_minus_q():
    q = from_axis_angle([0, 1, 0], 0.7)
    assert np.allclose(to_dcm(q), to_dcm(-q))


def test_multiplication_does_not_commute():
    a = from_axis_angle([1, 0, 0], 0.9)
    b = from_axis_angle([0, 1, 0], 0.7)
    assert not np.allclose(multiply(a, b), multiply(b, a))


def test_multiply_composes_dcm_reversed_order():
    a = from_axis_angle([1, 0, 0], 0.9)
    b = from_axis_angle([0, 1, 0], 0.7)
    assert np.allclose(to_dcm(multiply(a, b)), to_dcm(b) @ to_dcm(a))


# ---------- kinematics ----------
def test_qdot_constant_spin_reproduces_axis_angle():
    q = np.array([1.0, 0, 0, 0])
    w = np.array([0.3, 0.0, 0.0])   # spin about body-x at 0.3 rad/s
    dt, T = 1e-4, 2.0
    for _ in range(int(T / dt)):
        q = normalize(q + q_dot(q, w) * dt)
    expected = from_axis_angle([1, 0, 0], 0.3 * T)
    assert np.allclose(q, expected, atol=1e-3) or np.allclose(q, -expected, atol=1e-3)


# ---------- from_dcm (Stanley) + the F1 GATE ----------
def test_from_dcm_roundtrip():
    q = from_axis_angle([0.3, -0.7, 0.2], 2.3)
    C = to_dcm(q)
    assert np.allclose(to_dcm(from_dcm(C)), C)   # same attitude (q or -q ok)


def test_from_dcm_survives_180deg():
    C = Rot1(np.pi)                              # q0 -> 0; naive (3.93) would divide by ~0
    assert np.allclose(to_dcm(from_dcm(C)), C)


def test_F1_gate_sun_vector_via_quaternion():
    """Reproduce F1's rolled-spacecraft result s_B = [0,-1,0] through quaternions."""
    inc = np.radians(30)
    LN = LN_matrix(inc)
    q_LN = from_dcm(LN)
    q_BL = from_axis_angle([1, 0, 0], np.pi / 2)     # 90 deg roll about velocity (x) axis
    q_BN = multiply(q_LN, q_BL)                      # reversed order (see convention note)
    BL = Rot1(np.pi / 2)
    assert np.allclose(to_dcm(q_BN), BL @ LN)        # quaternion path == matrix path [BN]=[BL][LN]
    s_N = np.array([1.0, 0, 0])
    s_B = apply(q_BN, s_N)
    assert np.allclose(s_B, [0, -1, 0], atol=1e-9)
