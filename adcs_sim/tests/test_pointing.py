"""Node C2 -- quaternion feedback pointing with reaction wheels.

THE GATE is test_C2_GATE_MEETS_SPEC_ON_NONLINEAR_PLANT: a 10 deg step about a
skewed axis, through the A1 plant (pyramid of 4, torque and momentum limits,
Coulomb + viscous friction), must meet the JvV agility spec:
overshoot <= 5 %, 1 % settling time <= 30 s. Your gains come from
gains_from_spec(ZETA, T_SETTLE_DESIGN) -- the two numbers you choose in
pointing.py.

Criterion 1 (design the controller) is the block of tests before the gate.
Criterion 2 (tune to a spec on the nonlinear plant) is the gate.
After the gate: what the linear design does NOT tell you -- saturation, the
long way round, and the PD offset that A5 and integral control exist for.
"""
import numpy as np
import pytest

from adcs_sim.pointing import (attitude_error, gains_from_spec, pd_torque, allocate_pinv,
                               scale_to_limit, error_angle, signed_angle_about, step_metrics,
                               simulate_pointing, ZETA, T_SETTLE_DESIGN)
from adcs_sim.quaternion import from_axis_angle, multiply, to_dcm
from adcs_sim.wheels import inertia_rw, rw_omega_dot

J_TOTAL = np.diag([0.10, 0.12, 0.05])       # same 6U-ish body as A1
JS = 47.3e-6                                # ESL Large CubeWheel flywheel (Junaid Sec. 3.5)
U_MAX, H_MAX = 2.1e-3, 0.04                 # Large CubeWheel torque and momentum
CV = 1.35e-5 * 1e-3 / (2*np.pi/60)          # viscous, Junaid Sec. 3.5.3, N m s/rad


def _pyramid(beta_deg=30.0):
    b = np.radians(beta_deg)
    phis = np.radians([0.0, 90.0, 180.0, 270.0])
    return np.column_stack([[np.cos(b)*np.cos(p), np.cos(b)*np.sin(p), np.sin(b)] for p in phis])


GS = _pyramid()
J_RW = inertia_rw(J_TOTAL, GS, JS)
NULL = np.array([1.0, -1.0, 1.0, -1.0]) / 2.0     # GS @ NULL = 0 (M&C Sec. 4.8.3 null space)
H_BIAS = 1e-2 * NULL                               # wheels spinning, zero net momentum
Q_REF = from_axis_angle([0.2, 0.1, 1.0], 0.5)
AXIS = np.array([1.0, -2.0, 0.7]) / np.linalg.norm([1.0, -2.0, 0.7])


def _start(deg, axis=AXIS):
    """Body attitude deg away from Q_REF about a body-fixed axis."""
    return multiply(Q_REF, from_axis_angle(axis, np.radians(deg)))


# ---------- your task: the error quaternion ----------
def test_error_is_identity_on_target():
    assert np.allclose(attitude_error(Q_REF, Q_REF), [1, 0, 0, 0])


def test_error_matches_the_dcm_definition():
    """[BR] = [BN][RN]^T, for arbitrary attitudes."""
    q_BN = from_axis_angle([-1.0, 0.5, 2.0], 1.1)
    q_RN = from_axis_angle([1.0, 2.0, 3.0], 0.7)
    assert np.allclose(to_dcm(attitude_error(q_BN, q_RN)), to_dcm(q_BN) @ to_dcm(q_RN).T)


def test_error_angle_and_axis_are_the_offset_you_applied():
    q_e = attitude_error(_start(10.0), Q_REF)
    assert np.isclose(error_angle(q_e), np.radians(10.0))
    assert np.allclose(q_e[1:] / np.linalg.norm(q_e[1:]), AXIS)


def test_error_takes_the_short_way():
    """S&J p. 236: q and -q are the same attitude; keep beta0 >= 0."""
    q_e = attitude_error(_start(350.0, [1, 0, 0]), Q_REF)
    assert q_e[0] >= 0
    assert np.isclose(error_angle(q_e), np.radians(10.0))
    assert q_e[1] < 0                                  # -10 deg about x, not +350


def test_error_does_not_care_about_quaternion_sign():
    q = _start(40.0)
    assert np.allclose(attitude_error(q, Q_REF), attitude_error(-q, Q_REF))
    assert np.allclose(attitude_error(q, Q_REF), attitude_error(q, -Q_REF))


# ---------- your task: gains (JvV Eqs. 6.3.8-6.3.9) ----------
def test_gains_jvv_design_point():
    """zeta = 0.707, 1 % settling 30 s (JvV Sec. 6.3.2.2) with J = I."""
    K, P = gains_from_spec(np.eye(3), 0.707, 30.0)
    wn = np.log(100.0) / (0.707 * 30.0)                # 0.217 rad/s
    assert np.allclose(K, 2*wn**2 * np.eye(3))         # Kp = 0.0943 1/s^2
    assert np.allclose(P, 2*0.707*wn * np.eye(3))      # Kd = 0.307 1/s


def test_gains_scale_with_inertia():
    K, P = gains_from_spec(J_RW, 0.8, 20.0)
    K1, P1 = gains_from_spec(np.eye(3), 0.8, 20.0)
    assert np.allclose(K, K1[0, 0] * J_RW) and np.allclose(P, P1[0, 0] * J_RW)


# ---------- your task: the control law (S&J Eq. 7.166) ----------
def test_closed_loop_is_exactly_linear_in_omega_dot():
    """With the law in the A1 plant:  [I_RW] omega_dot = -K eps - [P] omega,
    whatever the body rate and stored wheel momentum (gyroscopic term cancelled)."""
    K, P = gains_from_spec(J_RW, 0.7, 25.0)
    rng = np.random.default_rng(3)
    for _ in range(5):
        q = from_axis_angle(rng.normal(size=3), rng.uniform(0, 3))
        w = rng.normal(scale=0.05, size=3)
        h = rng.normal(scale=0.01, size=4)
        L_r = pd_torque(q, w, h, Q_REF, K, P, J_RW, GS)
        wd = rw_omega_dot(w, h, allocate_pinv(L_r, GS), J_RW, GS)
        eps = attitude_error(q, Q_REF)[1:]
        assert np.allclose(J_RW @ wd, -K @ eps - P @ w, atol=1e-12)


def test_known_torque_is_fed_forward():
    K, P = gains_from_spec(J_RW, 0.7, 25.0)
    L = np.array([1e-5, -2e-5, 3e-5])
    a = pd_torque(_start(5.0), [0.01, 0, 0], np.zeros(4), Q_REF, K, P, J_RW, GS, L_known=L)
    b = pd_torque(_start(5.0), [0.01, 0, 0], np.zeros(4), Q_REF, K, P, J_RW, GS)
    assert np.allclose(a - b, L)


# ---------- your task: allocation (M&C Sec. 4.8.3) ----------
def test_allocation_delivers_the_torque():
    L_r = np.array([1e-3, -0.5e-3, 0.2e-3])
    assert np.allclose(GS @ allocate_pinv(L_r, GS), L_r)


def test_allocation_is_minimum_norm():
    """Min-norm <=> no component along the null space."""
    u = allocate_pinv(np.array([1e-3, -0.5e-3, 0.2e-3]), GS)
    assert abs(u @ NULL) < 1e-15
    for c in (1e-4, -1e-4):
        assert np.linalg.norm(u) < np.linalg.norm(u + c*NULL)


def test_allocation_three_orthogonal_wheels_is_identity():
    """S&J Eq. 7.168: [Gs] = I  ->  u_s = L_r."""
    L_r = np.array([1e-3, -2e-3, 3e-4])
    assert np.allclose(allocate_pinv(L_r, np.eye(3)), L_r)


# ---------- your task: direction-preserving limit ----------
def test_scaling_keeps_direction():
    u = np.array([4e-3, -1e-3, 2e-3, 0.5e-3])
    s = scale_to_limit(u, U_MAX)
    assert np.isclose(np.max(np.abs(s)), U_MAX)
    assert np.allclose(s / np.linalg.norm(s), u / np.linalg.norm(u))
    assert np.allclose(u, [4e-3, -1e-3, 2e-3, 0.5e-3])        # input untouched


def test_scaling_leaves_feasible_commands_alone():
    u = np.array([1e-3, -2e-3, 0.0, 2.1e-3])
    assert np.allclose(scale_to_limit(u, U_MAX), u)


# ---------- THE GATE ----------
def test_C2_GATE_MEETS_SPEC_ON_NONLINEAR_PLANT():
    """10 deg step about a skewed axis, A1 plant with limits and friction.
    Spec (JvV Sec. 3.2 / 6.3.2.2): overshoot <= 5 %, 1 % settling <= 30 s."""
    K, P = gains_from_spec(J_RW, ZETA, T_SETTLE_DESIGN)
    t, q, w, h, u, q_e = simulate_pointing(_start(10.0), np.zeros(3), H_BIAS, Q_REF, J_TOTAL, GS, JS,
                                           K, P, t_end=60.0, dt=0.05, u_max=U_MAX, h_max=H_MAX,
                                           coulomb=2e-6, viscous=CV)
    phi = np.array([signed_angle_about(e, AXIS) for e in q_e])
    overshoot, t_settle = step_metrics(t, phi)
    print(f"\n  zeta={ZETA}, design t_s={T_SETTLE_DESIGN} s  ->  overshoot {100*overshoot:.2f} %, "
          f"1 % settling {t_settle:.2f} s, peak |u| {1e3*np.abs(u).max():.2f} mNm")
    assert overshoot <= 0.05
    assert t_settle <= 30.0
    assert np.abs(u).max() < U_MAX                 # linear design valid: never saturated
    assert error_angle(q_e[-1]) < np.radians(0.01)


# ---------- given: what the linear design does not tell you ----------
def test_scaling_keeps_a_saturated_slew_on_the_eigenaxis():
    """90 deg: the PD law asks for ~4x the wheel torque. Per-wheel clipping
    (the plant alone) bends the path off the eigenaxis; scaling does not."""
    K, P = gains_from_spec(J_RW, 0.707, 28.0)

    def off_axis_deg(scale):
        t, q, w, h, u, q_e = simulate_pointing(_start(90.0), np.zeros(3), H_BIAS, Q_REF, J_TOTAL, GS,
                                               JS, K, P, 60.0, 0.05, u_max=U_MAX, h_max=H_MAX,
                                               scale=scale)
        tot = np.array([error_angle(e) for e in q_e])
        along = np.array([signed_angle_about(e, AXIS) for e in q_e])
        return np.degrees(np.max(np.sqrt(np.maximum(tot**2 - along**2, 0.0))))

    assert off_axis_deg(True) < 0.01
    assert off_axis_deg(False) > 1.0


def test_short_way_round_in_closed_loop():
    """350 deg about x is 10 deg the other way. Without the beta0 >= 0 switch
    the law 'unwinds' through 350 deg (S&J p. 236)."""
    K, P = gains_from_spec(J_RW, 0.707, 28.0)
    t, q, w, h, u, q_e = simulate_pointing(_start(350.0, [1, 0, 0]), np.zeros(3), H_BIAS, Q_REF,
                                           J_TOTAL, GS, JS, K, P, 60.0, 0.05, u_max=U_MAX, h_max=H_MAX)
    assert np.max([error_angle(e) for e in q_e]) <= np.radians(10.0) + 1e-9
    assert error_angle(q_e[-1]) < np.radians(0.01)


def test_pd_leaves_an_offset_and_the_wheels_keep_filling():
    """Constant unmodelled torque L. S&J Eq. 7.96 (MRP form; p. 238): a PD law
    settles where the 'spring' balances the load -- here eps_ss = K^-1 L.
    Meanwhile the wheels absorb L forever:  [Gs] dh = L t  -> A5."""
    K, P = gains_from_spec(J_RW, 0.707, 28.0)
    L = np.array([5e-6, -3e-6, 4e-6])
    t, q, w, h, u, q_e = simulate_pointing(Q_REF, np.zeros(3), H_BIAS, Q_REF, J_TOTAL, GS, JS, K, P,
                                           400.0, 0.1, u_max=U_MAX, h_max=H_MAX,
                                           L_fn=lambda t, q, w: L)
    assert np.allclose(q_e[-1][1:], np.linalg.solve(K, L), rtol=1e-3)
    assert np.allclose(GS @ (h[-1] - h[0]), L * t[-1], rtol=1e-3)
