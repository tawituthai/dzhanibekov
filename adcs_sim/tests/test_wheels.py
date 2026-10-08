"""Node A1 -- reaction wheels.

THE GATE is test_WHEELS_CANNOT_CHANGE_TOTAL_MOMENTUM: four wheels in a pyramid,
driven by arbitrary motor torques, with friction on. The body tumbles, the
wheels spin up and down -- and the inertial angular momentum of body + wheels
never moves. It is the mirror image of the A2 gate: magnetorquers change
momentum (perpendicular to B only); wheels cannot change it at all.

Second pillar: test_wheels_absorb_a_disturbance_until_they_saturate -- why
node A5 (momentum dumping) has to exist.

Sizing reproduces Junaid (2015) Sec. 3.3.1.1 and the wheel he flew-tested
(ESL Large CubeWheel, Sec. 3.5).
"""
import numpy as np
import pytest

from adcs_sim.wheels import (rw_omega_dot, limit_wheel_torque, slew_torque, slew_momentum,
                             wheel_momentum_requirement, inertia_rw, wheel_speed,
                             wheel_friction, total_momentum_inertial, kinetic_energy_rw,
                             propagate_rw)
from adcs_sim.dynamics import omega_dot
from adcs_sim.quaternion import from_axis_angle

J_TOTAL = np.diag([0.10, 0.12, 0.05])       # 6U-ish, wheels included, kg m^2
JS = 47.3e-6                                # Large CubeWheel flywheel, kg m^2 (Junaid Sec. 3.5)
GS3 = np.eye(3)                             # one wheel per body axis
J_RW3 = inertia_rw(J_TOTAL, GS3, JS)


def _pyramid(beta_deg=30.0):
    """Four wheels, spin axes tilted beta above the x-y plane, 90 deg apart in azimuth."""
    b = np.radians(beta_deg)
    phis = np.radians([0.0, 90.0, 180.0, 270.0])
    return np.column_stack([[np.cos(b)*np.cos(p), np.cos(b)*np.sin(p), np.sin(b)] for p in phis])


# ---------- your task: the equations of motion ----------
def test_no_wheel_activity_is_plain_euler():
    """h_s = u_s = 0: S&J 4.120 collapses to F4's Euler equation."""
    w = np.radians([3.0, -5.0, 8.0])
    L = np.array([1e-5, -2e-5, 3e-6])
    assert np.allclose(rw_omega_dot(w, np.zeros(3), np.zeros(3), J_RW3, GS3, L),
                       omega_dot(w, J_RW3, L), rtol=1e-12)


def test_spinning_a_wheel_up_turns_the_body_the_other_way():
    """Body at rest, +1 mNm on the x wheel: body accelerates about -x (Yang 10.1: u = -h_w_dot)."""
    wd = rw_omega_dot(np.zeros(3), np.zeros(3), np.array([1e-3, 0, 0]), J_RW3, GS3)
    assert np.allclose(wd, [-1e-3 / J_RW3[0, 0], 0, 0])


def test_gyroscopic_coupling_of_stored_momentum():
    """Wheel momentum along x, body turning about z, no motor torque:
    omega x h = z x x = +y, so the body accelerates about -y."""
    w, h = np.array([0.0, 0.0, 0.1]), np.array([0.01, 0.0, 0.0])
    wd = rw_omega_dot(w, h, np.zeros(3), J_RW3, GS3)
    assert np.allclose(wd, [0.0, -0.1*0.01 / J_RW3[1, 1], 0.0])


def test_external_torque_enters_directly():
    wd = rw_omega_dot(np.zeros(3), np.zeros(3), np.zeros(3), J_RW3, GS3, L=[0, 0, 2e-6])
    assert np.allclose(wd, [0, 0, 2e-6 / J_RW3[2, 2]])


def test_eom_handles_a_redundant_pyramid():
    """N = 4 wheels: Gs is 3x4 and u_s, h_s have 4 entries."""
    Gs = _pyramid()
    J_RW = inertia_rw(J_TOTAL, Gs, JS)
    u = np.array([1e-3, 0, -1e-3, 0])                     # opposite wheels: net torque in x-z only
    wd = rw_omega_dot(np.zeros(3), np.zeros(4), u, J_RW, Gs)
    assert np.allclose(wd, np.linalg.solve(J_RW, -Gs @ u))
    assert abs(wd[1]) < 1e-15


# ---------- your task: what the wheels can deliver ----------
def test_torque_is_clipped_per_wheel():
    u = limit_wheel_torque([5e-3, -1e-3, -9e-3], np.zeros(3), u_max=2.1e-3, h_max=0.04)
    assert np.allclose(u, [2.1e-3, -1e-3, -2.1e-3])


def test_full_wheel_refuses_more_momentum_the_same_way():
    h = np.array([0.04, -0.04, 0.01])                      # x full (+), y full (-), z not
    u = limit_wheel_torque([1e-3, -1e-3, 1e-3], h, u_max=2.1e-3, h_max=0.04)
    assert np.allclose(u, [0.0, 0.0, 1e-3])


def test_full_wheel_can_still_be_emptied():
    h = np.array([0.04, -0.04, 0.0])
    u = limit_wheel_torque([-1e-3, 1e-3, 0.0], h, u_max=2.1e-3, h_max=0.04)
    assert np.allclose(u, [-1e-3, 1e-3, 0.0])


# ---------- your task: sizing, Junaid (2015) Sec. 3.3.1.1 ----------
def test_slew_torque_junaid_eq_3_2():
    # 30 deg in 30 s about the 0.36 kg m^2 axis -> 0.838 mNm
    assert np.isclose(slew_torque(np.radians(30), 0.36, 30.0), 0.838e-3, rtol=1e-3)


def test_slew_momentum_junaid_eq_3_4():
    assert np.isclose(slew_momentum(0.838e-3, 30.0), 12.57e-3, rtol=1e-3)


def test_wheel_storage_junaid_eq_3_5():
    # N_d = 5.78 uNm (Table 2-1), T_orb = 5677.2 s, T_ecl = 2145 s -> 32.98 mNms
    assert np.isclose(wheel_momentum_requirement(5.78e-6, 5677.2, 2145.0, 12.57e-3), 32.98e-3, rtol=1e-3)


def test_large_cubewheel_meets_the_requirement():
    """Junaid Sec. 3.5 picks the ESL Large CubeWheel: 40 mNms, 2.1 mNm."""
    N_req = slew_torque(np.radians(30), 0.36, 30.0) + 5.78e-6           # Eq. 3.3: 0.843 mNm
    h_req = wheel_momentum_requirement(5.78e-6, 5677.2, 2145.0,
                                       slew_momentum(slew_torque(np.radians(30), 0.36, 30.0), 30.0))
    assert np.isclose(N_req, 0.843e-3, rtol=2e-3)
    assert 2.1e-3 > N_req and 40e-3 > h_req


# ---------- given pieces ----------
def test_inertia_rw_removes_only_the_spin_inertia():
    assert np.allclose(J_TOTAL - J_RW3, JS * np.eye(3))


def test_coulomb_friction_jumps_at_zero_crossing():
    """ACS w/o Att. p. 105: the friction torque reverses as the wheel speed passes zero."""
    tau = wheel_friction([-1e-3, 1e-3], coulomb=1e-5, viscous=0.0)
    assert np.isclose(tau[1] - tau[0], 2e-5)


# ---------- THE GATE ----------
def test_WHEELS_CANNOT_CHANGE_TOTAL_MOMENTUM():
    """Pyramid of 4, arbitrary motor torques, Coulomb + viscous friction, no external torque."""
    Gs = _pyramid()
    J_RW = inertia_rw(J_TOTAL, Gs, JS)
    q0 = from_axis_angle([0.3, 0.5, -0.2], 0.4)
    w0 = np.radians([1.0, -2.0, 0.5])
    h0 = np.array([5e-3, -2e-3, 0.0, 1e-3])

    def motor(t, q, w, h):
        return 1.5e-3 * np.array([np.sin(0.05*t), np.cos(0.07*t), np.sin(0.11*t + 1), -np.cos(0.03*t)])

    t, q, w, h, _ = propagate_rw(q0, w0, h0, J_RW, Gs, JS, t_end=200.0, dt=0.05, motor_fn=motor,
                                 u_max=2.1e-3, h_max=0.04, coulomb=2e-6, viscous=1e-8)
    H = np.array([total_momentum_inertial(q[k], w[k], h[k], J_RW, Gs) for k in range(len(t))])
    H_body = np.array([J_RW @ w[k] for k in range(len(t))])
    assert np.ptp(np.linalg.norm(H_body, axis=1)) > 3 * np.linalg.norm(H[0])    # body momentum swings a lot
    assert np.max(np.linalg.norm(H - H[0], axis=1)) < 1e-5 * np.linalg.norm(H[0])     # RK4 truncation, not physics


def test_energy_rate_matches_sj_4_119():
    """dT/dt = omega^T L + sum Omega_i u_si: an independent check of the EoM,
    since a wrong sign or a missing gyroscopic term breaks it."""
    J_RW = J_RW3
    L = np.array([2e-5, -1e-5, 3e-5])
    motor = lambda t, q, w, h: 1e-3 * np.array([np.sin(0.2*t), 0.5, -np.cos(0.1*t)])
    t, q, w, h, u = propagate_rw(from_axis_angle([1, 0, 0], 0.1), np.radians([2, 1, -3]),
                                 [3e-3, -1e-3, 2e-3], J_RW, GS3, JS, 60.0, 0.01, motor_fn=motor,
                                 L_fn=lambda t, q, w: L)
    T = np.array([kinetic_energy_rw(w[k], h[k], J_RW, JS) for k in range(len(t))])
    Tdot_num = np.gradient(T, t)
    Tdot_sj = np.array([w[k] @ L + wheel_speed(h[k], w[k], GS3, JS) @ u[k] for k in range(len(t))])
    sl = slice(5, -5)
    assert np.max(np.abs(Tdot_num[sl] - Tdot_sj[sl])) < 1e-3 * np.max(np.abs(Tdot_sj))


def test_wheels_absorb_a_disturbance_until_they_saturate():
    """A constant 0.1 mNm disturbance about x, cancelled exactly by the x wheel
    (u = L keeps omega_dot = 0). The body stays still while the wheel fills at
    0.1 mNm per second... until it hits h_max = 10 mNms at t = 100 s. After
    that the disturbance goes straight into the body. That is why A5 exists."""
    L = np.array([1e-4, 0, 0])
    t, q, w, h, _ = propagate_rw([1, 0, 0, 0], np.zeros(3), np.zeros(3), J_RW3, GS3, JS, 150.0, 0.05,
                                 motor_fn=lambda t, q, w, h: L.copy(), L_fn=lambda t, q, w: L,
                                 u_max=2.1e-3, h_max=0.01)
    before, after = t < 95.0, t > 105.0
    assert np.max(np.abs(w[before])) < 1e-9                            # held still
    assert np.isclose(h[np.argmin(np.abs(t - 50.0)), 0], 5e-3, rtol=1e-3)
    assert np.isclose(h[-1, 0], 0.01, rtol=1e-3)                        # full
    assert np.isclose(w[-1, 0], 1e-4 * (150.0 - 100.0) / J_RW3[0, 0], rtol=2e-2)   # body takes the rest
    assert np.all(np.abs(w[after, 1:]) < 1e-9)


def test_friction_spins_a_coasting_wheel_down_into_the_body():
    """No motor torque: viscous friction drains the wheel, and its momentum
    reappears in the body (total conserved -- friction is internal).
    Viscous coefficient from Junaid Sec. 3.5.3: Cv = 1.35e-5 mNm/rpm = 1.29e-7 N m s/rad."""
    CV = 1.35e-5 * 1e-3 / (2*np.pi/60)
    h0 = np.array([0.0, 0.0, 0.02])
    t, q, w, h, _ = propagate_rw([1, 0, 0, 0], np.zeros(3), h0, J_RW3, GS3, JS, 300.0, 0.1,
                                 viscous=CV)
    assert h[-1, 2] < 0.6 * h0[2]
    assert np.isclose(J_RW3[2, 2] * w[-1, 2] + h[-1, 2], h0[2], rtol=1e-9)
