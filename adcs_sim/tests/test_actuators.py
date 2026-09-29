"""Node A2 -- magnetorquers.

THE GATE is test_MAGNETORQUERS_CANNOT_CHANGE_MOMENTUM_ALONG_B: close the loop
through propagate_state (the first time torque_fn is not None) and show that,
whatever the rods do, the inertial angular momentum component along a fixed
field never changes. That is underactuation, demonstrated on the full
nonlinear dynamics rather than asserted.

Sizing tests reproduce a real flown-class design: Junaid (2015), Stellenbosch
CubeSat ADCS, Sec. 3.3-3.4 and Table 3-2 (p. 34).
"""
import numpy as np

from adcs_sim.actuators import (saturate_dipole, magnetorquer_torque, required_dipole,
                                rod_amplification, dipole_from_coil, cross_product_law,
                                NT_TO_T)
from adcs_sim.quaternion import from_axis_angle, apply, to_dcm
from adcs_sim.dynamics import propagate_state


# ---------- saturation ----------
def test_saturation_scales_instead_of_clipping():
    m = saturate_dipole([2.0, 1.0, 0.0], 0.2)
    assert np.allclose(m, [0.2, 0.1, 0.0])          # clipping would give [0.2, 0.2, 0]


def test_saturation_leaves_small_commands_alone():
    m_cmd = np.array([0.05, -0.1, 0.15])
    assert np.allclose(saturate_dipole(m_cmd, 0.2), m_cmd)


def test_saturation_per_axis_limits():
    # z rod is a weaker air coil: 0.1 vs 0.2. z is the worst axis (0.3/0.1 = 3x).
    m = saturate_dipole([0.3, 0.3, 0.3], np.array([0.2, 0.2, 0.1]))
    assert np.allclose(m, [0.1, 0.1, 0.1])


# ---------- torque ----------
def test_torque_units_and_direction():
    # x-dipole in a y-field: torque along +z (x cross y = z). 0.2 A m^2 * 30 000 nT = 6e-6 N m
    tau = magnetorquer_torque([0.2, 0, 0], [0, 30000.0, 0], m_max=0.2)
    assert np.allclose(tau, [0, 0, 6e-6])


def test_dipole_parallel_to_field_gives_no_torque():
    B = np.array([12000.0, -25000.0, 8000.0])
    m = 0.1 * B / np.linalg.norm(B)
    assert np.allclose(magnetorquer_torque(m, B, 0.2), 0, atol=1e-15)


def test_torque_is_always_perpendicular_to_B():
    """Underactuation, instantaneous version: 500 random commands, zero torque along B."""
    rng = np.random.default_rng(0)
    for _ in range(500):
        B = rng.normal(0, 30000, 3)
        tau = magnetorquer_torque(rng.normal(0, 0.3, 3), B, 0.2)
        assert abs(np.dot(tau, B / np.linalg.norm(B))) < 1e-18


def test_torque_respects_saturation():
    B = np.array([0, 30000.0, 0])
    assert np.allclose(magnetorquer_torque([5.0, 0, 0], B, 0.2),
                       magnetorquer_torque([0.2, 0, 0], B, 0.2))


# ---------- cross-product law (given) ----------
def test_cross_product_law_loses_the_component_along_B():
    """Ask for any torque; you get it minus its projection on B."""
    B = np.array([10000.0, 20000.0, -15000.0])
    tau_des = np.array([3e-6, -1e-6, 2e-6])
    m = cross_product_law(tau_des, B)
    tau_got = np.cross(m, B * NT_TO_T)
    b_hat = B / np.linalg.norm(B)
    assert np.allclose(tau_got, tau_des - np.dot(tau_des, b_hat) * b_hat)


# ---------- sizing: Junaid (2015) ----------
def test_rod_sizing_junaid_eq_3_7():
    # 15.291 uN m to dump momentum over eclipse, weakest field 25 uT -> 0.612 A m^2
    assert np.isclose(required_dipole(15.291e-6, 25000.0), 0.612, rtol=1e-3)


def test_rod_design_reproduces_junaid_table_3_2():
    """L = 150 mm, D = 6.25 mm, 4000 turns, 72 ohm. Table 3-2: amplification 195,
    1.66 A m^2 at 5 V and 0.83 A m^2 at 2.5 V."""
    mu = rod_amplification(0.150, 0.00625)
    assert np.isclose(mu, 195, rtol=0.01)
    area = np.pi * (0.00625 / 2) ** 2
    assert np.isclose(dipole_from_coil(4000, 5.0 / 72, area, mu), 1.66, rtol=0.01)
    assert np.isclose(dipole_from_coil(4000, 2.5 / 72, area, mu), 0.83, rtol=0.01)


# ---------- THE GATE ----------
def test_MAGNETORQUERS_CANNOT_CHANGE_MOMENTUM_ALONG_B():
    """Closed loop on the full dynamics, with a fixed inertial field. The rods
    try as hard as they can to kill ALL the angular momentum (tau_des = -k H,
    via the cross-product law, saturated at 0.2 A m^2). They remove most of the
    momentum perpendicular to B -- and none of the component along B."""
    J = np.diag([0.10, 0.12, 0.05])                       # 6U-ish, kg m^2
    B_eci = np.array([15000.0, -20000.0, 25000.0])        # nT, held fixed
    b_hat = B_eci / np.linalg.norm(B_eci)

    def torque_fn(t, q, w):
        B_body = apply(q, B_eci)
        m_cmd = cross_product_law(-0.01 * (J @ w), B_body)   # "stop everything"
        return magnetorquer_torque(m_cmd, B_body, m_max=0.2)

    q0 = from_axis_angle([0.3, 0.5, -0.2], 0.4)
    w0 = np.radians([3.0, -5.0, 8.0])
    t, qh, wh = propagate_state(q0, w0, J, t_end=3000.0, dt=0.5, torque_fn=torque_fn)

    H_eci = np.array([to_dcm(q).T @ (J @ w) for q, w in zip(qh, wh)])
    along = H_eci @ b_hat
    across = np.linalg.norm(H_eci - np.outer(along, b_hat), axis=1)

    assert across[-1] < 0.1 * across[0]                   # perpendicular momentum: mostly gone
    assert np.ptp(along) < 1e-6 * np.linalg.norm(H_eci[0])  # along B: untouched

def test_saturation_handles_negative_commands():
    """Rod limits are symmetric: -2 is as far over the limit as +2."""
    assert np.allclose(saturate_dipole([-2.0, -1.0, 0.0], 0.2), [-0.2, -0.1, 0.0])