r"""Attitude control laws  (node C1: magnetic detumbling).

Primary source: Markley & Crassidis, Sec. 7.5.1 "Detumbling" (pp. 308-310).
Analysis:       Lovera (2015), "Magnetic satellite detumbling: the b-dot
                algorithm revisited", Sec. III-VI.
Variants:       Willis et al. (2024), "Building a Better B-Dot", Sec. 3.

THE CHAIN (all frames body unless noted):

  1. B-CROSS (M&C Eq. 7.48) -- needs a gyro:
         m = (k/|B|) w x b ,  b = B/|B|        i.e.   m = k (w x B) / |B|^2
     It is Yang Eq. 10.6 (m = B x u / |B|^2) with the rate damper u = -k w.
     Torque produced (M&C 7.49):  L = -k (I - b b^T) w   -- w with its
     component along B removed. Lyapunov V = 1/2 w^T J w gives
     V_dot = -k w^T (I - b b^T) w <= 0   (M&C 7.50-7.51).

  2. WHAT THE MAGNETOMETER SEES (M&C 7.52, Lovera Eq. 17):
         B = A R   ->   B_dot = -w x B + A R_dot
     (-w x B: the body turning under the field; A R_dot: the field changing
     along the orbit, ~2 turns per orbit.)

  3. B-DOT (M&C 7.53): assume R_dot << B_dot (true early in a detumble, when
     the body spins much faster than the field turns), so  w x B ~= -B_dot :
         m = -k B_dot / |B|^2
     NOTE: M&C 7.53 is printed with |B| to the first power. Substituting
     w x B = -B_dot into 7.48 gives |B|^2, and only |B|^2 is dimensionally
     consistent with the gain of Eq. 7.55 (k in kg m^2/s -> m in A m^2).
     Lovera Eq. 18 uses |B|^2 as well.

  4. THE FLOOR. As w shrinks, A R_dot becomes a comparable part of B_dot and
     B-dot starts "damping" the field's own rotation. The rate settles near
     the field-rotation rate (~2x orbital), not zero: M&C p. 309 ("around
     1e-3 rad/s"), Lovera Prop. 3. B-cross, fed by a gyro, has no such floor.

UNITS: B in tesla inside every law. The field model and magnetometer give nT;
B_dot/|B|^2 computed in nT is 1e9 times too large (same trap as A2).
"""
import numpy as np

from adcs_sim.actuators import magnetorquer_torque, NT_TO_T
from adcs_sim.dynamics import propagate_state
from adcs_sim.environment import magnetic_field_eci, magnetometer
from adcs_sim.orbit import elements_to_rv, mean_motion
from adcs_sim.quaternion import apply


# --------------------------------------------------------------------------
# YOUR TASKS
# --------------------------------------------------------------------------
def estimate_bdot(B_now_nT, B_prev_nT, Ts):
    """Finite-difference rate of change of the MEASURED field, body frame.  <-- YOUR TASK

        B_dot ~= (B_k - B_{k-1}) / Ts            (Turan & Celik Eq. 6)

    B_now_nT, B_prev_nT : (3,) magnetometer readings, nT, Ts seconds apart
    Return (3,) in nT/s. (M&C p. 309 notes this difference is noisy and is
    often filtered in practice; keep it plain here.)
    """
    B_now_nT = np.asarray(B_now_nT, float)
    B_prev_nT = np.asarray(B_prev_nT, float)
    
    return (B_now_nT - B_prev_nT)/Ts
    # raise NotImplementedError("implement the finite-difference B_dot")


def bdot_law(Bdot_nT_s, B_nT, k):
    """B-dot dipole command (M&C Eq. 7.53 with |B|^2, Lovera Eq. 18).  <-- YOUR TASK

        m = -k * B_dot / |B|^2          with B and B_dot in TESLA

    Bdot_nT_s : (3,) nT/s    B_nT : (3,) nT    k : gain, kg m^2/s (M&C 7.55)
    Return (3,) commanded dipole, A m^2 (unsaturated -- magnetorquer_torque
    applies the rod limit).

    Convert BOTH inputs with NT_TO_T first. Sanity check of the units:
    (kg m^2/s)(T/s)/T^2 = N m / T = A m^2.
    """
    
    Bdot_T_s = Bdot_nT_s*NT_TO_T
    B_T = B_nT*NT_TO_T
    
    return -k*(Bdot_T_s)/np.pow(np.linalg.norm(B_T), 2)
    # raise NotImplementedError("implement m = -k B_dot / |B|^2 in tesla")


# --------------------------------------------------------------------------
# given
# --------------------------------------------------------------------------
def bdot_gain(T_orb, xi_m, J_min):
    """Detumbling gain, M&C Eq. 7.55 (from Avanzini & Giulietti 2012):

        k = (4 pi / T_orb) (1 + sin xi_m) J_min

    T_orb : orbital period, s      xi_m : inclination of the orbit relative to
    the geomagnetic equator, rad   J_min : minimum principal inertia, kg m^2.
    4 pi / T_orb is twice the mean motion -- the rate the field turns.
    """
    return 4*np.pi / T_orb * (1 + np.sin(xi_m)) * J_min


def bcross_law(omega, B_nT, k):
    """B-cross dipole command, M&C Eq. 7.48:  m = k (w x B) / |B|^2  (B in T).
    Needs the true body rate -- i.e. a gyro."""
    B = np.asarray(B_nT, float) * NT_TO_T
    return k * np.cross(np.asarray(omega, float), B) / np.dot(B, B)


def simulate_detumble(law, q0, w0, J, orbit, t_end, k, m_max=0.2, Ts=1.0,
                      t0=0.0, sigma_nT=0.0, gyro_bias=None, rng=None):
    """Closed-loop magnetic detumbling on a real orbit.

    law   : "bdot" (magnetometer only) or "bcross" (uses the true rate)
    orbit : dict(a, e, i, Omega, omega, M0) -- km, rad (orbit.elements_to_rv)
    k     : control gain (bdot_gain)        m_max : rod limit, A m^2
    Ts    : control period, s. The dipole is held constant over each period
            (zero-order hold), as flight software does.
    t0    : epoch offset for the Earth-fixed dipole (magnetic_field_eci)
    gyro_bias : (3,) rad/s added to the rate B-cross sees (B-dot ignores it).
            B-cross drives the MEASURED rate to zero, so the true rate
            settles at the bias -- see test_bcross_floor_is_the_gyro_bias.

    Each control step: orbit -> position -> field in ECI -> magnetometer (body)
    -> law -> dipole held for Ts -> propagate_state with torque m x B.
    Returns dict of arrays: t, w (body rate, rad/s), m (dipole), h_eci.
    """
    rng = np.random.default_rng() if rng is None else rng
    J = np.asarray(J, float)
    n = mean_motion(orbit["a"])
    def B_eci_at(t):
        r, _ = elements_to_rv(orbit["a"], orbit["e"], orbit["i"], orbit["Omega"],
                              orbit["omega"], orbit["M0"] + n*t)
        return magnetic_field_eci(r, t0 + t)

    q, w = np.asarray(q0, float), np.asarray(w0, float)
    steps = int(round(t_end / Ts))
    T, W, M, H = np.zeros(steps+1), np.zeros((steps+1, 3)), np.zeros((steps+1, 3)), np.zeros((steps+1, 3))
    B_prev = None
    Be0 = B_eci_at(0.0)
    for kstep in range(steps + 1):
        t = kstep * Ts
        T[kstep], W[kstep] = t, w
        H[kstep] = apply(np.array([q[0], -q[1], -q[2], -q[3]]), J @ w)   # body -> ECI
        if kstep == steps:
            break
        Be1 = B_eci_at(t + Ts)
        B_meas = magnetometer(q, Be0, sigma_nT=sigma_nT, rng=rng)
        if law == "bdot":
            m = np.zeros(3) if B_prev is None else bdot_law(estimate_bdot(B_meas, B_prev, Ts), B_meas, k)
            B_prev = B_meas
        elif law == "bcross":
            w_meas = w if gyro_bias is None else w + np.asarray(gyro_bias, float)
            m = bcross_law(w_meas, B_meas, k)
        else:
            raise ValueError(law)
        M[kstep] = m
        def torque_fn(tt, qq, ww, Be0=Be0, Be1=Be1, m=m):
            Be = Be0 + (Be1 - Be0) * (tt / Ts)          # field changes slowly in ECI
            return magnetorquer_torque(m, apply(qq, Be), m_max)
        _, qh, wh = propagate_state(q, w, J, Ts, Ts, torque_fn)
        q, w, Be0 = qh[-1], wh[-1], Be1
    return dict(t=T, w=W, m=M, h_eci=H)
