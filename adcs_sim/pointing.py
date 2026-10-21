r"""Three-axis pointing with reaction wheels  (node C2: quaternion feedback).

A1 built the plant: a body with N wheels, S&J Eq. 4.120 / 7.158,
    [I_RW] omega_dot = -omega x ([I_RW] omega + [Gs] h_s) - [Gs] u_s + L
C2 closes the loop around it. The chain, run once per control step:

    q_BN, omega  ->  attitude error q_BR  ->  required torque L_r  ->  wheel torques u_s
                    (attitude_error)        (pd_torque)           (allocate_pinv, scale_to_limit)

PRIMARY SOURCE: Schaub & Junkins, Ch. 7.
  * Sec. 7.3.1, Eq. 7.84 (p. 236): the Euler-parameter feedback law
        u = -K eps - [P] delta_omega + [I](omega_r_dot - omega x omega_r) + omega x [I] omega - L
    eps = (beta1, beta2, beta3) is the VECTOR part of the error quaternion.
    Same page: the law "will not automatically drive the body back to the
    reference trajectory through the shortest rotational path" unless the
    error quaternion is switched to the set with beta0 >= 0.
  * Sec. 7.6, Eqs. 7.163-7.168 (pp. 259-260): the same law for reaction wheels.
    The wheels must produce the required torque L_r:
        [Gs] u_s = L_r                                              (7.167)
        L_r = K sigma + [P] delta_omega - omega x ([I_RW] omega + [Gs] h_s) + L   (7.166,
              regulation case omega_r = 0; MRP sigma there, eps here)
    For three wheels on the body axes [Gs] = I and u_s = L_r       (7.168).
  * Sec. 7.3.2, Eq. 7.96 (p. 238): a PD law leaves a steady attitude offset
    under a constant unmodelled torque -- the spring under gravity.

GAINS: Janse van Vuuren Sec. 6.3.2, Eqs. 6.3.4-6.3.9 (pp. 81-82) writes the
same law with gains scaled by the inertia,  K = Kp [I],  [P] = Kd [I].
Then the gyroscopic term cancels exactly and the closed loop is
        omega_dot + Kd omega + Kp eps = 0                           (JvV 6.3.5)
For a small rotation Phi about the eigenaxis, eps = sin(Phi/2) lambda ~ (Phi/2) lambda:
        Phi_ddot + Kd Phi_dot + (Kp/2) Phi = 0
      = Phi_ddot + 2 zeta wn Phi_dot + wn^2 Phi                     (JvV 6.3.7)
  ->    Kd = 2 zeta wn ,   Kp = 2 wn^2                              (JvV 6.3.8-6.3.9)
Junaid Eq. 4.66 prints these two SWAPPED (Kp = 2 zeta wn, Kd = 2 wn^2); his own
Laplace form two lines above, (s^2 + Kd s + Kp/2), gives the JvV result.
Check the units: Kp eps must be rad/s^2 (eps is dimensionless) so Kp is 1/s^2 --
that is wn^2, not zeta wn.

ALLOCATION: Markley & Crassidis Sec. 4.8.3 (pp. 152-157). With N > 3 wheels,
[Gs] u_s = L_r has infinitely many solutions. The minimum-norm one is the
pseudoinverse  u_s = [Gs]^T ([Gs][Gs]^T)^-1 L_r. Anything in the null space of
[Gs] can be added without changing the body torque.

REAL USE CASES: JvV Sec. 6.3.2.2 (zeta = 0.707, 1% settling time 30 s, from the
30 deg in 30 s agility spec of his Sec. 3.2); Junaid Sec. 4.2.2.1 (the same
controller flown on a CubeSat with the ESL CubeWheels).

QUATERNION CONVENTION (repo F2): to_dcm(multiply(p, q)) = to_dcm(q) @ to_dcm(p).
Junaid Eq. 4.64 is in the Markley-Crassidis scalar-LAST convention; the
Multi-Mode CubeSat paper (Eq. 8) writes q_ref (x) q^-1 in yet another. Copy an
error-quaternion formula from a paper only after checking its convention --
the tests below will catch a wrong one.

NOT MODELLED HERE: estimator in the loop (that is SY1.2), sampled control
(the law is evaluated continuously), reference tracking omega_r != 0 (C5).
"""
import numpy as np

from adcs_sim.quaternion import conjugate, multiply, normalize
from adcs_sim.wheels import inertia_rw, propagate_rw

# --------------------------------------------------------------------------
# YOUR DESIGN CHOICE (criterion 2: tune to a spec on the nonlinear plant)
#   Spec (JvV Sec. 3.2 / 6.3.2.2): 10 deg step, overshoot <= 5 %,
#   1 % settling time <= 30 s.  Pick zeta and the DESIGN settling time you
#   feed to gains_from_spec so that the gate test passes. Then explain, in a
#   comment, why the design value is not simply 30 s.
# --------------------------------------------------------------------------
ZETA = 0.707
T_SETTLE_DESIGN = 28.0


# --------------------------------------------------------------------------
# YOUR TASKS
# --------------------------------------------------------------------------
def attitude_error(q_BN, q_RN):
    """Error quaternion q_BR: body B relative to the reference R.      <-- YOUR TASK

    1. Compose:  [BR] = [BN][RN]^T.  In this repo's convention that is
           q_BR = q_RN* (x) q_BN
       (prove it to yourself: to_dcm(multiply(p, q)) = to_dcm(q) @ to_dcm(p)).
    2. Shortest path (S&J p. 236): q and -q are the same attitude. If the
       scalar part is negative, return -q_BR so that beta0 >= 0 and eps
       points along the SHORT way back (rotation angle <= 180 deg).
    Return (4,) unit quaternion, scalar first.
    """
    # Step 1: q_BR = q_RN* (x) q_BN
    q_BR = multiply(conjugate(normalize(q_RN)), normalize(q_BN))
    
    # Step 2: Check for shortest path, make sure the scalar part of quaternion > 0
    if q_BR[0] < 0 :
        q_BR = -q_BR
    return q_BR
    # raise NotImplementedError("compose q_BR and take the short way")


def gains_from_spec(J, zeta, t_settle, pct=0.01):
    """Inertia-scaled PD gains from a damping ratio and settling time. <-- YOUR TASK

    1. Settling time: the error envelope decays as exp(-zeta wn t). It falls
       to pct of its start when  zeta wn t_settle = ln(1/pct)  (4.6 for 1 %):
           wn = ln(1/pct) / (zeta t_settle)
    2. JvV Eqs. 6.3.8-6.3.9:  Kp = 2 wn^2 ,  Kd = 2 zeta wn
    3. Scale by the inertia:   K = Kp [J] ,  [P] = Kd [J]
    J : (3,3).  Return (K, P), both (3,3).
    """
    # Make sure J is an array, and not list
    J = np.asarray(J, dtype=float)
    
    # Step 1
    wn = np.log(1/pct)/(zeta * t_settle)
    
    # Step 2
    Kp = 2 * (wn**2)
    Kd = 2 * zeta * wn
    
    # Step 3
    K = Kp * J
    P = Kd * J
    
    return K, P
    # raise NotImplementedError("implement JvV Eqs. 6.3.8-6.3.9")


def pd_torque(q_BN, omega, h_s, q_RN, K, P, J_RW, Gs, L_known=None):
    """Torque the wheel cluster must produce, L_r = [Gs] u_s.         <-- YOUR TASK

        L_r = K eps + [P] omega - omega x ([I_RW] omega + [Gs] h_s) + L_known
                                         (S&J Eq. 7.166 with omega_r = 0)
    eps = vector part of attitude_error(q_BN, q_RN).
    Signs: L_r is the torque ON THE WHEELS. The body feels -[Gs] u_s (A1), so
    a positive attitude error must give a positive L_r.

    Check yourself: put this into A1's rw_omega_dot and the result must be
        [I_RW] omega_dot = -K eps - [P] omega     for ANY omega and h_s.
    Return (3,).
    """
    L = np.zeros(3) if L_known is None else np.asanyarray(L_known, float)
    
    # Attitude error quaternion, take only the vector part -- scalar first
    eps = attitude_error(q_BN, q_RN)[1:]
    
    omega = np.asarray(omega, float)
    h_s = np.asarray(h_s, float)
    # Calculate required torque
    H = J_RW @ omega + Gs @ h_s         # total body + wheel momentum
    L_r = (K @ eps) + (P @ omega) - np.cross(omega, H) + L
    return L_r
    # raise NotImplementedError("implement S&J Eq. 7.166")


def allocate_pinv(L_r, Gs):
    """Minimum-norm wheel torques:  u_s = [Gs]^T ([Gs][Gs]^T)^-1 L_r   <-- YOUR TASK
    (M&C Sec. 4.8.3). Solve, do not invert.  Return (N,).
    """
    Gs = np.asarray(Gs, dtype=float)
    L_r = np.asarray(L_r, dtype=float)
    return Gs.T @ np.linalg.solve(Gs @ Gs.T, L_r)
    # raise NotImplementedError("implement the pseudoinverse")


def scale_to_limit(u, u_max):
    """Keep the commanded DIRECTION when a wheel would exceed u_max.  <-- YOUR TASK

    A1's limit_wheel_torque clips each wheel on its own (that is the plant).
    Clipping one wheel and not the others turns the body torque away from the
    commanded axis. The controller avoids it by scaling the whole vector:
        if max_i |u_i| > u_max:  u <- u * u_max / max_i |u_i|
    (Same idea as A2's saturate_dipole.)  Return (N,), a new array.
    """
    u_scale = np.array(u, dtype=float)
    max_u = np.max(np.abs(u_scale))
    if max_u > u_max:
        u_scale = u_scale * (u_max / max_u)
        
    return u_scale
    # raise NotImplementedError("implement direction-preserving scaling")


# --------------------------------------------------------------------------
# given
# --------------------------------------------------------------------------
def error_angle(q_err):
    """Principal rotation angle of an error quaternion, rad, in [0, pi]."""
    q = normalize(q_err)
    return 2.0 * np.arctan2(np.linalg.norm(q[1:]), abs(q[0]))


def signed_angle_about(q_err, axis):
    """Rotation angle of q_err measured about a fixed unit axis, rad, signed.
    For an eigenaxis manoeuvre this is the whole error; it changes sign when
    the body overshoots the target."""
    q = normalize(q_err)
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    return 2.0 * np.arctan2(q[1:] @ a, q[0])


def step_metrics(t, phi, pct=0.01):
    """Overshoot and settling time of a regulation response phi(t) -> 0.

    overshoot : largest excursion PAST zero, as a fraction of phi(0)
    t_settle  : last time |phi| exceeds pct * |phi(0)|
    """
    t, phi = np.asarray(t, float), np.asarray(phi, float)
    s = np.sign(phi[0])
    overshoot = max(0.0, -np.min(s * phi)) / abs(phi[0])
    out = np.where(np.abs(phi) > pct * abs(phi[0]))[0]
    t_settle = t[out[-1]] if len(out) else 0.0
    return overshoot, t_settle


def simulate_pointing(q0, w0, h0, q_ref, J_total, Gs, Js, K, P, t_end, dt,
                      u_max=np.inf, h_max=np.inf, L_fn=None, scale=True,
                      coulomb=0.0, viscous=0.0):
    """Closed loop: pd_torque -> allocate_pinv -> (scale_to_limit) -> A1 plant.

    The controller sees the TRUE state (no estimator) and does not know L_fn.
    scale=False leaves saturation to the plant's per-wheel clipping, to show
    what scale_to_limit buys you.
    Returns (t, q, omega, h_s, u_eff, q_err).
    """
    Gs = np.asarray(Gs, float)
    J_RW = inertia_rw(J_total, Gs, Js)

    def motor(t, q, w, h):
        u = allocate_pinv(pd_torque(q, w, h, q_ref, K, P, J_RW, Gs), Gs)
        return scale_to_limit(u, u_max) if scale else u

    t, q, w, h, u = propagate_rw(q0, w0, h0, J_RW, Gs, Js, t_end, dt, motor_fn=motor,
                                 L_fn=L_fn, u_max=u_max, h_max=h_max,
                                 coulomb=coulomb, viscous=viscous)
    q_err = np.array([attitude_error(qk, q_ref) for qk in q])
    return t, q, w, h, u, q_err
