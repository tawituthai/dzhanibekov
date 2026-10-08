r"""Reaction wheels  (node A1).

A reaction wheel is a flywheel on a motor, fixed along a body axis. The motor
torque u_s spins the wheel up; the equal and opposite torque spins the body the
other way. Nothing leaves the spacecraft: momentum only moves between body and
wheels. That is the exact opposite of A2 -- magnetorquers CHANGE the total
momentum (but only perpendicular to B); wheels can NEVER change it.

PRIMARY SOURCE: Schaub & Junkins, Sec. 4.3 (momentum exchange devices) and
Sec. 7.6 (reaction wheel control devices). Spacecraft with N wheels, spin axes
g_si as the columns of [Gs] (3xN), Eq. 4.120 / 7.158:

    [I_RW] omega_dot = -omega x ([I_RW] omega + [Gs] h_s) - [Gs] u_s + L

  [I_RW]  spacecraft inertia with the wheels' SPIN inertias taken out (7.159):
          [I_RW] = [I] - sum_i Js_i g_si g_si^T            (inertia_rw below)
  h_s     wheel spin-axis momenta, h_si = Js_i (g_si^T omega + Omega_i)  (7.161)
          -- the wheel's ABSOLUTE spin rate, body rate included
  Omega_i wheel speed RELATIVE to the body (what a tachometer reads)
  u_s     axial torque on each wheel, u_si = Js_i (Omega_dot_i + g_si^T omega_dot)
          (7.160) -- differentiate h_si and you get exactly this, so
                h_s_dot = u_s                                     (wheel momentum ODE)
  L       external torque (disturbances, magnetorquers)

Read the EoM term by term:
  -[Gs] u_s            the reaction: spin a wheel up (+u) and the body goes the
                       other way. Yang Eq. 10.1 says the same thing: u = -h_w_dot.
  -omega x [Gs] h_s    the gyroscopic term: stored wheel momentum, carried
                       around by a rotating body, pushes the body sideways.
                       Zero when the wheels are empty; dominant when they are full.

SATURATION (two kinds, both per wheel -- this is the plant, not the controller):
  torque  |u_si| <= u_max      the motor's current limit
  momentum |h_si| <= h_max     the maximum speed. A full wheel cannot take more
                               momentum in the same direction (Yang Sec. 10.1:
                               "once the maximum speed is reached ... one cannot
                               get the torque by increasing the flywheel speed").
  The cure for momentum saturation is an EXTERNAL torque to unload the wheels:
  magnetorquers or thrusters (ACS without an Attitude p. 104; node A5).

FRICTION (ACS without an Attitude Sec. 5.1, p. 105): stiction, Coulomb (constant,
opposite to the motion) and viscous (proportional to speed). Coulomb friction
flips sign when a wheel speed crosses zero -- a torque jolt on the body. Friction
is internal: it moves momentum between wheel and body but cannot change the total.
Stiction is not modelled here.

REAL USE CASE: Junaid (2015) Sec. 3.3.1.1 sizes a CubeSat wheel (Eqs. 3.2-3.5)
and picks the ESL "Large CubeWheel": 40 mNms, 2.1 mNm, flywheel 47.3 kg mm^2
(Sec. 3.5). The A2 rod sizing (15.291 uNm) came from the same chain.
"""
import numpy as np

from adcs_sim.kinematics import rk4_step
from adcs_sim.quaternion import normalize, q_dot, to_dcm


# --------------------------------------------------------------------------
# YOUR TASKS
# --------------------------------------------------------------------------
def rw_omega_dot(omega, h_s, u_s, J_RW, Gs, L=None):
    """Body angular acceleration of a spacecraft with N reaction wheels.  <-- YOUR TASK
    Check BookStack note -- Momentum Exchange Devices

        omega_dot = [I_RW]^-1 ( -omega x ([I_RW] omega + [Gs] h_s) - [Gs] u_s + L )
                                                             (S&J Eq. 4.120 / 7.158)
    omega : (3,) body rate, rad/s        h_s : (N,) wheel momenta, N m s
    u_s   : (N,) wheel axial torques, N m (already limited, friction included)
    J_RW  : (3,3) [I_RW]                 Gs  : (3,N) spin axes as columns
    L     : (3,) external torque, N m, or None for zero
    Return (3,).

    Check yourself: with h_s = u_s = 0 this must be exactly F4's omega_dot.
    """
    L = np.zeros(3) if L is None else np.asarray(L, float)
    
    # Step 1 
    H_sys = J_RW @ omega + Gs @ h_s
    
    #Step 2
    return np.linalg.solve(J_RW, L - Gs @ u_s - np.cross(omega, H_sys))
    # raise NotImplementedError("implement S&J Eq. 4.120")


def limit_wheel_torque(u_cmd, h_s, u_max, h_max):
    """What the wheels can actually deliver, wheel by wheel.            <-- YOUR TASK

    1. Torque limit: clip each u_cmd_i to [-u_max, +u_max].
       (Per wheel, NOT direction-preserving scaling as in A2's saturate_dipole:
       this is the PLANT. Each motor hits its own current limit regardless of
       what the others do. Keeping the commanded direction is the controller's
       job -- you will meet it in C2.)
    2. Momentum limit: if a wheel is full (|h_si| >= h_max) AND the torque would
       push it further the same way (u_i has the same sign as h_si), that wheel
       delivers 0. Torque that EMPTIES a full wheel is still allowed.

    u_cmd, h_s : (N,)    u_max, h_max : scalars    Return (N,).
    """
    u = np.array(u_cmd, dtype=float)
    h_s = np.asarray(h_s, dtype=float)
    
    u = np.clip(u, -u_max, u_max)
    
    for index in range(len(u)):
        # Momentum limit (wheel at top speed):
        # If the wheel is full (|h| >= h_max) and the torque would spin it up further
        # (u and h have the same sign, i.e. u*h > 0), the wheel can't deliver it -> 0.
        # Torque in the opposite direction brakes the wheel (empties it) and is left untouched.
        
        # (since h_dot = u: u with the same sign as h makes |h| grow, opposite sign makes it shrink)
        if (np.abs(h_s[index]) >= h_max) and (u[index]*h_s[index] > 0): 
            u[index] = 0
            
    return u
    # raise NotImplementedError("implement per-wheel torque and momentum limits")


def slew_torque(theta, I, t):
    """Wheel torque for a rest-to-rest slew of angle theta in time t,  <-- YOUR TASK
    bang-bang (accelerate half the time, brake half):

        N_slew = 4 theta I / t^2                         (Junaid Eq. 3.2)

    theta : rad    I : kg m^2 about the slew axis    t : s     Return N m.
    """
    return (4*theta*I)/(t**2)
    # raise NotImplementedError("implement Junaid Eq. 3.2")


def slew_momentum(N_slew, t):
    """Peak wheel momentum during that slew (reached at mid-slew):     <-- YOUR TASK

        h_slew = N_slew t / 2                            (Junaid Eq. 3.4)
    """
    return N_slew*t / 2
    # raise NotImplementedError("implement Junaid Eq. 3.4")


def wheel_momentum_requirement(N_d, T_orb, T_ecl, h_slew):
    """Momentum the wheel must store: the worst-case disturbance        <-- YOUR TASK
    integrated over the sunlit part of the orbit (the rods unload in eclipse),
    plus the slew peak:

        h_wheel = N_d (T_orb - T_ecl) + h_slew           (Junaid Eq. 3.5)
    """
    return (N_d * (T_orb - T_ecl)) + h_slew
    # raise NotImplementedError("implement Junaid Eq. 3.5")


# --------------------------------------------------------------------------
# given
# --------------------------------------------------------------------------
def inertia_rw(J_total, Gs, Js):
    """[I_RW] = [I] - sum_i Js_i g_si g_si^T   (S&J Eq. 4.115 with 7.159).

    J_total is the whole spacecraft, wheels included, as a mass-properties
    tool reports it. The wheels' spin inertias are removed because their spin
    momentum is carried separately in h_s."""
    Gs = np.asarray(Gs, float)
    Js = np.broadcast_to(np.asarray(Js, float), (Gs.shape[1],))
    return np.asarray(J_total, float) - (Gs * Js) @ Gs.T


def wheel_speed(h_s, omega, Gs, Js):
    """Wheel speed relative to the body (tachometer), from h_si = Js_i (g_si^T omega + Omega_i)."""
    Js = np.broadcast_to(np.asarray(Js, float), (np.asarray(Gs).shape[1],))
    return np.asarray(h_s, float) / Js - np.asarray(Gs, float).T @ np.asarray(omega, float)


def wheel_friction(Omega, coulomb=0.0, viscous=0.0):
    """Friction torque ON each wheel's axis, opposing its motion relative to the body:

        tau_f = coulomb * sign(Omega) + viscous * Omega      (ACS w/o Att. Sec. 5.1)

    The Coulomb part jumps by 2*coulomb when Omega crosses zero.
    Subtract it from the motor torque: u_s = u_motor - tau_f."""
    Omega = np.asarray(Omega, float)
    return coulomb * np.sign(Omega) + viscous * Omega


def total_momentum_inertial(q, omega, h_s, J_RW, Gs):
    """System angular momentum (body + wheels) in the inertial frame:
    H_N = [BN]^T ([I_RW] omega + [Gs] h_s). Constant when L = 0."""
    H_B = np.asarray(J_RW, float) @ np.asarray(omega, float) + np.asarray(Gs, float) @ np.asarray(h_s, float)
    return to_dcm(q).T @ H_B


def kinetic_energy_rw(omega, h_s, J_RW, Js):
    """T = 1/2 omega^T [I_RW] omega + 1/2 sum h_si^2 / Js_i   (S&J Eq. 4.118, RW case).
    Its rate must equal  omega^T L + sum Omega_i u_si          (S&J Eq. 4.119)."""
    omega, h_s = np.asarray(omega, float), np.asarray(h_s, float)
    Js = np.broadcast_to(np.asarray(Js, float), h_s.shape)
    return 0.5 * omega @ (np.asarray(J_RW, float) @ omega) + 0.5 * np.sum(h_s**2 / Js)


def propagate_rw(q0, omega0, h0, J_RW, Gs, Js, t_end, dt, motor_fn=None, L_fn=None,
                 u_max=np.inf, h_max=np.inf, coulomb=0.0, viscous=0.0):
    """RK4 on the (7+N)-state x = [q(4), omega(3), h_s(N)].

    motor_fn(t, q, omega, h_s) -> (N,) commanded motor torque (None = 0)
    L_fn(t, q, omega)          -> (3,) external torque, body frame (None = 0)

    Inside each derivative evaluation:
        u     = limit_wheel_torque(motor_fn(...), h_s, u_max, h_max)
        u_eff = u - wheel_friction(Omega, coulomb, viscous)
        omega_dot = rw_omega_dot(omega, h_s, u_eff, J_RW, Gs, L)
        h_s_dot   = u_eff
    Returns (t, q, omega, h_s, u_eff) histories; u_eff is evaluated at each
    stored sample (for energy/power checks).
    """
    Gs = np.asarray(Gs, float)
    N = Gs.shape[1]

    def parts(t, x):
        q, w, h = x[:4], x[4:7], x[7:]
        u = np.zeros(N) if motor_fn is None else np.asarray(motor_fn(t, q, w, h), float)
        u = limit_wheel_torque(u, h, u_max, h_max)
        u_eff = u - wheel_friction(wheel_speed(h, w, Gs, Js), coulomb, viscous)
        L = np.zeros(3) if L_fn is None else np.asarray(L_fn(t, q, w), float)
        return q, w, u_eff, L

    def f(x, t):
        q, w, u_eff, L = parts(t, x)
        return np.concatenate([q_dot(q, w), rw_omega_dot(w, x[7:], u_eff, J_RW, Gs, L), u_eff])

    n = int(round(t_end / dt))
    x = np.concatenate([normalize(q0), np.asarray(omega0, float), np.asarray(h0, float)])
    X, U, T = np.zeros((n + 1, 7 + N)), np.zeros((n + 1, N)), np.arange(n + 1) * dt
    X[0], U[0] = x, parts(0.0, x)[2]
    for k in range(n):
        x = rk4_step(f, x, T[k], dt)
        x[:4] = normalize(x[:4])
        X[k + 1], U[k + 1] = x, parts(T[k + 1], x)[2]
    return T, X[:, :4], X[:, 4:7], X[:, 7:], U
