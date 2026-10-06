r"""ADCS mode management  (node SY1: detumble -> calibrate -> rate damping).

C1 left two facts on the table:
  * B-dot needs no gyro, but cannot go below ~2x orbital rate (the A R_dot
    term of M&C Eq. 7.52; ACS without an Attitude Sec. 12.3, p. 242: B-dot
    "creates a steady-state situation where the spacecraft follows the
    geomagnetic field, resulting in a twice-orbital frequency vehicle rotation
    rate").
  * B-cross has no such floor, but it drives the MEASURED rate to zero, so the
    true rate settles at whatever gyro bias is left uncorrected.
E3 is what removes the bias: the 6-state MEKF estimates beta and, through P,
says how well it knows it.

So the flight logic is a two-state machine:

    DETUMBLE  --[ rate < enter_dps  AND  bias known to < bias_gate_dps ]-->  RATE_DAMP
    RATE_DAMP --[ rate > exit_dps ]---------------------------------------->  DETUMBLE

  DETUMBLE  : B-dot (magnetometer only). The MEKF runs in the background from
              the first sample on which it can be initialised (TRIAD needs the
              Sun, so not in eclipse), learning the bias while B-dot works.
  RATE_DAMP : B-cross fed by the DEBIASED gyro, omega_gyro - beta_hat.

Two design rules, both visible in the real use case (Junaid 2015, Sec. 5.3.2:
enters the stabilised mode below 0.3 deg/s, falls back to the detumble (Safe)
mode "at any instant in the mission orbit, if the satellite body rates exceed a
threshold of 3 deg/s"):
  1. HYSTERESIS. exit_dps is well above enter_dps. With one threshold, noise on
     the rate estimate near that threshold would flip the mode every sample.
  2. FALL BACK TO THE MODE THAT NEEDS THE LEAST. B-dot needs only a
     magnetometer and works at any rate, so it is the safe mode to return to
     when something kicks the spacecraft (a thruster stuck on, a venting tank:
     ACS Sec. 12.3 lists exactly these as why B-dot modes exist).

The rate the mode logic looks at is |omega_gyro - beta_hat| -- the best rate
estimate available on board. Before the filter is initialised beta_hat = 0 and
the bias is unknown (bias_sigma = inf), so the gate holds the spacecraft in
DETUMBLE.

NOT MODELLED (on purpose, for now): magnetorquer/magnetometer interference
(real systems time-share: rods off while the magnetometer samples), IGRF model
error in the filter's reference field, actuator on-time quantisation.
"""
import numpy as np
from dataclasses import dataclass

from adcs_sim.actuators import magnetorquer_torque
from adcs_sim.control import estimate_bdot, bdot_law, bcross_law
from adcs_sim.determination import triad_quaternion
from adcs_sim.dynamics import propagate_state
from adcs_sim.environment import (magnetic_field_eci, magnetometer, sun_sensor,
                                  sun_vector_eci, in_eclipse)
from adcs_sim.orbit import elements_to_rv, mean_motion
from adcs_sim.quaternion import apply, normalize
from filters.mekf_bias import MEKFBias, gyro_noise_from_datasheet, Q_from_noise

DETUMBLE = "detumble"
RATE_DAMP = "rate_damp"


@dataclass
class ModeConfig:
    enter_dps: float = 0.5        # DETUMBLE -> RATE_DAMP below this rate
    exit_dps: float = 3.0         # RATE_DAMP -> DETUMBLE above this rate (Junaid: 3 deg/s)
    bias_gate_dps: float = 0.01   # ... and only if the bias 1-sigma is below this


# --------------------------------------------------------------------------
# YOUR TASKS
# --------------------------------------------------------------------------
def bias_sigma_dps(P):
    """How well the filter knows the gyro bias, from its covariance.  <-- YOUR TASK

    P is MEKFBias's 6x6 covariance of [dtheta(3); dbeta(3)]. The bias block is
    P[3:6, 3:6], in (rad/s)^2. Return the 1-sigma of the WORST-known axis,
    in deg/s:
        sqrt( max of the three bias variances )   ->  degrees
    (The worst axis, not the average: one badly-known axis is enough to leave
    a rate floor on that axis.)
    """
    bias_matrix = P[3:6, 3:6]
    bias_axis = np.diag(bias_matrix)
    bias_maximum = np.max(bias_axis)
    
    return np.sqrt(bias_maximum)*(180/np.pi)    
    # raise NotImplementedError("implement the worst-axis bias 1-sigma in deg/s")


def next_mode(mode, rate_dps, bias_sig_dps, cfg):
    """The mode transition logic.                                     <-- YOUR TASK

    mode         : DETUMBLE or RATE_DAMP (the current mode)
    rate_dps     : |omega_gyro - beta_hat| in deg/s (best on-board rate estimate)
    bias_sig_dps : bias_sigma_dps(P), or np.inf if the filter is not running yet
    cfg          : ModeConfig

    Rules (module docstring):
      DETUMBLE  -> RATE_DAMP  if rate < cfg.enter_dps AND bias_sig < cfg.bias_gate_dps
      RATE_DAMP -> DETUMBLE   if rate > cfg.exit_dps
      otherwise stay where you are.
    Raise ValueError for an unknown mode -- flight software must never fall
    through silently on a corrupted mode variable.
    """
    if (rate_dps < cfg.enter_dps) and (bias_sig_dps < cfg.bias_gate_dps) : 
        mode = RATE_DAMP
    elif (rate_dps > cfg.exit_dps) :
        mode = DETUMBLE
    return mode
    # raise NotImplementedError("implement the DETUMBLE <-> RATE_DAMP transition logic")


# --------------------------------------------------------------------------
# given: the closed loop
# --------------------------------------------------------------------------
# ADIS16505-1 datasheet numbers, as in node E3 (filters/tests/test_e3_datasheet.py)
ARW = np.array([0.13, 0.13, 0.19])      # deg/sqrt(hr)
BI = np.array([1.5, 2.3, 1.7])          # deg/hr
TAU_B = 1000.0                          # s, assumed (E3)


def simulate_modes(q0, w0, J, orbit, t_end, k, cfg=None, gyro_bias_dps=(0.2, -0.25, 0.15),
                   m_max=0.2, Ts=1.0, dt_gyro=0.1, t0=0.0, days=0.0,
                   sigma_mag_nT=50.0, sigma_sun_deg=0.5, kick=None, rng=None):
    """Closed loop: sensors -> MEKFBias -> next_mode -> B-dot or B-cross -> rods.

    Ts       : control / measurement period, s (dipole held constant over it)
    dt_gyro  : gyro sample period, s. The filter predicts at this rate: at a
               10 deg/s tumble the rate changes noticeably within one Ts, and
               predicting with a 1 s sample-and-hold would corrupt the filter.
    gyro_bias_dps : true turn-on bias, deg/s (a MEMS gyro straight out of
               launch: a few tenths of a deg/s). It then random-walks with the
               ADIS16505 sigma_u, and the gyro adds ADIS16505 ARW -- the SAME
               numbers the filter's Q is built from (E3).
    kick     : optional (t_kick, d_omega_rad_s): an instantaneous rate change
               at t_kick, e.g. a stuck thruster or venting.
    days     : days since the vernal equinox (Sun direction).

    Each control step:
      1. magnetometer (always) and sun sensor (only when not in eclipse)
      2. if the filter is not yet running and the Sun is visible: initialise
         it with TRIAD (magnetometer primary: the more accurate sensor here)
      3. filter measurement updates (mag direction; sun if visible)
      4. mode = next_mode(mode, |omega_gyro - beta_hat|, bias_sigma, cfg)
      5. dipole from B-dot or B-cross(omega_gyro - beta_hat)
      6. propagate truth over Ts in dt_gyro substeps; the filter predicts with
         each gyro sample.
    Returns dict of arrays (one row per control step): t, w (true rate),
    mode (str), beta_hat, beta_true, bias_sig (deg/s, inf before init),
    att_err_deg (nan before init), m (dipole).
    """
    cfg = ModeConfig() if cfg is None else cfg
    rng = np.random.default_rng(0) if rng is None else rng
    J = np.asarray(J, float)
    n = mean_motion(orbit["a"])
    sun_eci = sun_vector_eci(days)

    sv, su = gyro_noise_from_datasheet(ARW, BI, TAU_B)
    Q = Q_from_noise(sv, su)
    P0 = np.diag([np.radians(10.0)**2]*3 + [np.radians(0.5)**2]*3)
    R_sun = np.radians(sigma_sun_deg)**2 * np.eye(3)

    def r_at(t):
        r, _ = elements_to_rv(orbit["a"], orbit["e"], orbit["i"], orbit["Omega"],
                              orbit["omega"], orbit["M0"] + n*t)
        return r

    q, w = normalize(np.asarray(q0, float)), np.asarray(w0, float).copy()
    beta = np.radians(np.asarray(gyro_bias_dps, float))
    sub = int(round(Ts / dt_gyro))
    steps = int(round(t_end / Ts))

    out = {key: [] for key in ("t", "w", "mode", "beta_hat", "beta_true",
                               "bias_sig", "att_err_deg", "m")}
    filt, mode, B_prev = None, DETUMBLE, None
    w_gyro = w + beta + rng.normal(0, 1, 3) * sv / np.sqrt(dt_gyro)

    for kstep in range(steps):
        t = kstep * Ts
        if kick is not None and np.isclose(t, kick[0]):
            w = w + np.asarray(kick[1], float)
            w_gyro = w + beta + rng.normal(0, 1, 3) * sv / np.sqrt(dt_gyro)
        r = r_at(t)
        Be0, Be1 = magnetic_field_eci(r, t0 + t), magnetic_field_eci(r_at(t + Ts), t0 + t + Ts)
        lit = not in_eclipse(r, sun_eci)

        # 1. sensors
        B_meas = magnetometer(q, Be0, sigma_nT=sigma_mag_nT, rng=rng)
        s_meas = sun_sensor(q, sun_eci, sigma_deg=sigma_sun_deg, rng=rng) if lit else None
        b_ref = Be0 / np.linalg.norm(Be0)
        R_mag = (sigma_mag_nT / np.linalg.norm(Be0))**2 * np.eye(3)

        # 2-3. filter
        if filt is None and lit:
            q_init = triad_quaternion(B_meas, s_meas, b_ref, sun_eci)
            filt = MEKFBias(q_init, np.zeros(3), P0, Q)
        if filt is not None:
            filt.update_vector(B_meas / np.linalg.norm(B_meas), b_ref, R_mag)
            if lit:
                filt.update_vector(s_meas, sun_eci, R_sun)
        beta_hat = np.zeros(3) if filt is None else filt.beta
        sig = np.inf if filt is None else bias_sigma_dps(filt.P)

        # 4. mode
        w_est = w_gyro - beta_hat
        mode = next_mode(mode, np.degrees(np.linalg.norm(w_est)), sig, cfg)

        # 5. control law
        if mode == DETUMBLE:
            m = np.zeros(3) if B_prev is None else bdot_law(estimate_bdot(B_meas, B_prev, Ts), B_meas, k)
        else:
            m = bcross_law(w_est, B_meas, k)
        B_prev = B_meas

        out["t"].append(t); out["w"].append(w.copy()); out["mode"].append(mode)
        out["beta_hat"].append(beta_hat.copy()); out["beta_true"].append(beta.copy())
        out["bias_sig"].append(sig); out["m"].append(m)
        out["att_err_deg"].append(np.nan if filt is None else filt.error_deg(q))

        # 6. propagate truth in gyro substeps; filter predicts on each sample
        for j in range(sub):
            ts = j * dt_gyro
            def torque_fn(tt, qq, ww, ts=ts, m=m):
                Be = Be0 + (Be1 - Be0) * ((ts + tt) / Ts)
                return magnetorquer_torque(m, apply(qq, Be), m_max)
            if filt is not None:
                filt.predict(w_gyro, dt_gyro)
            _, qh, wh = propagate_state(q, w, J, dt_gyro, dt_gyro, torque_fn)
            q, w = qh[-1], wh[-1]
            beta = beta + rng.normal(0, 1, 3) * su * np.sqrt(dt_gyro)
            w_gyro = w + beta + rng.normal(0, 1, 3) * sv / np.sqrt(dt_gyro)

    return {key: np.array(v) for key, v in out.items()}
