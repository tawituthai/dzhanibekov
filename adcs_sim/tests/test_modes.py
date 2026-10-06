"""Node SY1 -- mode management: B-dot detumble, bias calibration, B-cross.

THE GATE is test_SY1_DETUMBLE_THEN_RATE_DAMP_BEATS_THE_BDOT_FLOOR: from a
~10 deg/s tumble with a 0.36 deg/s gyro bias, the spacecraft must switch from
B-dot to B-cross exactly once, and end far below B-dot's 2n floor -- which C1
showed B-dot alone cannot do, and B-cross with a raw gyro cannot do either.

Mode logic: ACS without an Attitude Sec. 12.3 (pp. 241-242) for why B-dot /
rate-null modes exist; Junaid (2015) Sec. 5.3.2 as the real use case for the
enter/fall-back thresholds.
"""
import numpy as np
import pytest

from adcs_sim.modes import (DETUMBLE, RATE_DAMP, ModeConfig, bias_sigma_dps, next_mode,
                            simulate_modes)
from adcs_sim.control import bdot_gain
from adcs_sim.orbit import mean_motion
from adcs_sim.quaternion import from_axis_angle

J = np.diag([0.10, 0.12, 0.05])
A_KM = 6371.0 + 500.0
N = mean_motion(A_KM)
T_ORB = 2*np.pi / N
TWO_N_DEG = np.degrees(2*N)
K = bdot_gain(T_ORB, np.radians(82.6), J.diagonal().min())
Q0 = from_axis_angle([0.3, 0.5, -0.2], 0.4)
W0 = np.radians([5.0, -6.0, 5.5])               # ~9.6 deg/s
CFG = ModeConfig()                              # 0.5 / 3.0 deg/s, gate 0.01 deg/s


def _orbit(M0_deg=0.0):
    return dict(a=A_KM, e=0.0, i=np.radians(97.4), Omega=0.3, omega=0.0, M0=np.radians(M0_deg))


def _rate_deg(run):
    return np.degrees(np.linalg.norm(run["w"], axis=1))


def _switches(run):
    """Indices where the mode changed, and the mode entered there."""
    idx = np.flatnonzero(run["mode"][1:] != run["mode"][:-1]) + 1
    return idx, [str(run["mode"][i]) for i in idx]


def _bias_err_dps(run, i):
    return np.degrees(np.linalg.norm(run["beta_hat"][i] - run["beta_true"][i]))


_CACHE = {}
def _run(name, **kw):
    if name not in _CACHE:
        _CACHE[name] = simulate_modes(Q0, kw.pop("w0", W0), J, _orbit(kw.pop("M0_deg", 0.0)),
                                      kw.pop("t_end"), K, **kw)
    return _CACHE[name]


# ---------- your task 1: the bias uncertainty from P ----------
def test_bias_sigma_is_the_worst_axis_in_deg_per_s():
    P = np.diag([1e-4, 1e-4, 1e-4,
                 np.radians(0.02)**2, np.radians(0.05)**2, np.radians(0.01)**2])
    assert np.isclose(bias_sigma_dps(P), 0.05)


def test_bias_sigma_ignores_the_attitude_block():
    P = np.diag([1.0, 1.0, 1.0] + [np.radians(0.01)**2]*3)    # huge attitude variance
    assert np.isclose(bias_sigma_dps(P), 0.01)


# ---------- your task 2: the transition logic ----------
@pytest.mark.parametrize("mode, rate, sig, expected, why", [
    (DETUMBLE,  0.3, 0.001,  RATE_DAMP, "slow and bias known: switch"),
    (DETUMBLE,  0.3, np.inf, DETUMBLE,  "filter not running yet"),
    (DETUMBLE,  0.3, 0.05,   DETUMBLE,  "bias not known well enough yet"),
    (DETUMBLE,  1.0, 0.001,  DETUMBLE,  "still too fast"),
    (RATE_DAMP, 1.0, 0.001,  RATE_DAMP, "inside the hysteresis band: stay"),
    (RATE_DAMP, 4.0, 0.001,  DETUMBLE,  "kicked above exit_dps: fall back"),
    (RATE_DAMP, 0.1, 0.001,  RATE_DAMP, "nominal"),
])
def test_next_mode_table(mode, rate, sig, expected, why):
    assert next_mode(mode, rate, sig, CFG) == expected, why


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        next_mode("pointing", 0.1, 0.001, CFG)


def test_nan_bias_sigma_keeps_detumble():
    """Every comparison with NaN is False. Write the switch condition so that
    'False' means the SAFE choice -- then a corrupted covariance can never
    push the spacecraft into the mode that trusts the gyro."""
    assert next_mode(DETUMBLE, 0.1, np.nan, CFG) == DETUMBLE


def test_hysteresis_prevents_chatter():
    """A noisy rate estimate hovering around enter_dps. With hysteresis: one
    switch. With a single threshold (exit = enter): the mode flips constantly."""
    rates = 0.5 + 0.15*np.random.default_rng(3).normal(size=600)
    def n_switches(cfg):
        mode, n = DETUMBLE, 0
        for r in rates:
            new = next_mode(mode, r, 0.001, cfg)
            n += new != mode
            mode = new
        return n
    assert n_switches(CFG) == 1
    assert n_switches(ModeConfig(enter_dps=0.5, exit_dps=0.5)) > 50


# ---------- THE GATE ----------
@pytest.mark.slow
def test_SY1_DETUMBLE_THEN_RATE_DAMP_BEATS_THE_BDOT_FLOOR():
    run = _run("nominal", t_end=2.5*3600)
    idx, entered = _switches(run)
    assert entered == [RATE_DAMP], f"mode history {entered}"         # exactly one switch
    assert run["t"][idx[0]] < T_ORB                                    # within one orbit
    assert _bias_err_dps(run, idx[0]) < 3*CFG.bias_gate_dps            # switched on a good bias
    floor = _rate_deg(run)[-1800:].mean()                              # last 30 min
    assert floor < 0.1*TWO_N_DEG, f"{floor:.4f} deg/s vs B-dot floor {TWO_N_DEG:.3f}"


# ---------- why the bias gate ----------
@pytest.mark.slow
def test_gate_waits_for_a_converged_filter():
    """Start in eclipse, already slow. The filter cannot start until the Sun
    appears (TRIAD needs two vectors), and right after it starts its bias
    estimate swings by tenths of a deg/s -- P says so (bias sigma ~0.5 deg/s).
    Gated: stays in B-dot until P says the bias is known. Ungated: switches on
    the first sample the filter runs, with the bias still completely wrong."""
    kw = dict(t_end=2100.0, M0_deg=125.0, w0=np.radians([1.2, -1.0, 1.1]))
    gated = _run("eclipse_gated", **kw)
    ungated = _run("eclipse_ungated", cfg=ModeConfig(bias_gate_dps=np.inf), **kw)

    t_init = gated["t"][np.argmax(np.isfinite(gated["bias_sig"]))]
    assert t_init > 1500.0                                             # really did start in eclipse

    idx, entered = _switches(gated)
    assert entered == [RATE_DAMP]
    assert gated["t"][idx[0]] > t_init                                 # never before the filter runs
    assert _bias_err_dps(gated, idx[0]) < 3*CFG.bias_gate_dps

    idx_u, _ = _switches(ungated)
    assert _bias_err_dps(ungated, idx_u[0]) > 10*CFG.bias_gate_dps     # trusted a garbage bias


# ---------- fall back to the mode that needs the least ----------
@pytest.mark.slow
def test_kick_falls_back_to_detumble_and_recovers():
    """A stuck thruster / venting: +7 deg/s at t = 3600 s while in RATE_DAMP.
    The logic must drop to B-dot at once, detumble again, and return to
    B-cross -- and the filter covariance must stay valid through it (this is
    the run where a non-PSD covariance step first showed up as NaN)."""
    run = _run("kick", t_end=7200.0, kick=(3600.0, np.radians([4.0, 3.0, -5.0])))
    idx, entered = _switches(run)
    assert entered == [RATE_DAMP, DETUMBLE, RATE_DAMP], f"mode history {entered}"
    assert run["t"][idx[1]] == 3600.0                                  # fell back on the kick sample
    sig = run["bias_sig"]
    assert np.all(np.isfinite(sig[np.argmax(np.isfinite(sig)):]))      # P never went invalid
    assert _rate_deg(run)[-1] < CFG.enter_dps
