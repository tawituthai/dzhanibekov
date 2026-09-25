"""Node E3, criterion 2: tune the bias-estimating MEKF from a REAL datasheet.

Gyro: Analog Devices ADIS16505-1 (the +/-125 deg/s grade -- right range for a
CubeSat). Numbers from its datasheet specification table:

    angle random walk     0.13 deg/sqrt(hr) (x, y)   0.19 deg/sqrt(hr) (z)
    in-run bias stability 1.5 / 2.3 / 1.7 deg/hr     (x / y / z)

Rate random walk is NOT tabulated -- it is only visible as the +1/2-slope
branch of the Allan deviation plot -- so sigma_u is bounded from the bias
instability with an assumed correlation time (gyro_noise_from_datasheet
docstring).

THE GATE is test_DATASHEET_Q_IS_CONSISTENT: a Monte Carlo where the simulated
gyro is generated from the SAME (sigma_v, sigma_u) that build Q. If the
conversion is right, the filter's covariance P matches its actual errors and
the NEES averages to the state dimension, 6. Wrong units (the classic slip is
forgetting the /60 or the /sqrt(dt)) shows up as NEES far from 6.
"""
import numpy as np

from adcs_sim.quaternion import from_axis_angle, normalize, apply, multiply, q_dot
from adcs_sim.kinematics import rk4_step
from filters.mekf import small_dq
from filters.mekf_bias import MEKFBias, gyro_noise_from_datasheet, Q_from_noise

ARW = np.array([0.13, 0.13, 0.19])      # deg/sqrt(hr)
BI = np.array([1.5, 2.3, 1.7])          # deg/hr
TAU_B = 1000.0                          # s  -- ASSUMED bias correlation time

OMEGA = np.array([0.03, -0.02, 0.05])
S_REF = normalize([1.0, 0.2, 0.1])
B_REF = normalize([-0.3, 0.9, 0.25])
DT = 0.1


# ---------- the unit conversion itself ----------
def test_arw_converts_to_sigma_v():
    sv, _ = gyro_noise_from_datasheet(ARW, BI, TAU_B)
    # 0.13 deg/sqrt(hr) * (pi/180) / 60  =  3.7815e-5 rad/s^0.5
    assert np.allclose(sv, [3.7815e-5, 3.7815e-5, 5.5269e-5], rtol=1e-4)


def test_bias_instability_bounds_sigma_u():
    _, su = gyro_noise_from_datasheet(ARW, BI, TAU_B)
    # 1.5 deg/hr = 7.272e-6 rad/s ; / sqrt(1000) = 2.2997e-7 rad/s^1.5
    assert np.allclose(su, [2.2997e-7, 3.5262e-7, 2.6063e-7], rtol=1e-4)


def test_scalars_broadcast_to_three_axes():
    sv, su = gyro_noise_from_datasheet(0.13, 1.5)
    assert np.shape(sv) == (3,) and np.shape(su) == (3,)
    assert np.allclose(sv, sv[0]) and np.allclose(su, su[0])


def test_longer_correlation_time_means_smaller_sigma_u():
    """Same wander spread over a longer time => a slower random walk."""
    _, su_short = gyro_noise_from_datasheet(ARW, BI, 100.0)
    _, su_long = gyro_noise_from_datasheet(ARW, BI, 10000.0)
    assert np.allclose(su_short / su_long, 10.0)


# ---------- what ARW means physically ----------
def test_ARW_is_sample_rate_independent():
    """Integrate a stationary gyro. The angle drift after t seconds is
    sigma_v*sqrt(t) whether you sample at 1 Hz or 100 Hz -- the per-sample
    noise is different, the DENSITY is not. This is why Q uses sigma_v^2 and
    not the per-sample variance."""
    sv, _ = gyro_noise_from_datasheet(ARW, BI, TAU_B)
    sv = sv[0]
    T = 900.0                                        # 15 minutes
    rng = np.random.default_rng(1)
    for dt in (1.0, 0.01):
        n = int(T / dt)
        # 400 independent gyros, each integrated over T
        rate = rng.normal(0.0, sv / np.sqrt(dt), size=(400, n))
        angle = rate.sum(axis=1) * dt
        assert abs(np.std(angle) / (sv * np.sqrt(T)) - 1.0) < 0.12


# ---------- THE GATE ----------
def _nees_run(Q_scale, seed, sv, su, T=1200.0, sigma_deg=0.5, meas_every=10):
    """One run where truth and filter share the same noise model.
    Returns the NEES time history."""
    rng = np.random.default_rng(seed)
    q_true = from_axis_angle([0.3, -0.6, 0.2], np.radians(40))
    beta_true = np.radians([0.05, -0.03, 0.04])      # ~180 deg/hr turn-on bias
    P0 = np.diag([np.radians(5)**2]*3 + [np.radians(0.1)**2]*3)

    # draw the initial error FROM P0, so P0 is honest from t = 0
    e0 = rng.multivariate_normal(np.zeros(6), P0)
    q_hat = normalize(multiply(q_true, small_dq(e0[:3])))
    f = MEKFBias(q_hat, beta_true + e0[3:], P0, Q_from_noise(sv, su) * Q_scale)

    R = np.radians(sigma_deg)**2 * np.eye(3)
    out = []
    for k in range(int(T / DT)):
        q_true = normalize(rk4_step(lambda q, u: q_dot(q, OMEGA), q_true, 0.0, DT))
        beta_true = beta_true + rng.normal(0, 1, 3) * su * np.sqrt(DT)    # bias walks
        omega_gyro = OMEGA + beta_true + rng.normal(0, 1, 3) * sv / np.sqrt(DT)
        f.predict(omega_gyro, DT)
        if k % meas_every == 0:
            for ref in (S_REF, B_REF):
                y = normalize(apply(q_true, ref) + rng.normal(0, np.radians(sigma_deg), 3))
                f.update_vector(y, ref, R)
        out.append(f.nees(q_true, beta_true))
    return np.array(out)


def _mean_nees(Q_scale, n_seeds=8):
    """Monte-Carlo mean NEES over the SECOND half of each run.

    The first half is excluded because a mis-tuned filter starts out honest
    (P0 is honest by construction) and only becomes over- or under-confident
    as the wrong Q accumulates in P."""
    sv, su = gyro_noise_from_datasheet(ARW, BI, TAU_B)
    runs = [_nees_run(Q_scale, s, sv, su) for s in range(n_seeds)]
    return np.mean([r[len(r)//2:] for r in runs])


def test_DATASHEET_Q_IS_CONSISTENT():
    """THE E3 GATE. Datasheet-derived Q => mean NEES ~ 6 (the state dimension).

    The band is wide-ish because runs are time-correlated and the covariance
    uses first-order Euler propagation of P. The exact discrete Q has
    dt^2 and dt^3 cross terms -- see the Stellenbosch thesis, Eqs. B.28-B.32:
    Q_k = Ts*S1 + Ts^2/2*S2 + Ts^3/3*S3. At dt = 0.1 s they are negligible."""
    nees = _mean_nees(1.0)
    assert 4.5 < nees < 10.0, f"mean NEES {nees:.2f}, expected ~6"


def test_understated_Q_is_overconfident():
    """Q / 100 -- e.g. forgetting that ARW is per sqrt(HOUR). P too small."""
    assert _mean_nees(0.01) > 30.0


def test_overstated_Q_is_timid():
    """Q * 100 -- P too big, the filter discounts good gyro data."""
    assert _mean_nees(100.0) < 3.5
