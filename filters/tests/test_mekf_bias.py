"""Stage 2 MEKF tests: attitude + gyro bias.

THE GATE is test_GYRO_BIAS_converges -- the filter starts knowing nothing about
the bias (beta_hat = 0) against a real constant bias, and must back it out of
the attitude drift alone. A vector measurement never sees the bias directly;
only the F[0:3,3:6] = -I3 coupling makes it observable.
"""
import numpy as np

from adcs_sim.quaternion import from_axis_angle, normalize, apply, multiply, q_dot
from adcs_sim.kinematics import rk4_step
from filters.mekf_bias import MEKFBias

OMEGA = np.array([0.03, -0.02, 0.05])           # true body rate, rad/s
BIAS_TRUE = np.radians([0.5, -0.3, 0.4])        # gyro bias, rad/s (~0.5 deg/s)
S_REF = normalize([1.0, 0.2, 0.1])
B_REF = normalize([-0.3, 0.9, 0.25])
DT = 0.1

# GYRO NOISE -- M&C Eq. 6.38 densities. Both the simulated gyro AND the
# filter's Q derive from these two numbers, so they cannot disagree.
#   sigma_v : rate-noise density, rad/s^0.5  (== ARW in SI units)
#   sigma_u : bias random-walk density, rad/s^1.5
# Per-sample std of the rate noise at sample time dt is sigma_v / sqrt(dt)
# (Dan Simon Sec. 8.1.1). Earlier versions of this file used ONE number both
# as the per-sample std and squared into Q -- off by a factor of dt.
SIGMA_V = 1.1e-4                                 # rad/s^0.5  (~0.38 deg/sqrt(hr), a noisy MEMS)
SIGMA_U = np.radians(0.001)                      # rad/s^1.5  (deliberately loose)
GYRO_SAMPLE_SIG = SIGMA_V / np.sqrt(DT)          # per-sample std at DT, rad/s


def _make(q0, beta0=None):
    P0 = np.diag([np.radians(30)**2]*3 + [np.radians(1.0)**2]*3)
    Q = np.diag([SIGMA_V**2]*3 + [SIGMA_U**2]*3)
    return MEKFBias(q0=q0,
                    beta0=np.zeros(3) if beta0 is None else beta0,
                    P0=P0, Q=Q)


def _run(T=600.0, seed=0, sigma_deg=0.5, meas_every=5, q0_err_deg=20.0,
         bias_true=BIAS_TRUE):
    rng = np.random.default_rng(seed)
    q_true = from_axis_angle([0.3, -0.6, 0.2], np.radians(40))
    q_hat = multiply(from_axis_angle([0, 1, 0], np.radians(q0_err_deg)), q_true.copy())
    f = _make(q_hat)
    sigma = np.radians(sigma_deg)
    R = sigma**2 * np.eye(3)
    qt = q_true.copy()
    att, bias = [f.error_deg(qt)], [f.bias_error_deg_s(bias_true)]
    for k in range(int(T/DT)):
        t = k*DT
        qt = normalize(rk4_step(lambda q, u: q_dot(q, OMEGA), qt, t, DT))
        omega_gyro = OMEGA + bias_true + rng.normal(0, GYRO_SAMPLE_SIG, 3)
        f.predict(omega_gyro, DT)
        if k % meas_every == 0:
            for ref in (S_REF, B_REF):
                y = normalize(apply(qt, ref) + rng.normal(0, sigma, 3))
                f.update_vector(y, ref, R)
        att.append(f.error_deg(qt))
        bias.append(f.bias_error_deg_s(bias_true))
    return np.array(att), np.array(bias), f


# ---------- shape / wiring ----------
def test_predict_debiases_the_gyro():
    """With beta_hat == the true bias, a biased gyro must propagate like a clean one."""
    q0 = from_axis_angle([0.2, -0.4, 0.5], 0.8)
    f = _make(q0.copy(), beta0=BIAS_TRUE)
    qt = q0.copy()
    for k in range(200):
        t = k*DT
        qt = normalize(rk4_step(lambda q, u: q_dot(q, OMEGA), qt, t, DT))
        f.predict(OMEGA + BIAS_TRUE, DT)         # biased gyro, exact bias known
    assert f.error_deg(qt) < 1e-6


def test_bias_is_unchanged_by_predict():
    """A random walk has no deterministic drift -- predict must not move beta_hat."""
    f = _make(from_axis_angle([1, 0, 0], 0.3), beta0=np.array([0.01, -0.02, 0.03]))
    b0 = f.beta.copy()
    for _ in range(50):
        f.predict(OMEGA, DT)
    assert np.allclose(f.beta, b0)


def test_covariance_is_6x6_and_symmetric():
    f = _make(from_axis_angle([0, 1, 0], 0.4))
    f.predict(OMEGA, DT)
    assert f.P.shape == (6, 6)
    assert np.allclose(f.P, f.P.T)


def test_H_zero_block_means_one_update_barely_moves_bias():
    """A vector measurement does not see the bias directly; with P's cross-terms
    still near zero, a single update must move beta_hat only slightly."""
    q_true = from_axis_angle([0.3, -0.6, 0.2], np.radians(40))
    f = _make(multiply(from_axis_angle([0, 1, 0], np.radians(20)), q_true.copy()))
    b0 = f.beta.copy()
    f.update_vector(apply(q_true, S_REF), S_REF, np.radians(0.5)**2*np.eye(3))
    assert np.linalg.norm(f.beta - b0) < 1e-6


# ---------- THE GATE ----------
def test_GYRO_BIAS_converges():
    """THE STAGE 2 GATE.

    beta_hat starts at zero against a ~0.7 deg/s bias. The filter must back it
    out of the attitude drift alone -- proof the F[0:3,3:6] = -I3 coupling is
    right. A +I3 sign there diverges instead.
    """
    att, bias, f = _run()
    assert bias[0] > 0.5                          # started ignorant of the bias
    assert np.mean(bias[-50:]) < 0.05             # converged to within 0.05 deg/s
    assert np.mean(bias[-50:]) < bias[0] / 10     # at least a 10x improvement


def test_attitude_still_converges_with_bias_present():
    att, bias, f = _run()
    assert att[0] > 10.0                          # 20 deg initial error
    assert np.mean(att[-50:]) < 1.5


def test_uncorrected_bias_would_have_drifted():
    """Sanity: without bias estimation the attitude would run away, so the
    convergence above is genuinely the filter's doing."""
    qt = from_axis_angle([0.3, -0.6, 0.2], np.radians(40))
    q_open = qt.copy()
    for k in range(int(600.0/DT)):
        t = k*DT
        qt = normalize(rk4_step(lambda q, u: q_dot(q, OMEGA), qt, t, DT))
        q_open = normalize(rk4_step(lambda q, u: q_dot(q, OMEGA + BIAS_TRUE),
                                    q_open, t, DT))
    dq = multiply(np.array([qt[0], -qt[1], -qt[2], -qt[3]]), q_open)
    drift = 2*np.degrees(np.arcsin(np.clip(np.linalg.norm(dq[1:]), -1, 1)))
    assert drift > 30.0                           # unestimated bias => large drift
