"""Node SY1, prerequisite: MEKFBias covariance propagation must keep P
positive semidefinite.

Found while building SY1: after a 7 deg/s kick, in eclipse (magnetometer only),
the bias variance went NEGATIVE and bias_sigma became NaN. The cause is the
first-order Euler step  P <- P + dt (F P + P F^T + Q).

Dan Simon Eq. 8.27 shows where that step comes from. Discretise with
F_d = I + F dt (Eq. 8.23) and Q_d = Q dt, then

    P <- (I + F dt) P (I + F dt)^T + Q dt
       = P + (F P + P F^T + Q) dt  +  F P F^T dt^2          (Eq. 8.27)

The Euler step is this with the last term thrown away. That term is positive
semidefinite, and dropping it is exactly what lets P lose positive
semidefiniteness when it is nearly singular (tiny bias variance, strong
attitude-bias correlation, high rate). The full form is a congruence
Phi P Phi^T plus a PSD Q dt, so it is PSD whenever P was -- by construction.

(The Joseph-form UPDATE does NOT fix this: it preserves PSD-ness, but cannot
restore it once the predict step has destroyed it.)
"""
import numpy as np

from filters.mekf import skew
from filters.mekf_bias import MEKFBias, Q_from_noise, gyro_noise_from_datasheet

SV, SU = gyro_noise_from_datasheet([0.13, 0.13, 0.19], [1.5, 2.3, 1.7], 1000.0)
Q = Q_from_noise(SV, SU)
OMEGA = np.array([0.08, -0.10, 0.06])       # ~8 deg/s: a kicked spacecraft
DT = 0.1


def _nearly_singular_P():
    """Rank-1 P: attitude and bias errors perfectly correlated."""
    v = np.array([1e-2, -1e-2, 1e-2, 1e-3, 1e-3, -1e-3])
    return np.outer(v, v)


def _F(omega_hat):
    F = np.zeros((6, 6))
    F[:3, :3] = -skew(omega_hat)
    F[:3, 3:] = -np.eye(3)
    return F


def test_euler_step_loses_positive_semidefiniteness():
    """Not your code -- the demonstration of the bug the next test guards against."""
    P = _nearly_singular_P()
    F = _F(OMEGA)
    P_euler = P + DT*(F @ P + P @ F.T + Q)
    assert np.linalg.eigvalsh(P_euler).min() < -1e-9


def test_predict_matches_simon_eq_8_27():
    P = _nearly_singular_P()
    f = MEKFBias([1, 0, 0, 0], np.zeros(3), P, Q)
    f.predict(OMEGA, DT)                         # beta_hat = 0, so omega_hat = OMEGA
    F = _F(OMEGA)
    expected = P + (F @ P + P @ F.T + Q)*DT + F @ P @ F.T * DT**2
    assert np.allclose(f.P, expected, rtol=1e-10, atol=1e-18)


def test_predict_keeps_P_positive_semidefinite():
    P = _nearly_singular_P()
    f = MEKFBias([1, 0, 0, 0], np.zeros(3), P, Q)
    for _ in range(100):
        f.predict(OMEGA, DT)
        lam = np.linalg.eigvalsh(f.P)
        assert lam.min() > -1e-12 * lam.max()
