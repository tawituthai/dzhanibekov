"""Extended Kalman Filter -- the nonlinear member of the family.

Compare with kf.py and mekf.py:
  * KF   : F, H are CONSTANT matrices. Additive update.
  * EKF  : F, H are JACOBIANS recomputed each step. Additive update.        <-- here
  * MEKF : error-state; MULTIPLICATIVE update + reset.

The ONLY difference from the linear KF is that F and H are now Jacobians
(df/dx, dh/dx) evaluated at the current estimate -- the bootstrap linearization.
The predict/update structure is otherwise identical. Reuses the same rk4_step
and covariance plumbing as every other filter (via filters._common).
"""
import numpy as np

from adcs_sim.kinematics import rk4_step
from filters._common import rk4_matrix, symmetrize, joseph_update


class EKF:
    """Hybrid EKF: continuous nonlinear dynamics, discrete measurements.

    Models you supply:
        f(x, u, t) -> dx/dt        nonlinear dynamics
        F(x, u, t) -> (n,n)        Jacobian df/dx
        h(x, t)    -> (m,)         measurement model
        H(x, t)    -> (m,n)        Jacobian dh/dx
    """

    def __init__(self, f, F, h, H, Q, R, x0, P0):
        self.f, self.F, self.h, self.H = f, F, h, H
        self.Q = np.asarray(Q, float)
        self.R = np.asarray(R, float)
        self.x = np.asarray(x0, float).copy()
        self.P = np.asarray(P0, float).copy()
        self.n = self.x.size

    def predict(self, u, t, dt):
        # state: the SHARED rk4_step, given this filter's f as f(x, tau)
        self.x = rk4_step(lambda x, tau: self.f(x, u, tau), self.x, t, dt)
        # covariance: P_dot = F P + P F^T + Q, F the Jacobian at the estimate
        def Pdot(P):
            Fj = self.F(self.x, u, t)
            return Fj @ P + P @ Fj.T + self.Q
        self.P = symmetrize(rk4_matrix(Pdot, self.P, dt))

    def update(self, y, t):
        y = np.asarray(y, float)
        Hj = self.H(self.x, t)
        S = Hj @ self.P @ Hj.T + self.R
        K = self.P @ Hj.T @ np.linalg.inv(S)
        innovation = y - self.h(self.x, t)          # nonlinear h(x)
        self.x = self.x + K @ innovation            # ADDITIVE update
        self.P = joseph_update(self.P, K, Hj, self.R)
        return innovation
