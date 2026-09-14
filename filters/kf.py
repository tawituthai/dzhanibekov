"""Linear Kalman Filter  -- the baseline of the family.

Compare with ekf.py and mekf.py:
  * KF   : F, H are CONSTANT matrices. Additive update. No Jacobians.   <-- here
  * EKF  : F, H are Jacobians recomputed each step. Additive update.
  * MEKF : error-state; F from the error dynamics; MULTIPLICATIVE update + reset.

The predict/update cycle and all plumbing are identical across the three; only
the above differences distinguish them. That is the whole point of this family.
"""
import numpy as np

from filters._common import rk4_matrix, symmetrize, joseph_update


class KF:
    """Discrete/continuous-hybrid linear Kalman filter.

    Models (all constant matrices):
        F (n,n)  continuous state matrix   x_dot = F x + B u
        H (m,n)  measurement matrix        y = H x + v
        Q (n,n)  process noise, R (m,m) measurement noise
    """

    def __init__(self, F, H, Q, R, x0, P0, B=None):
        self.F = np.asarray(F, float)
        self.H = np.asarray(H, float)
        self.Q = np.asarray(Q, float)
        self.R = np.asarray(R, float)
        self.B = None if B is None else np.asarray(B, float)
        self.x = np.asarray(x0, float).copy()
        self.P = np.asarray(P0, float).copy()
        self.n = self.x.size

    def predict(self, u=None, dt=0.01):
        # state: x_dot = F x (+ B u)   -- linear, integrate with RK4
        def xdot(x):
            d = self.F @ x
            if self.B is not None and u is not None:
                d = d + self.B @ np.asarray(u, float)
            return d
        # simple RK4 on the vector state (linear, so this is exact-ish)
        k1 = xdot(self.x); k2 = xdot(self.x + 0.5*dt*k1)
        k3 = xdot(self.x + 0.5*dt*k2); k4 = xdot(self.x + dt*k3)
        self.x = self.x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        # covariance: P_dot = F P + P F^T + Q
        self.P = symmetrize(rk4_matrix(
            lambda P: self.F @ P + P @ self.F.T + self.Q, self.P, dt))

    def update(self, y):
        y = np.asarray(y, float)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        innovation = y - self.H @ self.x            # linear: h(x) = H x
        self.x = self.x + K @ innovation            # ADDITIVE update
        self.P = joseph_update(self.P, K, self.H, self.R)
        return innovation
