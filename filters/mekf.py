"""Multiplicative Extended Kalman Filter -- Stage 1: attitude only.

The third member of the filter family. Compare with kf.py and ekf.py:
  * KF   : F, H constant.     Additive update.
  * EKF  : F, H Jacobians.    Additive update.
  * MEKF : ERROR-STATE.       MULTIPLICATIVE update + reset.               <-- here

Why a *multiplicative* filter? A quaternion cannot be updated additively --
q_hat + correction leaves the unit sphere, and the 4-vector has a redundant DOF
that makes a 4x4 covariance singular. The MEKF fixes this (Markley & Crassidis
Sec 6.2.1) by:
  * carrying the attitude as a GLOBAL unit quaternion q_hat (always valid), and
  * estimating a small 3-component ERROR rotation delta (Euclidean, 3x3 cov),
    folded into q_hat by quaternion MULTIPLICATION at the reset.

Three-step cycle (M&C):  PREDICT -> UPDATE -> RESET.
  PREDICT : propagate q_hat with the gyro; propagate P with F (Eq. 6.35).
            The error state stays zero through propagation, so it is NOT
            propagated -- only q_hat and P are.
  UPDATE  : a vector measurement (sun, mag) arrives; run the ordinary linear
            Kalman update on the ERROR state delta.
  RESET   : fold delta into q_hat multiplicatively; zero delta.

CONVENTION (critical -- a left/right mismatch converges only partway):
  This module's quaternion.apply is PASSIVE, and the reset here RIGHT-multiplies
  the error:  q_hat <- q_hat (x) dq(delta).  With that choice the vector
  measurement sensitivity is  H = skew(h_hat),  h_hat = apply(q_hat, ref).
  The reset side and the H must match; they are paired here and must stay paired.

Stage 1 assumes a PERFECT gyro (rate known exactly). Stage 2 adds gyro bias.
"""
import numpy as np

from adcs_sim.quaternion import multiply, normalize, apply, conjugate, q_dot
from adcs_sim.kinematics import rk4_step


def skew(v):
    """[v x] cross-product matrix:  skew(v) @ w == cross(v, w)."""
    v = np.asarray(v, float)
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def small_dq(delta):
    """Small rotation vector -> unit quaternion (exact normalized form).       <-- YOUR TASK

    The reset must land EXACTLY on the unit sphere, so use the normalized
    small-angle quaternion (M&C Eq. 6.26 form), not the bare [delta/2, 1]:

        dq = [1, delta/2] / || [1, delta/2] ||       (scalar-first)

    Return a length-4 scalar-first unit quaternion.
    """
    q = np.array([1.0, *(0.5*np.asarray(delta))])
    return q / np.linalg.norm(q)
    # raise NotImplementedError("build the exact normalized small-rotation quaternion")


class MEKF:
    """Stage 1 attitude-only MEKF.

    State carried:
        q_hat : (4,) global attitude quaternion (scalar-first, unit)
        P     : (3,3) covariance of the attitude ERROR delta
    The error delta itself is implicit -- it is zero except momentarily between
    update and reset, so it is never stored.
    """

    def __init__(self, q0, P0, Q):
        self.q = normalize(np.asarray(q0, float))
        self.P = np.asarray(P0, float).copy()
        self.Q = np.asarray(Q, float)          # (3,3) attitude process noise

    # ---- PREDICT : propagate q_hat (gyro) and P (error dynamics) ----
    def predict(self, omega, dt):
        """Propagate over dt using measured body rate omega (Stage 1: exact).   <-- YOUR TASK

        Two things happen, and NOTHING else (the error state stays zero):
          1. q_hat: integrate the attitude kinematics with rk4_step and q_dot,
             using omega. Renormalize after.
          2. P: propagate the error covariance with  P_dot = F P + P F^T + Q,
             where the error-dynamics matrix (M&C Eq. 6.35) is
                 F = -skew(omega)
             Integrate one Euler step:  P <- P + dt (F P + P F^T + Q).
             Keep P symmetric.
        """
        self.q = normalize(rk4_step(lambda q, u: q_dot(q, omega), self.q, 0.0, dt))
        
        F = -skew(omega)  # M&C Eq. 6.35
        self.P = self.P + dt*(F @ self.P + self.P @ F.T + self.Q)
        self.P = 0.5*(self.P + self.P.T)  # keep symmetric
        # raise NotImplementedError("implement the MEKF predict (q_hat and P)")

    # ---- UPDATE + RESET : fold one vector measurement into the estimate ----
    def update_vector(self, meas_body, ref_inertial, R): 
        """Update with one unit vector measured in body vs its inertial reference.

        meas_body    : (3,) measured unit vector in the BODY frame (e.g. sun sensor)
        ref_inertial : (3,) the same vector's known direction in the INERTIAL frame
        R            : (3,3) measurement noise covariance

        Steps (M&C update + reset, in this module's right-multiply convention):
          1. predicted measurement  h = apply(q_hat, ref_inertial)
          2. sensitivity            H = skew(h)
          3. gain  S = H P H^T + R ;  K = P H^T S^-1
          4. error  delta = K (meas_body - h)
          5. RESET  q_hat <- normalize( q_hat (x) small_dq(delta) )   # RIGHT-multiply
          6. covariance  P <- (I - K H) P ;  keep symmetric
        The error is folded in and never stored -- that is the reset.
        """
        h = apply(self.q, ref_inertial)   # predicted measurement
        H = skew(h)   # sensitivity matrix
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        delta = K @ (np.asarray(meas_body, float) - h)
        self.q = normalize(multiply(self.q, small_dq(delta)))
        self.P = (np.eye(3) - K @ H) @ self.P
        self.P = 0.5*(self.P + self.P.T)
        # raise NotImplementedError("implement the MEKF vector update + reset")

    # ---- convenience: attitude error vs a known truth (for testing/plots) ----
    def error_deg(self, q_true):
        """Angle between the estimate and a known true quaternion, in degrees."""
        dq = multiply(conjugate(np.asarray(q_true, float)), self.q)
        return 2*np.degrees(np.arcsin(np.clip(np.linalg.norm(dq[1:]), -1, 1)))
