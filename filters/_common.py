"""Shared plumbing for the filter family (KF / EKF / MEKF).

Everything the filters have IN COMMON lives here, so that when you compare the
three later, the only differences between kf.py, ekf.py and mekf.py are the
things that actually distinguish those filters -- not incidental plumbing.

The vector integrator is the SAME rk4_step used everywhere else in adcs_sim
(imported, not re-implemented). Covariance propagation needs to integrate a
MATRIX ODE (P_dot = F P + P F^T + Q), so we provide a thin matrix-RK4 that
follows the identical 4-slope structure.
"""
import numpy as np

from adcs_sim.kinematics import rk4_step   # the one, shared integrator


def rk4_matrix(Pdot, P, dt):
    """One RK4 step for a matrix ODE  P_dot = Pdot(P).

    Same 4-slope Simpson weighting as rk4_step, but the 'state' is a matrix.
    Used for covariance propagation in every filter.
    """
    k1 = Pdot(P)
    k2 = Pdot(P + 0.5*dt*k1)
    k3 = Pdot(P + 0.5*dt*k2)
    k4 = Pdot(P + dt*k3)
    return P + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def symmetrize(P):
    """Force symmetry -- round-off makes P drift asymmetric over many steps."""
    return 0.5*(P + P.T)


def joseph_update(P, K, H, R):
    """Joseph-form covariance update: (I-KH) P (I-KH)^T + K R K^T.

    Numerically stable -- guarantees P stays positive-definite where the naive
    (I-KH)P form can lose it to round-off. Shared by all filters that do a
    linear measurement update.
    """
    I = np.eye(P.shape[0])
    A = I - K @ H
    return symmetrize(A @ P @ A.T + K @ R @ K.T)
