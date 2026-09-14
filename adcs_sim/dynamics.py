"""Rigid-body rotational dynamics  (node F4).

Conventions inherited: scalar-first Hamilton quaternion; omega and the inertia
matrix J are BOTH expressed in the BODY frame; torque in N*m, omega in rad/s.

Euler's rotational equations of motion (S&J Eq. 4.32, Yang 4.1.1):

    [I] omega_dot = -[omega~][I] omega + L_c
    =>  omega_dot = J^-1 ( tau - omega x (J omega) )

The  omega x J omega  term is the whole story of this node: it exists only
because J is a TENSOR (different inertia about different axes), which makes the
angular momentum H = J omega point in a different direction than omega itself.
If J were a scalar this term vanishes and rotation becomes boring.

THE COUPLED 7-STATE.  Attitude and rate cannot be propagated separately --
q_dot needs omega, and omega is itself changing. So they march together:

    x = [q0 q1 q2 q3 | w1 w2 w3]        (4 + 3 = 7)
    x_dot = [ 1/2 q (x) omega  |  J^-1 (tau - omega x J omega) ]
              \___ F3 ___/        \________ F4 ________/

and the SAME generic rk4_step from kinematics.py integrates all seven at once.
Only the quaternion part carries a norm constraint, so only x[:4] is
renormalized after each step.
"""
import numpy as np

from adcs_sim.quaternion import normalize, q_dot
from adcs_sim.kinematics import rk4_step


# --------------------------------------------------------------------------
# the coupled plant  (a.k.a. the "truth model")
# --------------------------------------------------------------------------
def state_dot(x, t, J, torque_fn=None, J_inv=None):
    """Derivative of the 7-state x = [q(4), omega(3)].

    J         : (3,3) inertia matrix in the body frame.
    torque_fn : callable torque_fn(t, q, omega) -> (3,) body-frame torque.
                None means torque-free.
    J_inv     : optional precomputed inverse (avoid re-inverting every call).
    """
    q = x[:4]
    w = x[4:]
    if J_inv is None:
        J_inv = np.linalg.inv(J)
    tau = np.zeros(3) if torque_fn is None else np.asarray(torque_fn(t, q, w), float)
    return np.concatenate([q_dot(q, w), omega_dot(w, J, tau, J_inv)])


def propagate_state(q0, omega0, J, t_end, dt, torque_fn=None):
    """Propagate the coupled attitude + rate state with RK4.

    Returns (t_hist, q_hist, w_hist) of shapes (N+1,), (N+1,4), (N+1,3).
    The quaternion block is renormalized after every step; omega is not
    (it has no constraint -- it is free to be any magnitude).
    """
    J = np.asarray(J, dtype=float)
    J_inv = np.linalg.inv(J)
    x = np.concatenate([normalize(q0), np.asarray(omega0, dtype=float)])

    n = int(round(t_end / dt))
    t_hist = np.zeros(n + 1)
    x_hist = np.zeros((n + 1, 7))
    x_hist[0] = x

    t = 0.0
    for i in range(n):
        x = rk4_step(lambda xx, tau: state_dot(xx, tau, J, torque_fn, J_inv), x, t, dt)
        x[:4] = normalize(x[:4])            # only the quaternion is constrained
        t += dt
        t_hist[i + 1] = t
        x_hist[i + 1] = x

    return t_hist, x_hist[:, :4], x_hist[:, 4:]


# --------------------------------------------------------------------------
# diagnostics -- conserved quantities, torque-free case
# --------------------------------------------------------------------------
def angular_momentum_body(omega, J):
    """H = J omega, in body-frame components (S&J Eq. 4.15)."""
    return np.asarray(J, float) @ np.asarray(omega, float)


def kinetic_energy(omega, J):
    """T = 1/2 omega^T J omega."""
    w = np.asarray(omega, float)
    return 0.5 * w @ (np.asarray(J, float) @ w)


# --------------------------------------------------------------------------
# YOUR TASK
# --------------------------------------------------------------------------
def omega_dot(omega, J, tau, J_inv=None):
    """Euler's rotational equations, solved for omega_dot.      <-- YOUR TASK

        omega_dot = J^-1 ( tau - omega x (J omega) )            [S&J Eq. 4.32]

    omega : (3,) body rate           J   : (3,3) body-frame inertia
    tau   : (3,) body-frame torque   J_inv : optional precomputed inverse

    Use np.cross for the cross product. Watch the ORDER: it is omega x (J omega),
    not (J omega) x omega -- the sign matters and the tennis-racket gate will
    catch you if you flip it.

    Return a (3,) numpy array.
    """
    omega = np.asarray(omega, float)
    tau   = np.asarray(tau, float)
    J     = np.asarray(J, float)

    if J_inv is None:
        J_inv = np.linalg.inv(J)
    
    return J_inv @ (tau - np.cross(omega, (J @ omega)))

def principal_axes(J):
    """Diagonalize the inertia matrix.                          <-- YOUR TASK

    Returns (I_principal, C) where I_principal is the (3,) sorted-ascending
    vector of principal moments and C is the (3,3) DCM whose ROWS are the
    principal axes, satisfying   C @ J @ C.T == diag(I_principal).

    S&J Eqs. (4.24)-(4.27) + Example 4.2. Two traps the example calls out:
      * eigenvectors from a numerical library may not be unit length
        -> normalize them;
      * the set may be LEFT-handed -> if det(C) < 0, flip the sign of one
        row to make it a proper right-handed rotation (det = +1).

    Hint: np.linalg.eigh(J) returns (eigenvalues ascending, eigenvectors as
    COLUMNS). S&J wants the axes as ROWS of C.
    """
    J     = np.asarray(J, float)
    
    eigenvalues, eigenvectors = np.linalg.eigh(J)
    eigenvectors = eigenvectors/np.linalg.norm(eigenvectors, axis=0, keepdims=True)
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] *= -1
        
    return eigenvalues, eigenvectors.T
