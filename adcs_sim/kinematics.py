"""Attitude kinematics and propagation  (node F3).

Conventions inherited from quaternion.py: scalar-first, Hamilton ij=k,
omega is the BODY-frame angular velocity in rad/s.

The kinematic differential equation (S&J 3.104 / 3.105, Yang 3.4.6):

    q_dot = 1/2 q (x) omega_bar        <- quaternion.q_dot  (Hamilton form)
          = 1/2 [B(q)] omega           <- q_dot_B           (matrix form, S&J 3.105)

Both are the same equation; the matrix form is what S&J writes compactly and
is convenient when you need [B] itself (e.g. the inverse map omega <- q_dot).

Propagation: integrate q_dot with RK4, then RENORMALIZE. S&J note that
q^T q = 1 is a rigorous analytic integral of the kinematics, but "in practice
the norm of beta may slightly differ from 1 when numerically integrating";
one scalar division fixes it (versus a full re-orthonormalization for a DCM).
"""
import numpy as np

from adcs_sim.quaternion import normalize, q_dot


# --------------------------------------------------------------------------
# generic integrator  (reused for the F4 rigid-body dynamics later)
# --------------------------------------------------------------------------
def rk4_step(f, x, t, dt):
    """One fourth-order Runge-Kutta step of  x_dot = f(x, t).

    Convention: k_i are SLOPES (h is NOT folded in), so the update carries
    dt/6. Your BookStack notes use the Burden & Faires form where h is folded
    into each k_i and the update is (k1+2k2+2k3+k4)/6 -- same method, and the
    two must not be mixed or you gain/lose a factor of dt.
    """
    k1 = f(x,                 t)
    k2 = f(x + 0.5*dt*k1,     t + 0.5*dt)
    k3 = f(x + 0.5*dt*k2,     t + 0.5*dt)
    k4 = f(x + dt*k3,         t + dt)
    return x + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)


def propagate(q0, omega, t_end, dt):
    """Propagate attitude from q0 for t_end seconds at fixed step dt.

    omega : either a constant array-like (3,), or a callable omega(t) -> (3,)
            giving BODY-frame angular velocity in rad/s.

    Returns (t_hist, q_hist) with shapes (N+1,) and (N+1, 4).
    Renormalizes after every step.
    """
    q = normalize(q0)
    w_of_t = omega if callable(omega) else (lambda t, _w=np.asarray(omega, float): _w)

    n_steps = int(round(t_end / dt))
    t_hist = np.zeros(n_steps + 1)
    q_hist = np.zeros((n_steps + 1, 4))
    q_hist[0] = q

    t = 0.0
    for i in range(n_steps):
        # omega is sampled at the RK4 sub-steps, exactly as Dan Simon does for
        # the control input u(t). For constant omega this is a no-op, but it
        # matters the moment omega comes from the dynamics (F4).
        q = rk4_step(lambda x, tau: q_dot(x, w_of_t(tau)), q, t, dt)
        q = normalize(q)                     # <-- S&J: reimpose q^T q = 1
        t += dt
        t_hist[i + 1] = t
        q_hist[i + 1] = q

    return t_hist, q_hist


# --------------------------------------------------------------------------
# YOUR TASK: the [B(q)] pieces you derived in your F3 notes
# --------------------------------------------------------------------------
def B_matrix(q):
    """The 4x3 matrix [B(q)] of S&J Eq. (3.106).                  <-- YOUR TASK

        [B(q)] = [[-q1, -q2, -q3],
                  [ q0, -q3,  q2],
                  [ q3,  q0, -q1],
                  [-q2,  q1,  q0]]

    Return a (4, 3) numpy array.
    """
    return np.array([
        [-q[1], -q[2],  -q[3]],
        [q[0],  -q[3],  q[2]],
        [q[3],  q[0],   -q[1]],
        [-q[2], q[1],   q[0]]
    ])


def q_dot_B(q, omega_body):
    """Kinematics in matrix form:  q_dot = 1/2 [B(q)] omega   (S&J 3.105).

    Must agree with quaternion.q_dot to machine precision -- the test checks it.  <-- YOUR TASK
    """
    return (B_matrix(q) @ omega_body) /2
    

def omega_from_qdot(q, qdot):
    """Inverse map:  omega = 2 [B(q)]^T q_dot.                    <-- YOUR TASK

    Derived (NOT a numbered equation in S&J): [B]^T [B] = I_3x3, so
    left-multiplying q_dot = 1/2 [B] omega by 2[B]^T returns omega.
    Always defined -- no inverse, no singularity.
    """
    return 2*(B_matrix(q).T @ qdot)
