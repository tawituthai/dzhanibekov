"""Multiplicative EKF -- Stage 2: attitude + gyro bias.

Stage 1 (MEKF, above) assumed a perfect gyro. A real gyro reads

    omega_gyro = omega_true + beta + noise

where beta is a slowly-drifting BIAS. Integrating a biased rate makes the
attitude estimate drift without bound, so the bias must be estimated and
subtracted. That is what this filter adds, and it is the standard flight
configuration -- M&C: "the traditional attitude on-orbit calibration approach
uses the 6-state EKF ... to estimate the current attitude and gyro biases
simultaneously."

WHAT CHANGES FROM STAGE 1 (and nothing else does):

  state        q_hat, P(3x3)            ->  q_hat, beta_hat, P(6x6)
  error state  [dtheta]                 ->  [dtheta, dbeta]
  predict rate omega_gyro               ->  omega_gyro - beta_hat
  F            -skew(w)                 ->  [[-skew(w), -I3], [0, 0]]
  H            skew(h)                  ->  [skew(h), 0_3x3]
  reset        quaternion only          ->  quaternion (mult) + bias (add)

THE KEY BLOCK is F[0:3, 3:6] = -I3. It comes from the error dynamics
(M&C Eq. 6.35)

    dtheta_dot = -omega_hat x dtheta + domega,     domega = -dbeta

i.e. an error in the bias estimate feeds directly into the attitude error.
That coupling is the ONLY reason the bias is observable from attitude
measurements: a wrong bias makes the attitude drift in a predictable way, the
sun/mag measurements see the drift, and the filter backs the bias out of it.
Set that block to +I3 instead and the filter diverges outright.

The bias itself is modelled as a random walk (M&C Eq. 6.38b: beta_dot = eta_u),
so it has no deterministic dynamics -- F[3:6, :] is all zeros -- and its only
propagation is the growth of its covariance through Q.
"""
import numpy as np

from adcs_sim.quaternion import multiply, normalize, apply, conjugate, q_dot
from adcs_sim.kinematics import rk4_step
from filters.mekf import skew, small_dq


class MEKFBias:
    """6-state MEKF: attitude error (3) + gyro-bias error (3).

    Carried state:
        q_hat    : (4,) global attitude quaternion
        beta_hat : (3,) gyro bias estimate, rad/s
        P        : (6,6) covariance of the error state [dtheta, dbeta]
    """

    def __init__(self, q0, beta0, P0, Q):
        self.q = normalize(np.asarray(q0, float))
        self.beta = np.asarray(beta0, float).copy()
        self.P = np.asarray(P0, float).copy()      # (6,6)
        self.Q = np.asarray(Q, float)              # (6,6): attitude block + bias random walk

    # ---- PREDICT ----------------------------------------------------
    def predict(self, omega_gyro, dt):
        """Propagate q_hat, beta_hat and P over dt.                       <-- YOUR TASK

        1. DEBIAS the gyro:      omega_hat = omega_gyro - beta_hat
           (M&C Table 6.1 propagation, with S_hat = 0 -- no scale/misalignment)
        2. PROPAGATE q_hat with omega_hat, exactly as Stage 1 (rk4_step + q_dot,
           then renormalize).
        3. beta_hat does NOT change -- a random walk has no deterministic drift.
        4. PROPAGATE P with  P <- P + dt (F P + P F^T + Q),  where

               F = [[ -skew(omega_hat),  -I3 ],
                    [  0_3x3,             0_3x3 ]]

           Build it with np.zeros((6,6)) and fill the two nonzero blocks.
           Keep P symmetric.
        """
        omega = omega_gyro - self.beta
        self.q = normalize(rk4_step(lambda q, u: q_dot(q, omega), self.q, 0.0, dt))
        
        F = np.zeros((6, 6))
        F[:3, :3] = -skew(omega)
        F[:3, 3:] = -np.eye(3)
        # assert F.shape == (6,6)
        
        self.P = self.P + dt*(F @ self.P + self.P @ F.T + self.Q)
        self.P = 0.5*(self.P + self.P.T)
        # raise NotImplementedError("implement the Stage 2 predict (debias, q_hat, P)")

    # ---- UPDATE + RESET ---------------------------------------------
    def update_vector(self, meas_body, ref_inertial, R):
        """Fold in one body-frame unit vector measurement.                <-- YOUR TASK

        Same shape as Stage 1, widened to 6 states:

          1. h = apply(q_hat, ref_inertial)                    (predicted measurement)
          2. H = [skew(h), 0_3x3]              (3x6 -- a vector measurement sees
                                                attitude only, never the bias directly)
          3. S = H P H^T + R ;  K = P H^T S^-1                 (K is 6x3)
          4. delta = K (meas_body - h)                         (6-vector)
          5. RESET:
                 q_hat    <- normalize( q_hat (x) small_dq(delta[:3]) )   # multiplicative
                 beta_hat <- beta_hat + delta[3:]                          # additive
             (M&C 6.23a for the quaternion, 6.23b for the bias)
          6. P <- (I6 - K H) P ; keep symmetric

        Note H's zero block: the bias is NOT measured. It is inferred only
        through the F coupling during propagation. That is the whole trick.
        """
        h = apply(self.q, ref_inertial)
        H = np.hstack([skew(h), np.zeros((3, 3))])  # Horizontal concatenation
        # assert H.shape == (3,6)
        
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        # assert K.shape == (6,3)
        
        delta = K @ (np.asarray(meas_body, float) - h)
        # assert delta.shape == (6,)
        
        self.q = normalize(multiply(self.q, small_dq(delta[:3])))   # only top-3 is attitude quaternion
        
        self.beta = self.beta + delta[3:]     # the last-3 is bias estimate
        
        self.P = (np.eye(6) - K @ H) @ self.P
        self.P = 0.5*(self.P + self.P.T)
        # raise NotImplementedError("implement the Stage 2 vector update + reset")

    # ---- diagnostics -------------------------------------------------
    def error_deg(self, q_true):
        """Attitude error vs a known truth, in degrees."""
        dq = multiply(conjugate(np.asarray(q_true, float)), self.q)
        return 2*np.degrees(np.arcsin(np.clip(np.linalg.norm(dq[1:]), -1, 1)))

    def bias_error_deg_s(self, beta_true):
        """Gyro-bias estimate error magnitude, in deg/s."""
        return np.degrees(np.linalg.norm(self.beta - np.asarray(beta_true, float)))

    def error_state(self, q_true, beta_true):
        """The true 6-vector error [dtheta(3); dbeta(3)] the filter is estimating.

        The reset RIGHT-multiplies (q_hat <- q_hat (x) dq), so the error
        quaternion consistent with that convention is

            dq = conj(q_true) (x) q_hat,      dtheta = 2 * dq[1:]

        (sign-fixed so dq[0] >= 0 -- q and -q are the same rotation).
        This is the quantity whose covariance P claims to be; nees() below
        turns it into a consistency statistic.
        """
        dq = multiply(conjugate(np.asarray(q_true, float)), self.q)
        if dq[0] < 0:
            dq = -dq
        dtheta = 2.0 * dq[1:]
        dbeta = self.beta - np.asarray(beta_true, float)
        return np.concatenate([dtheta, dbeta])

    def nees(self, q_true, beta_true):
        """Normalized Estimation Error Squared:  e^T P^-1 e,  e = error_state.

        For a CONSISTENT filter this averages to the state dimension, 6.
        Much above 6 => P is too small, the filter is overconfident (Q or R
        understated). Much below 6 => P is too big, the filter is timid and
        throwing away information. This is the number that tells you whether
        your datasheet-derived Q is actually right.
        """
        e = self.error_state(q_true, beta_true)
        return float(e @ np.linalg.solve(self.P, e))


# --------------------------------------------------------------------------
# YOUR TASK -- node E3, criterion 2: build Q from a real datasheet
# --------------------------------------------------------------------------
def gyro_noise_from_datasheet(arw_deg_rthr, bias_instab_deg_hr, tau_bias_s=1000.0):
    """Datasheet numbers -> the two spectral densities in M&C Eq. 6.38. <-- YOUR TASK

    M&C's gyro model (Eq. 6.38) is

        omega_gyro = omega_true + beta + eta_v ,   E[eta_v eta_v^T] = sigma_v^2 I d(t-t')
        beta_dot   = eta_u                     ,   E[eta_u eta_u^T] = sigma_u^2 I d(t-t')

    Both sigmas are DENSITIES, not per-sample standard deviations. They are
    what goes into Q for the covariance ODE  P_dot = F P + P F^T + Q:

        Q = diag( sigma_v^2 * ones(3),  sigma_u^2 * ones(3) )

    INPUT 1 -- ARW (angle random walk), quoted in deg/sqrt(hr).
        This IS sigma_v, in disguised units. It says: integrate the gyro for
        t seconds with no signal and the angle error grows as sigma_v*sqrt(t).
        Convert deg/sqrt(hr) -> rad/sqrt(s):
            deg -> rad   : multiply by pi/180
            /sqrt(hr) -> /sqrt(s) : divide by sqrt(3600) = 60
        so   sigma_v = arw_deg_rthr * (pi/180) / 60      [rad/s^0.5]

    INPUT 2 -- in-run bias instability, quoted in deg/hr.
        This is NOT sigma_u. It is the FLOOR of the Allan deviation -- the
        best the bias ever gets, at one particular averaging time. The random
        walk that sigma_u describes is the rising +1/2-slope branch to its
        right, and datasheets almost never tabulate it (on the ADIS16505 you
        can only read it off the Allan plot, Fig. 7).
        So you BOUND it: assume the bias wanders by about the bias-instability
        amount over a correlation time tau_bias_s. A random walk covers
        sigma_u*sqrt(tau) in time tau, hence
            sigma_u ~= BI_rad_per_s / sqrt(tau_bias_s)   [rad/s^1.5]
        with BI_rad_per_s = bias_instab_deg_hr * (pi/180) / 3600.
        tau_bias_s = 1000 is a common default. STATE IT AS AN ASSUMPTION --
        this is an engineering bound, not a datasheet value.

    Accept scalars or (3,) arrays for either input (the ADIS16505 quotes a
    different ARW for z than for x,y). Return (sigma_v, sigma_u), each a
    (3,) array, by np.broadcast_to-ing scalars up to 3.
    """
    sigma_v = np.asarray(arw_deg_rthr) * (np.pi/180) / 60  # unit [rad/s^0.5]
    
    bias_rad_per_s = np.asarray(bias_instab_deg_hr) * (np.pi/180) / 3600
    sigma_u = bias_rad_per_s / np.sqrt(tau_bias_s) # unit [rad/s^1.5]
    
    return sigma_v, sigma_u
    # raise NotImplementedError("implement the datasheet -> (sigma_v, sigma_u) conversion")


def Q_from_noise(sigma_v, sigma_u):
    """Assemble the 6x6 process-noise density from the two gyro densities.

        Q = diag( sigma_v^2 (3),  sigma_u^2 (3) )      [M&C Eq. 6.38]

    Units: the top block is rad^2/s (angle-rate noise density), the bottom
    block rad^2/s^3 (bias random-walk density). Both are DENSITIES -- the
    covariance ODE multiplies them by dt itself. If you ever switch to a
    discrete propagation P <- Phi P Phi^T + Q_d, you need Q_d = Q*dt
    (Dan Simon Eq. 8.11), not Q.
    """
    sv = np.broadcast_to(np.asarray(sigma_v, float), (3,))
    su = np.broadcast_to(np.asarray(sigma_u, float), (3,))
    return np.diag(np.concatenate([sv**2, su**2]))
