"""Attitude determination  (node DET1: TRIAD).

TRIAD recovers attitude from TWO vectors, each known in both frames:
  - a REFERENCE pair in ECI  (Sun from ephemeris, field from the dipole/IGRF model)
  - a MEASURED pair in BODY   (sun sensor, magnetometer)

It builds an orthonormal triad from each pair and returns the rotation between
them. The primary vector is used whole (2 DOF); the secondary contributes only
its component perpendicular to the primary (1 DOF) -- so pass your MORE ACCURATE
sensor as primary.

    u_hat = primary
    v_hat = (primary x secondary) / |primary x secondary|   # perp axis; discards parallel part
    w_hat = u_hat x v_hat                                    # completes the frame
    M = [u_hat | v_hat | w_hat]     (axes as COLUMNS)
    A = M_body @ M_ref.T            (S&J / Yang: rotation between the two triads)

Fails if primary || secondary (cross product -> 0) or in eclipse (no Sun).
"""
import numpy as np

from adcs_sim.quaternion import normalize, from_dcm


def triad_frame(primary, secondary):
    """Build the orthonormal triad [u | v | w] (axes as columns) from two    <-- YOUR TASK
    vectors expressed in ONE frame.

        u_hat = normalize(primary)
        v_hat = normalize(cross(primary, secondary))
        w_hat = cross(u_hat, v_hat)

    Return a (3,3) numpy array whose COLUMNS are u_hat, v_hat, w_hat
    (np.column_stack is the clean way).

    Raise ValueError if primary and secondary are parallel (|cross| ~ 0) --
    there is no perpendicular component to extract, so attitude is undefined.
    """
    u_hat = primary/np.linalg.norm(primary)
    
    cross = np.cross(primary, secondary)
    if np.linalg.norm(cross) < 1e-10 :
        raise ValueError("primary and secondary are parallel")
    v_hat = cross/np.linalg.norm(cross)
    
    w_hat = np.cross(u_hat, v_hat)
    return np.column_stack([u_hat, v_hat, w_hat])


def triad_attitude(primary_body, secondary_body, primary_ref, secondary_ref):
    """TRIAD attitude matrix A = [BN] (body-from-reference DCM).

    Build the triad in the body frame and in the reference frame from the SAME
    (primary, secondary) roles, then A = M_body @ M_ref.T.
    Returns a (3,3) DCM.
    """
    M_body = triad_frame(primary_body, secondary_body)
    M_ref = triad_frame(primary_ref, secondary_ref)
    return M_body @ M_ref.T


def triad_quaternion(primary_body, secondary_body, primary_ref, secondary_ref):
    """Same as triad_attitude but returns a scalar-first quaternion (via Stanley)."""
    return from_dcm(triad_attitude(primary_body, secondary_body,
                                   primary_ref, secondary_ref))
