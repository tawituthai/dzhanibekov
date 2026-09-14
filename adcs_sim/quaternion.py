"""Quaternion module for adcs-sim  (node F2: attitude representation).

CONVENTIONS (locked — see BookStack F2 notes). Mixing these silently
transposes rotations, so they are stated once, here, and never deviated from:

  * Scalar-first:      q = [q0, q1, q2, q3], q0 is the scalar part.
  * Hamilton product:  ij = k   (matches Schaub & Junkins and Yang).
                       NOT Markley & Crassidis (scalar-last q4 + transpose).
  * to_dcm(q):         the S&J (3.92) attitude matrix [C], used PASSIVELY:
                           v_target = to_dcm(q) @ v_source
                       i.e. it re-expresses a vector's coordinates in the
                       rotated frame (a coordinate transform).
  * angles in radians; angular velocity in rad/s, in the BODY frame.
"""
import numpy as np


def normalize(q):
    """Scale q to unit norm. Call this after every integration step."""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n == 0.0:
        raise ValueError("zero quaternion has no orientation")
    return q / n


def conjugate(q):
    """q* = (q0, -q_vec).  For a unit quaternion, q* is also the inverse q^-1."""
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def multiply(p, q):
    """Hamilton product  p (x) q  (scalar-first), via the left matrix [P(p)].

    From your notes (Yang 3.4.5):  p (x) q = [P(p)] q.
    Composes rotations. Does NOT commute in general.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p0, p1, p2, p3 = p
    P = np.array([
        [p0, -p1, -p2, -p3],
        [p1,  p0, -p3,  p2],
        [p2,  p3,  p0, -p1],
        [p3, -p2,  p1,  p0],
    ])
    return P @ q


def from_axis_angle(e, alpha):
    """Euler axis/angle -> quaternion  (S&J 3.90, the half-angle map).

        q = [cos(alpha/2),  e_hat * sin(alpha/2)]
    """
    e = np.asarray(e, dtype=float)
    e = e / np.linalg.norm(e)
    h = 0.5 * alpha
    return np.array([np.cos(h), *(e * np.sin(h))])


def to_dcm(q):
    """Quaternion -> DCM  (S&J 3.92).  Passive:  v_target = to_dcm(q) @ v_source."""
    q0, q1, q2, q3 = normalize(q)
    return np.array([
        [q0*q0 + q1*q1 - q2*q2 - q3*q3, 2*(q1*q2 + q0*q3),             2*(q1*q3 - q0*q2)],
        [2*(q1*q2 - q0*q3),             q0*q0 - q1*q1 + q2*q2 - q3*q3, 2*(q2*q3 + q0*q1)],
        [2*(q1*q3 + q0*q2),             2*(q2*q3 - q0*q1),             q0*q0 - q1*q1 - q2*q2 + q3*q3],
    ])


def apply(q, v):
    """Transform a vector's coordinates by q (PASSIVE); equals to_dcm(q) @ v.

    Computed with the sandwich product. For THIS passive convention the
    conjugate sits on the LEFT:
        v_target = q* (x) v_bar (x) q          (v_bar = (0, v))
    The ACTIVE rotation (physically spin the vector) is the swap  q (x) v_bar (x) q*.
    """
    q = normalize(q)
    v = np.asarray(v, dtype=float)
    vbar = np.array([0.0, v[0], v[1], v[2]])
    out = multiply(multiply(conjugate(q), vbar), q)
    return out[1:]


def q_dot(q, omega_body):
    """Attitude kinematics  q_dot = 1/2 q (x) omega_bar,  omega_bar = (0, omega_body).

    omega_body: angular velocity in the BODY frame (rad/s).
    Integrate this (then normalize) to propagate attitude instead of the DCM.
    """
    q = np.asarray(q, dtype=float)
    w = np.asarray(omega_body, dtype=float)
    wbar = np.array([0.0, w[0], w[1], w[2]])
    return 0.5 * multiply(q, wbar)


def from_dcm(C):
    """DCM -> quaternion via STANLEY'S METHOD  (S&J 3.94-3.95).   <-- YOUR TASK

    Naive extraction (S&J 3.93) divides by q0 and blows up at 180 deg. Stanley:

      Step 1 (3.94) - the four squared components:
          q0^2 = (1 + tr)/4
          q1^2 = (1 + 2*C[0,0] - tr)/4
          q2^2 = (1 + 2*C[1,1] - tr)/4
          q3^2 = (1 + 2*C[2,2] - tr)/4
      Step 2 - take sqrt of the LARGEST of those four (choose the + sign).
      Step 3 (3.95) - the other three from off-diagonals / (4 * chosen):
          q0*q1 = (C[1,2] - C[2,1]) / 4
          q0*q2 = (C[2,0] - C[0,2]) / 4
          q0*q3 = (C[0,1] - C[1,0]) / 4
          q2*q3 = (C[1,2] + C[2,1]) / 4
          q3*q1 = (C[2,0] + C[0,2]) / 4
          q1*q2 = (C[0,1] + C[1,0]) / 4
      (S&J indices are 1-based: C23 -> C[1,2] here.)

    Return a length-4 numpy array. Tests in test_quaternion.py are the oracle.
    """
    #Step 1
    tr = np.trace(C)
    sq = [(1 + tr)/4, (1 + 2*C[0,0] - tr)/4, (1 + 2*C[1,1] - tr)/4, (1 + 2*C[2,2] - tr)/4]
    #Step 2
    max_index = np.argmax(sq)
    #Step 3
    if max_index == 0:
        q_0 = np.sqrt(sq[0])
        q_1 = (C[1,2] - C[2,1])/(4*q_0)
        q_2 = (C[2,0] - C[0,2])/(4*q_0)
        q_3 = (C[0,1] - C[1,0])/(4*q_0)
        
    elif max_index == 1:
        q_1 = np.sqrt(sq[1])
        q_0 = (C[1,2] - C[2,1])/(4*q_1)
        q_2 = (C[0,1] + C[1,0])/(4*q_1)
        q_3 = (C[2,0] + C[0,2])/(4*q_1)
        
    elif max_index == 2:
        q_2 = np.sqrt(sq[2])
        q_0 = (C[2,0] - C[0,2])/(4*q_2)
        q_1 = (C[0,1] + C[1,0])/(4*q_2)
        q_3 = (C[1,2] + C[2,1])/(4*q_2)
    elif max_index == 3:
        q_3 = np.sqrt(sq[3])
        q_0 = (C[0,1] - C[1,0])/(4*q_3)
        q_1 = (C[2,0] + C[0,2])/(4*q_3)
        q_2 = (C[1,2] + C[2,1])/(4*q_3)
    
    q = [q_0, q_1, q_2, q_3]
    return np.array(q, dtype=float)
    # raise NotImplementedError("implement Stanley's method (S&J 3.94-3.95)")
