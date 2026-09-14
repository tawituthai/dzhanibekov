r"""Orbit propagation and the LVLH reference frame  (node F5: position at time t).

Closes the "position" half of the F5 gate ("produce position, sun vector, IGRF
field and eclipse at time t"). The environment models (environment.py) already
give sun/field/eclipse at a position; this module supplies that position over
time, and builds the LVLH frame that attitude is measured against -- the
time-varying generalization of the [LN] you hand-derived at the ascending node
in F1.

The conversion elements -> r,v is S&J Eqs. 11.51-11.59:
    (11.53) Kepler:      M = E - e sin E          (solve for E)
    (11.55) orbit-frame position from E
    (11.56) orbit-frame velocity from E
    (11.59) [NO] = 3-1-3 Euler DCM from (Omega, i, omega)   -- your F1 machinery
    (11.57/58) r,v in ECI  =  [NO] @ (orbit-frame r,v)

For a two-body orbit the five shape/orientation elements are constant; only the
mean anomaly advances, M(t) = M0 + n t, n = sqrt(mu/a^3). So propagation is:
advance M, re-solve Kepler, re-convert. Units: km, km/s, radians, seconds.
"""
import numpy as np

MU_EARTH = 398600.4418        # km^3/s^2  (match S&J's constant when you validate)


def mean_motion(a, mu=MU_EARTH):
    """n = sqrt(mu / a^3), rad/s."""
    return np.sqrt(mu / a**3)


def solve_kepler(M, e, tol=1e-12, max_iter=100):
    """Solve Kepler's equation  M = E - e sin E  for the eccentric anomaly E.  <-- YOUR TASK

    Newton's method:  E_{k+1} = E_k - (E_k - e sin E_k - M) / (1 - e cos E_k)
    A good starting guess is E = M. Iterate until |E - e sin E - M| < tol
    (or max_iter reached). Return E in radians.

    (This is the one genuinely non-trivial computation in the node -- everything
    else is plug-and-chug. It vanishes for e=0, where E = M.)
    """
    # M and E need to be in radians
    E = M
    for i in range(max_iter):
        f = E - e*np.sin(E) - M
        fp = 1 - e*np.cos(E)
        E_next = E - f/fp
        if abs(E_next - E) < tol:
            return E_next
        E = E_next
    return E

def perifocal_to_eci(Omega, i, omega):
    """The [NO] DCM (S&J 11.59): 3-1-3 Euler rotation from the orbit (perifocal) <-- YOUR TASK
    frame to ECI, built from RAAN (Omega), inclination (i), argument of perigee (omega).

    This is exactly the F1 Euler-chain machinery. One correct form:

      [NO] = [[cw*cO - sw*ci*sO,  -sw*cO - cw*ci*sO,   si*sO],
              [cw*sO + sw*ci*cO,  -sw*sO + cw*ci*cO,  -si*cO],
              [sw*si,              cw*si,              ci   ]]

      with cO=cos(Omega), sO=sin(Omega), ci=cos(i), si=sin(i), cw=cos(omega), sw=sin(omega).

    Return a (3,3) numpy array.
    """
    sO = np.sin(Omega)
    si = np.sin(i)
    sw = np.sin(omega)
    cO = np.cos(Omega)
    ci = np.cos(i)
    cw = np.cos(omega)
    
    return np.array([[cw*cO - sw*ci*sO,  -sw*cO - cw*ci*sO,   si*sO],
                     [cw*sO + sw*ci*cO,  -sw*sO + cw*ci*cO,  -si*cO],
                     [sw*si           ,   cw*si           ,   ci   ]])


def elements_to_rv(a, e, i, Omega, omega, M, mu=MU_EARTH):
    """Classical elements -> (r, v) in ECI. S&J 11.53-11.59.

    Uses solve_kepler and perifocal_to_eci (your two functions above).
    Returns (r, v), each a (3,) numpy array in km and km/s.
    """
    E = solve_kepler(M, e)
    b = a * np.sqrt(1.0 - e**2)
    n = mean_motion(a, mu)
    r_orbit = np.array([a * (np.cos(E) - e), b * np.sin(E), 0.0])                  # 11.55
    v_orbit = np.array([-a * np.sin(E), b * np.cos(E), 0.0]) * (n / (1 - e * np.cos(E)))  # 11.56
    NO = perifocal_to_eci(Omega, i, omega)                                         # 11.59
    return NO @ r_orbit, NO @ v_orbit                                              # 11.57/58


def propagate_orbit(a, e, i, Omega, omega, M0, t_end, dt, mu=MU_EARTH):
    """Two-body propagation: advance mean anomaly, re-convert at each step.

    Returns (t_hist, r_hist, v_hist) of shapes (N+1,), (N+1,3), (N+1,3).
    """
    n = mean_motion(a, mu)
    n_steps = int(round(t_end / dt))
    t_hist = np.zeros(n_steps + 1)
    r_hist = np.zeros((n_steps + 1, 3))
    v_hist = np.zeros((n_steps + 1, 3))
    for k in range(n_steps + 1):
        t = k * dt
        M = M0 + n * t
        r, v = elements_to_rv(a, e, i, Omega, omega, M, mu)
        t_hist[k] = t
        r_hist[k] = r
        v_hist[k] = v
    return t_hist, r_hist, v_hist


def lvlh_frame(r, v):
    """Build the LVLH (local-vertical/local-horizontal) DCM [LN] from r, v.   <-- YOUR TASK

    This is the time-varying generalization of the F1 ascending-node frame.
    Rows of [LN] are the LVLH axes expressed in ECI; [LN] @ v_eci gives a
    vector's LVLH coordinates.

    Common convention (o1 along-ish track, o3 nadir):
        o3 = -r / |r|                    (nadir: points down, toward Earth center)
        o2 = -(r x v) / |r x v|          (negative orbit normal)
        o1 =  o2 x o3                    (completes the right-handed triad,
                                          roughly along the velocity direction)
    Stack o1, o2, o3 as ROWS.

    Return a (3,3) numpy array. (Watch the signs -- nadir is -r, and the orbit
    normal sign is a convention choice; this one gives a proper right-handed frame.)
    """
    o3 = -r/np.linalg.norm(r)
    o2 = -np.cross(r,v)/np.linalg.norm(np.cross(r,v))
    o1 = np.cross(o2,o3)
    return np.array([o1, o2, o3])
