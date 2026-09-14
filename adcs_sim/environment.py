"""Environment models and sensor measurement models  (vertical slice: F5 + sensors).

Two layers:
  1. ENVIRONMENT (truth) -- reference vectors in the ECI frame as functions of
     position/time: magnetic field (tilted dipole), Sun direction, eclipse flag.
     These are deliberately low-fidelity (n=1 IGRF dipole, low-precision Sun);
     good enough to detumble against and to feed determination. Upgrade paths
     noted inline (full IGRF n=13, precise ephemeris).

  2. SENSORS (measurement) -- given the TRUE attitude q and a reference vector
     in ECI, produce the vector as a body-frame sensor would see it:
         measurement_body = apply(q, ref_eci) + noise
     This is the truth-side sensor model that TRIAD/QUEST will consume.

Convention: apply(q, v) transforms an ECI vector into the body frame
(the passive convention pinned in quaternion.py).
"""
import numpy as np

from adcs_sim.quaternion import apply, normalize

# --- constants ---
R_EARTH = 6371.2                       # km, IGRF reference radius `a` in Eq. (7.1)
OBLIQUITY = np.radians(23.4393)        # ecliptic tilt epsilon

# IGRF n=1 (dipole) Gauss coefficients, nT, ~2020 epoch.  Full IGRF sums to n=13;
# this n=1 truncation IS the classic "tilted dipole" -- ~10-20% accurate, plenty
# for a magnetometer sim and B-dot control.
_G10, _G11, _H11 = -29404.8, -1450.9, 4652.5


# ==========================================================================
# ENVIRONMENT (truth reference vectors in ECI)
# ==========================================================================
def magnetic_field_eci(r_eci):
    """Tilted-dipole geomagnetic field at position r_eci (km), returned in nT.

    Centered-dipole model (IGRF n=1):
        B(r) = (a^3 B0 / r^3) [ 3 (m_hat . r_hat) r_hat - m_hat ]
    with dipole strength B0 = sqrt(g10^2 + g11^2 + h11^2) and axis m_hat built
    from the three n=1 coefficients. (Dipole tilt ~11 deg from the spin axis.)
    """
    r_eci = np.asarray(r_eci, float)
    r = np.linalg.norm(r_eci)
    r_hat = r_eci / r
    B0 = np.sqrt(_G10**2 + _G11**2 + _H11**2)
    m_hat = -np.array([_G11, _H11, _G10]) / B0        # dipole axis direction
    return (R_EARTH**3 * B0 / r**3) * (3.0*np.dot(m_hat, r_hat)*r_hat - m_hat)


def sun_vector_eci(days_since_equinox):
    """Low-precision unit Sun direction in ECI, as a function of days since the
    vernal equinox. The Sun tracks the ecliptic (~360/365.25 deg per day),
    tilted from the equator by the obliquity `OBLIQUITY`.
    Good to ~1 deg -- fine for a sun-sensor sim; swap for a real ephemeris later.
    """
    lam = 2*np.pi * (days_since_equinox / 365.25)      # ecliptic longitude
    eps = OBLIQUITY
    return np.array([np.cos(lam),
                     np.sin(lam)*np.cos(eps),
                     np.sin(lam)*np.sin(eps)])


def in_eclipse(r_eci, sun_hat):
    """True if the spacecraft is in Earth's cylindrical (umbral) shadow.

    Geometry: behind Earth relative to the Sun (r . sun_hat < 0) AND the
    perpendicular distance from the Earth-Sun line is less than one Earth radius.
    """
    r_eci = np.asarray(r_eci, float)
    sun_hat = normalize(sun_hat)
    along = np.dot(r_eci, sun_hat)
    if along >= 0:
        return False                                   # sunward side, always lit
    perp = np.linalg.norm(r_eci - along*sun_hat)
    return perp < R_EARTH


# ==========================================================================
# SENSORS (body-frame measurements)   <-- YOUR TASK
# ==========================================================================
def sun_sensor(q_true, sun_hat_eci, sigma_deg=1.0, rng=None):
    """Simulate a sun-sensor reading: the Sun direction in the BODY frame.  <-- YOUR TASK

    Steps:
      1. rotate the ECI Sun direction into the body frame with apply(q_true, .)
      2. add Gaussian noise of std sigma_deg DEGREES (small-angle: add a noise
         vector of magnitude ~ radians(sigma_deg) then renormalize to a unit vector)
      3. return a UNIT vector (a direction, so it must stay unit length)

    Use rng (a np.random.Generator) if given, else np.random.default_rng().
    Return a (3,) unit numpy array.
    """
    if rng is None:
        rng = np.random.default_rng()
    v = apply(q_true, sun_hat_eci)
    if sigma_deg > 0:
        v = v + rng.normal(0.0, np.radians(sigma_deg), size=3)
        
    return v / np.linalg.norm(v)


def magnetometer(q_true, B_eci, sigma_nT=200.0, bias_nT=None, rng=None):
    """Simulate a 3-axis magnetometer: the field in the BODY frame, in nT.  <-- YOUR TASK

    Unlike the sun sensor, a magnetometer measures MAGNITUDE too -- do NOT
    normalize. Model:
      1. rotate B_eci into the body frame with apply(q_true, .)
      2. add a constant bias vector bias_nT (default zeros) -- the systematic
         error ACS Ch.4 distinguishes from random noise
      3. add Gaussian white noise of std sigma_nT per axis
    Return a (3,) numpy array in nT (NOT unit length).
    """
    if rng is None:
        rng = np.random.default_rng()
    b = apply(q_true, B_eci)
    
    if sigma_nT > 0:
        b = b + rng.normal(0.0, sigma_nT, size=3)
    
    if bias_nT is not None:
        b = b + bias_nT
        
    return b
