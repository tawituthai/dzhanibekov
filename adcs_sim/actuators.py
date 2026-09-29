r"""Attitude actuators  (node A2: magnetorquers).

A magnetorquer (MTB, torque rod, air coil) is a coil that becomes a magnetic
dipole m when current flows. In the Earth's field B it feels a torque trying
to align m with B -- and since the rod is bolted to the spacecraft, the
spacecraft turns instead:

    tau = m x B                                   ACS without an Attitude Eq. 5.2
                                                  Junaid (2015) Eq. 3.8
    m   = mu_rod * N * I * A                      Junaid Eq. 3.9  (ACS Eq. 5.1: mu_rod = 1)

UNITS -- the classic trap. tau [N m] = m [A m^2] x B [T]. The field model in
environment.py returns nT, so B must be multiplied by 1e-9 before the cross
product. (ACS Eq. 5.2 labels B in gauss; 1 G = 1e-4 T. Use tesla.)

THE ONE PROPERTY THAT DEFINES MAGNETIC CONTROL: tau = m x B is ALWAYS
perpendicular to B. Whatever you command, you cannot make torque along the
local field direction -- one of three axes is missing at every instant. That
is what "underactuated" means (ACS Sec. 5.2, p. 108: "the direction of the
applied torque must always be perpendicular to the direction of the Earth's
magnetic field at that time"). Full 3-axis authority only appears over an
orbit, as B rotates -- which is why magnetic control is slow.
"""
import numpy as np

NT_TO_T = 1e-9


# --------------------------------------------------------------------------
# YOUR TASKS
# --------------------------------------------------------------------------
def saturate_dipole(m_cmd, m_max):
    """Limit a commanded dipole to what the rods can produce.       <-- YOUR TASK

    Each rod has a maximum dipole m_max [A m^2] (from its current limit).
    If any component exceeds it, SCALE THE WHOLE VECTOR DOWN so the largest
    component equals m_max -- do not clip each axis separately.

    Why scale, not clip: clipping axes independently changes the DIRECTION
    of m, so the torque m x B points somewhere the controller never asked
    for. Scaling keeps the direction and only shortens it.

        example: m_cmd = [2.0, 1.0, 0.0], m_max = 0.2  ->  [0.2, 0.1, 0.0]
                 (clipping would give [0.2, 0.2, 0.0] -- a different direction)

    m_max may be a scalar (identical rods) or a (3,) array (per-axis limits);
    use the ratio |m_i| / m_max_i to find the worst axis. Return a (3,) array.
    """
    m_max = np.asarray(m_max, float)
    m_cmd = np.asarray(m_cmd, float)
    
    if (m_cmd >= m_max).any() :
        ratio = np.abs(np.asarray(m_cmd))/m_max
        m_cmd = m_cmd/np.max(ratio)
    
    return m_cmd
    # raise NotImplementedError("implement direction-preserving dipole saturation")


def magnetorquer_torque(m_cmd, B_body_nT, m_max):
    """Torque produced by the magnetorquers, body frame, N m.        <-- YOUR TASK

        1. saturate m_cmd with saturate_dipole
        2. convert B from nT to T   (multiply by NT_TO_T)
        3. tau = m x B              (np.cross, and mind the ORDER: m first)

    m_cmd     : (3,) commanded dipole, A m^2, body frame
    B_body_nT : (3,) magnetic field in the body frame, nT (what the
                magnetometer / environment model gives you)
    m_max     : scalar or (3,) rod limit, A m^2
    Return a (3,) array.
    """
    m_cmd = saturate_dipole(m_cmd, m_max)
    B_body_T = np.asarray(B_body_nT, float)*NT_TO_T
    
    return np.cross(m_cmd, B_body_T)
    # raise NotImplementedError("implement tau = m x B with saturation and units")


def required_dipole(torque_req, B_min_nT):
    """Size a torque rod: the dipole needed for a torque requirement.  <-- YOUR TASK

    The torque available scales with |B|, so size against the WEAKEST field
    in the orbit (Junaid Eq. 3.7):

        M_min = N_req / B_min

    torque_req : required torque magnitude, N m
    B_min_nT   : minimum field magnitude over the orbit, nT
    Return the minimum dipole, A m^2 (a scalar).
    """
    return torque_req/(B_min_nT*NT_TO_T)
    # raise NotImplementedError("implement the rod sizing equation")


# --------------------------------------------------------------------------
# given: rod physics and the standard way to ask for a torque
# --------------------------------------------------------------------------
def rod_amplification(length, diameter):
    """Magnetic amplification of a ferromagnetic cylindrical core.

        mu_rod = 1.66 (L / D)^1.5                      Junaid Eq. 3.10

    L/D is the rod's shape factor. An air coil has mu_rod = 1.
    """
    return 1.66 * (length / diameter) ** 1.5


def dipole_from_coil(n_turns, current, area, mu_rod=1.0):
    """m = mu_rod * N * I * A, in A m^2.   Junaid Eq. 3.9 (ACS Eq. 5.1 with mu_rod = 1)."""
    return mu_rod * n_turns * current * area


def cross_product_law(tau_desired, B_body_nT):
    """The dipole that best produces a desired torque (Stellenbosch EO thesis
    Eq. 6.2.10, derived in its Appendix C; Junaid Eq. 4.55):

        m = (B x tau_des) / |B|^2         (B in tesla)

    Choosing m perpendicular to B maximises torque per unit dipole. The torque
    you then GET is  m x B = tau_des - (tau_des . B_hat) B_hat : the desired
    torque with its component along B deleted. That deleted part is the price
    of underactuation, made explicit -- see test_cross_product_law_loses_the_component_along_B.

    (The thesis writes it with a control error e = -tau_des: m = (e x B)/|B|^2.
    Same thing.)
    """
    B = np.asarray(B_body_nT, float) * NT_TO_T
    return np.cross(B, np.asarray(tau_desired, float)) / np.dot(B, B)
