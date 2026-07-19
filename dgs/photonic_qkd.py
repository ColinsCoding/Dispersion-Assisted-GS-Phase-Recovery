"""BB84 quantum key distribution as actual optics: Jones calculus on
polarized light, no digital bit-flip shortcuts, no electronics.

Every operation here is a real passive optical element acting on a 2-vector
(Jones vector) describing the photon's polarization:

  * a bit is encoded as a LINEAR POLARIZATION ANGLE (a wave plate setting),
    not an abstract 0/1;
  * a "basis choice" is a PHYSICAL ROTATION of the measurement apparatus
    (rectilinear 0deg/90deg vs. diagonal 45deg/135deg);
  * a "measurement" is a POLARIZING BEAM SPLITTER projection, with detection
    probability given by Malus's law cos^2(angle), not a coin flip dressed up
    as physics;
  * Eve's interception is modeled the same way Bob's detection is -- she is
    just another polarizer + detector inline in the beam path, and her
    resend really does collapse the polarization state, which is what
    produces the 25% QBER, not an assumption.

dgs.path_integral_qkd.bb84_intercept_resend_qber models the same protocol at
the abstract bit/basis level; this module re-derives the identical 25% QBER
from the underlying Jones-vector optics, as a cross-check that the abstract
model wasn't hiding any unjustified shortcuts.
"""

import numpy as np

# the two BB84 basis angles, each with two orthogonal bit-encoding directions
RECTILINEAR_BASIS = 0.0          # bit 0 -> 0 deg, bit 1 -> 90 deg
DIAGONAL_BASIS = np.pi / 4        # bit 0 -> 45 deg, bit 1 -> 135 deg

BASIS_ANGLES = {0: RECTILINEAR_BASIS, 1: DIAGONAL_BASIS}


def jones_vector(angle):
    """The Jones vector for light linearly polarized at `angle` (radians) to
    the lab x-axis: [cos(angle), sin(angle)] -- a unit 2-vector, no electronics,
    just the direction the E-field oscillates in."""
    return np.array([np.cos(angle), np.sin(angle)])


def encode_bit(bit, basis):
    """Encode one classical bit as a polarization angle in the chosen basis:
    rectilinear (0): bit 0 -> 0deg, bit 1 -> 90deg.
    diagonal (1):    bit 0 -> 45deg, bit 1 -> 135deg.
    Returns the photon's Jones vector."""
    base_angle = BASIS_ANGLES[basis]
    angle = base_angle + (np.pi / 2 if bit == 1 else 0.0)
    return jones_vector(angle)


def waveplate_rotation(angle):
    """A real, passive optical element: a half-wave-plate-equivalent basis
    rotation, the 2x2 Jones rotation matrix R(angle) = [[cos,-sin],[sin,cos]].
    Rotating the measurement basis is physically just rotating a polarizer/
    wave-plate assembly by this angle -- no digital logic involved."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def detection_probability(state, basis):
    """Malus's law: the probability a polarizing beam splitter aligned with
    `basis` detects the photon in its "bit 1" output port, given incoming
    Jones vector `state`. P = |<e_1|R(-base_angle) state>|^2, i.e. project
    the (rotated-into-basis) state onto the basis's second axis and square --
    real optics, not a coin flip standing in for one."""
    base_angle = BASIS_ANGLES[basis]
    rotated = waveplate_rotation(-base_angle) @ state
    return float(rotated[1] ** 2)   # |component along the "1" axis|^2


def measure_polarization(state, basis, rng):
    """Simulate one photon hitting a polarizing-beam-splitter detector pair
    in the given basis: sample bit=1 with the Malus's-law probability from
    detection_probability, else bit=0. This single Bernoulli draw IS the
    photon detection event -- not a classical bit being looked up."""
    p1 = detection_probability(state, basis)
    return int(rng.random() < p1)


def bb84_optical_intercept_resend(n_bits=100_000, eavesdrop=True, seed=0):
    """The full BB84 protocol, run entirely in Jones-vector optics: Alice
    encodes random bits as polarized photons in a random basis; if
    `eavesdrop`, Eve intercepts each photon with her own random-basis
    polarizer/detector and RESENDS a freshly encoded photon at her measured
    bit and basis (a real physical collapse-and-resend, not a bit-flip
    shortcut); Bob measures with his own random-basis polarizer/detector.
    Returns the Monte Carlo QBER on the sifted (matching-basis) key, to be
    compared against the analytic 25% intercept-resend prediction."""
    rng = np.random.default_rng(seed)
    alice_bits = rng.integers(0, 2, n_bits)
    alice_bases = rng.integers(0, 2, n_bits)

    channel_bits, channel_bases = alice_bits, alice_bases

    if eavesdrop:
        eve_bases = rng.integers(0, 2, n_bits)
        eve_bits = np.empty(n_bits, dtype=int)
        for i in range(n_bits):
            photon = encode_bit(int(alice_bits[i]), int(alice_bases[i]))
            eve_bits[i] = measure_polarization(photon, int(eve_bases[i]), rng)
        # Eve resends a NEW photon, physically re-encoded at her own
        # measured bit and basis -- this is the actual collapse, not a label
        channel_bits, channel_bases = eve_bits, eve_bases

    bob_bases = rng.integers(0, 2, n_bits)
    bob_bits = np.empty(n_bits, dtype=int)
    for i in range(n_bits):
        photon = encode_bit(int(channel_bits[i]), int(channel_bases[i]))
        bob_bits[i] = measure_polarization(photon, int(bob_bases[i]), rng)

    sifted = alice_bases == bob_bases
    n_sifted = int(np.sum(sifted))
    errors = int(np.sum(alice_bits[sifted] != bob_bits[sifted]))
    qber_mc = errors / n_sifted if n_sifted else float("nan")

    return {
        "n_bits": n_bits, "n_sifted": n_sifted,
        "qber_mc": qber_mc, "qber_analytic": 0.25 if eavesdrop else 0.0,
    }


def malus_law_curve(angle_offsets):
    """Detection probability vs. angle mismatch between the photon's
    polarization and the analyzer's basis -- Malus's law itself,
    P = cos^2(angle_offset), the single physical fact the entire BB84
    security proof rests on."""
    return np.cos(np.asarray(angle_offsets)) ** 2
