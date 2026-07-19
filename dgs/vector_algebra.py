"""
Vector Algebra Problem Sets
============================
Griffiths Electrodynamics Ch.1 + AP Physics C + Photonics

OPERATIONS (in order):
  1. Addition of two vectors:        A + B = (Ax+Bx, Ay+By, Az+Bz)
  2. Multiplication by a scalar:     c * A = (c*Ax, c*Ay, c*Az)
  3. Dot product of two vectors:     A . B = Ax*Bx + Ay*By + Az*Bz
  4. Cross product of two vectors:   A x B = det | i  j  k  |
                                                  | Ax Ay Az |
                                                  | Bx By Bz |

THREE RULES (Griffiths §1.1.3 triple products):
  Rule (i):   A . (B x C) = B . (C x A) = C . (A x B)   [scalar triple product]
  Rule (ii):  A x (B x C) = B(A.C) - C(A.B)              [BAC-CAB rule]
  Rule (iii): (A x B) x C = B(A.C) - A(B.C)              [note sign differs from (ii)]

PRODUCT RULES FOR DERIVATIVES (Griffiths §1.2.4):
  Rule (i):   grad(f*g)  = f*grad(g) + g*grad(f)
  Rule (ii):  grad(A.B)  = A x curl(B) + B x curl(A) + (A.grad)B + (B.grad)A
  Rule (iii): div(f*A)   = f*div(A) + A.grad(f)
  Rule (iv):  div(A x B) = B.curl(A) - A.curl(B)
  Rule (v):   curl(f*A)  = f*curl(A) - A x grad(f)
  Rule (vi):  curl(A x B)= (B.grad)A - (A.grad)B + A*div(B) - B*div(A)

FUNDAMENTAL THEOREM:
  Divergence theorem:  closed_surface_integral(A.da) = volume_integral(div(A) dV)
  Stokes theorem:      closed_line_integral(A.dl)   = surface_integral(curl(A).da)

PHOTONICS CONNECTION:
  E-field is a vector: E = Ex*x_hat + Ey*y_hat + Ez*z_hat
  Poynting vector: S = (1/mu0) * E x B       [power flow direction = cross product]
  EM wave: k . E = 0, k . B = 0              [transverse: dot product = 0]
  Intensity: I = |E|^2 = E . E               [dot product with itself]
  Phase gradient: grad(phi) = k_vector       [grad of scalar phase -> k-vector]

PRESSURE (KINETIC THEORY / STATISTICS):
  P = (1/3) * n * m * <v^2>     [kinetic theory pressure]
  P = n * kB * T                [ideal gas]
  Radiation pressure: P_rad = I/c  (for perfect absorber), 2I/c (perfect reflector)
  In photonics: photon momentum p = hbar*k, pressure P = dp/dt/Area
"""
import math
import numpy as np


# ============================================================
# Vector class: clean 3D vector with all operations
# ============================================================

class Vec3:
    """
    3D vector with all standard operations.

    >>> A = Vec3(1, 2, 3)
    >>> B = Vec3(4, 5, 6)
    >>> A + B
    Vec3(5, 7, 9)
    >>> A.dot(B)
    32
    >>> A.cross(B)
    Vec3(-3, 6, -3)
    """
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    # ── (1) Addition ──────────────────────────────────────────
    def __add__(self, other):
        """A + B = (Ax+Bx, Ay+By, Az+Bz)"""
        return Vec3(self.x+other.x, self.y+other.y, self.z+other.z)

    def __sub__(self, other):
        """A - B = (Ax-Bx, Ay-By, Az-Bz)"""
        return Vec3(self.x-other.x, self.y-other.y, self.z-other.z)

    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)

    # ── (2) Scalar multiplication ──────────────────────────────
    def __mul__(self, scalar):
        """c * A = (c*Ax, c*Ay, c*Az)"""
        return Vec3(scalar*self.x, scalar*self.y, scalar*self.z)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        return Vec3(self.x/scalar, self.y/scalar, self.z/scalar)

    # ── (3) Dot product ───────────────────────────────────────
    def dot(self, other):
        """A . B = Ax*Bx + Ay*By + Az*Bz"""
        return self.x*other.x + self.y*other.y + self.z*other.z

    # ── (4) Cross product ─────────────────────────────────────
    def cross(self, other):
        """
        A x B = | i   j   k  |
                | Ax  Ay  Az |
                | Bx  By  Bz |
        = (Ay*Bz - Az*By)*i - (Ax*Bz - Az*Bx)*j + (Ax*By - Ay*Bx)*k
        """
        return Vec3(
            self.y*other.z - self.z*other.y,
            self.z*other.x - self.x*other.z,
            self.x*other.y - self.y*other.x
        )

    # ── Magnitude / unit vector ───────────────────────────────
    def mag(self):
        """|A| = sqrt(Ax^2 + Ay^2 + Az^2)"""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def unit(self):
        """A_hat = A / |A|"""
        m = self.mag()
        if m < 1e-30:
            raise ValueError("Cannot normalize zero vector")
        return self / m

    def __eq__(self, other, tol=1e-10):
        return (abs(self.x-other.x) < tol and
                abs(self.y-other.y) < tol and
                abs(self.z-other.z) < tol)

    def __repr__(self):
        return f"Vec3({self.x:.4g}, {self.y:.4g}, {self.z:.4g})"

    def to_list(self):
        return [self.x, self.y, self.z]

    @classmethod
    def from_list(cls, lst):
        return cls(*lst)

    def to_numpy(self):
        return np.array([self.x, self.y, self.z])


# Standard basis vectors
X_HAT = Vec3(1, 0, 0)
Y_HAT = Vec3(0, 1, 0)
Z_HAT = Vec3(0, 0, 1)


# ============================================================
# Problem Set 1: Basic Operations (written out)
# ============================================================

def problem_set_basic_operations(
    A=None, B=None, scalar_c=3.0
):
    """
    PROBLEM SET: Vector Algebra Basics
    ===================================

    GIVEN: A = Ax*x + Ay*y + Az*z
           B = Bx*x + By*y + Bz*z
           c = scalar

    PROBLEM 1: ADDITION
    -------------------
    A + B = (Ax + Bx)*x + (Ay + By)*y + (Az + Bz)*z

    Commutative: A + B = B + A
    Associative: (A + B) + C = A + (B + C)
    Identity:    A + 0 = A
    Inverse:     A + (-A) = 0

    PROBLEM 2: SCALAR MULTIPLICATION
    ---------------------------------
    c*A = (c*Ax)*x + (c*Ay)*y + (c*Az)*z

    Distributive over vectors:  c*(A+B) = c*A + c*B
    Distributive over scalars:  (c+d)*A = c*A + d*A
    Associative:                c*(d*A) = (c*d)*A
    Unit:                       1*A = A

    PROBLEM 3: DOT PRODUCT
    ----------------------
    A . B = Ax*Bx + Ay*By + Az*Bz   [component form]
          = |A||B|*cos(theta)         [geometric form]

    Properties:
      Commutative:  A.B = B.A
      Distributive: A.(B+C) = A.B + A.C
      Scalar out:   A.(c*B) = c*(A.B)
      Self:         A.A = |A|^2
      Orthogonal:   A.B = 0  iff A _|_ B  (theta = 90 deg)
      Parallel:     A.B = |A||B|  iff A || B  (theta = 0)

    PROBLEM 4: CROSS PRODUCT
    -------------------------
    A x B = | x   y   z  |    (determinant form)
            | Ax  Ay  Az |
            | Bx  By  Bz |

          = (Ay*Bz - Az*By)*x
          - (Ax*Bz - Az*Bx)*y
          + (Ax*By - Ay*Bx)*z

    Properties:
      Anti-commutative: A x B = -(B x A)
      Distributive:     A x (B+C) = A x B + A x C
      Scalar out:       A x (c*B) = c*(A x B)
      Self:             A x A = 0
      Magnitude:        |A x B| = |A||B|*sin(theta)
      Direction:        right-hand rule (fingers A -> B, thumb = A x B)
      Perpendicular:    A x B _|_ A  and  A x B _|_ B
      Parallel:         A x B = 0  iff A || B  (theta = 0 or 180)

    UNIT VECTOR CROSS PRODUCTS:
      x x y = z    y x z = x    z x x = y
      y x x = -z   z x y = -x   x x z = -y
      x x x = 0    y x y = 0    z x z = 0
    """
    if A is None:
        A = Vec3(1, 2, 3)
    if B is None:
        B = Vec3(4, 5, 6)
    c = scalar_c

    # ── (1) Addition ──────────────────────────────────────────
    ApB = A + B
    BpA = B + A
    commutative_add = (ApB == BpA)

    # ── (2) Scalar multiplication ──────────────────────────────
    cA = c * A
    cB = c * B
    cApB = c * (A + B)
    cApcB = cA + cB
    distributive_scalar = (cApB == cApcB)

    # ── (3) Dot product ───────────────────────────────────────
    AdotB = A.dot(B)
    BdotA = B.dot(A)
    commutative_dot = abs(AdotB - BdotA) < 1e-10
    AdotA = A.dot(A)
    mag_A_sq = A.mag()**2
    self_dot_check = abs(AdotA - mag_A_sq) < 1e-10
    # Angle
    cos_theta = AdotB / (A.mag() * B.mag()) if A.mag()*B.mag() > 1e-30 else 0.0
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta_deg = math.degrees(math.acos(cos_theta))

    # ── (4) Cross product ─────────────────────────────────────
    AxB = A.cross(B)
    BxA = B.cross(A)
    anti_commutative = (AxB == -BxA)
    # A x A = 0
    AxA = A.cross(A)
    self_cross_zero = AxA == Vec3(0, 0, 0)
    # |A x B| = |A||B|sin(theta)
    sin_theta = math.sqrt(max(1 - cos_theta**2, 0))
    mag_AxB_formula = A.mag() * B.mag() * sin_theta
    mag_AxB_direct = AxB.mag()
    cross_mag_check = abs(mag_AxB_formula - mag_AxB_direct) < 1e-8
    # A x B _|_ A  and  _|_ B
    perp_to_A = abs(AxB.dot(A)) < 1e-8
    perp_to_B = abs(AxB.dot(B)) < 1e-8

    # ── Unit vector cross products ─────────────────────────────
    unit_cross = {
        'x x y': X_HAT.cross(Y_HAT).to_list(),   # should be [0,0,1] = z
        'y x z': Y_HAT.cross(Z_HAT).to_list(),   # should be [1,0,0] = x
        'z x x': Z_HAT.cross(X_HAT).to_list(),   # should be [0,1,0] = y
        'y x x': Y_HAT.cross(X_HAT).to_list(),   # should be [0,0,-1] = -z
        'x x x': X_HAT.cross(X_HAT).to_list(),   # should be [0,0,0]
    }

    return {
        'A': A.to_list(), 'B': B.to_list(), 'c': c,
        'addition': {
            'A_plus_B': ApB.to_list(),
            'B_plus_A': BpA.to_list(),
            'commutative': bool(commutative_add),
            'formula': 'A+B = (Ax+Bx, Ay+By, Az+Bz)',
        },
        'scalar_mult': {
            'c_times_A': cA.to_list(),
            'c_times_B': cB.to_list(),
            'c_times_(A+B)': cApB.to_list(),
            'c*A + c*B':    cApcB.to_list(),
            'distributive': bool(distributive_scalar),
            'formula': 'c*A = (c*Ax, c*Ay, c*Az)',
        },
        'dot_product': {
            'A_dot_B': float(AdotB),
            'B_dot_A': float(BdotA),
            'commutative': bool(commutative_dot),
            'A_dot_A': float(AdotA),
            '|A|^2': float(mag_A_sq),
            'self_dot_check': bool(self_dot_check),
            'theta_deg': float(theta_deg),
            'formula_component': 'A.B = Ax*Bx + Ay*By + Az*Bz',
            'formula_geometric': 'A.B = |A||B|*cos(theta)',
        },
        'cross_product': {
            'A_cross_B': AxB.to_list(),
            'B_cross_A': BxA.to_list(),
            'anti_commutative': bool(anti_commutative),
            'A_cross_A': AxA.to_list(),
            'self_cross_zero': bool(self_cross_zero),
            '|A_cross_B|_formula': float(mag_AxB_formula),
            '|A_cross_B|_direct': float(mag_AxB_direct),
            'cross_mag_check': bool(cross_mag_check),
            'perpendicular_to_A': bool(perp_to_A),
            'perpendicular_to_B': bool(perp_to_B),
            'unit_cross_products': unit_cross,
            'formula_determinant': (
                'A x B = (Ay*Bz-Az*By)*x - (Ax*Bz-Az*Bx)*y + (Ax*By-Ay*Bx)*z'
            ),
        },
    }


# ============================================================
# Problem Set 2: Triple Product Rules (i), (ii), (iii)
# ============================================================

def triple_product_rules(A=None, B=None, C=None):
    """
    TRIPLE PRODUCT RULES (Griffiths §1.1.3)
    =========================================

    RULE (i): SCALAR TRIPLE PRODUCT
    --------------------------------
    A . (B x C) = B . (C x A) = C . (A x B)

    This is the VOLUME of the parallelepiped spanned by A, B, C.

    Proof sketch:
      A.(B x C) = Ax(By*Cz-Bz*Cy) + Ay(Bz*Cx-Bx*Cz) + Az(Bx*Cy-By*Cx)
               = det | Ax Ay Az |
                     | Bx By Bz |
                     | Cx Cy Cz |
    Cyclic permutation of rows doesn't change the determinant.
    => A.(B x C) = B.(C x A) = C.(A x B)   (even permutations)
       A.(B x C) = -A.(C x B) = -(odd permutation)

    RULE (ii): VECTOR TRIPLE PRODUCT (BAC-CAB)
    -------------------------------------------
    A x (B x C) = B(A.C) - C(A.B)

    Mnemonic: BAC - CAB
      "Back - Cab"
      B times (A dot C)  minus  C times (A dot B)

    Note: The result is IN THE PLANE of B and C (not along A).
    This is a VECTOR, not a scalar.

    Proof: expand in components, verify each x,y,z component.
    Key insight: A x (B x C) is the component of A perpendicular
    to (B x C), which lives in the B-C plane.

    RULE (iii): OTHER FORM
    -----------------------
    (A x B) x C = -C x (A x B) = -(A x B) x (-C)
               = B(A.C) - A(B.C)    [apply rule ii with different grouping]

    Note: (A x B) x C  =/=  A x (B x C)  in general!
    Cross product is NOT associative.
    Difference:
      A x (B x C) = B(A.C) - C(A.B)   [rule ii]
      (A x B) x C = B(A.C) - A(B.C)   [rule iii]
    The last terms differ: C(A.B) vs A(B.C)

    PRODUCT RULES FOR VECTOR DERIVATIVES (Griffiths §1.2.4):
    ---------------------------------------------------------
    Rule (i):   grad(fg) = f*grad(g) + g*grad(f)
    Rule (ii):  div(fA)  = f*div(A) + A.grad(f)
    Rule (iii): curl(fA) = f*curl(A) - A x grad(f)
    """
    if A is None:
        A = Vec3(1, 2, 3)
    if B is None:
        B = Vec3(0, 1, 4)
    if C is None:
        C = Vec3(-1, 3, 2)

    # ── Rule (i): Scalar triple product ─────────────────────
    BxC = B.cross(C)
    CxA = C.cross(A)
    AxB = A.cross(B)

    stp_ABC = A.dot(BxC)    # A.(B x C)
    stp_BCA = B.dot(CxA)    # B.(C x A)
    stp_CAB = C.dot(AxB)    # C.(A x B)

    rule_i_AB = abs(stp_ABC - stp_BCA) < 1e-8
    rule_i_BC = abs(stp_BCA - stp_CAB) < 1e-8
    rule_i_all = rule_i_AB and rule_i_BC

    # Verify via determinant
    M = np.array([A.to_numpy(), B.to_numpy(), C.to_numpy()])
    det_val = float(np.linalg.det(M))
    det_agrees = abs(stp_ABC - det_val) < 1e-8

    # Anti-symmetry: A.(B x C) = -A.(C x B)
    CxB = C.cross(B)
    stp_ACB = A.dot(CxB)   # A.(C x B) should be -stp_ABC
    anti_sym = abs(stp_ABC + stp_ACB) < 1e-8

    # ── Rule (ii): BAC-CAB ───────────────────────────────────
    # A x (B x C) = B(A.C) - C(A.B)
    lhs_ii = A.cross(BxC)             # left-hand side
    AdotC = A.dot(C); AdotB = A.dot(B)
    rhs_ii = B*AdotC - C*AdotB        # BAC - CAB
    rule_ii_ok = (lhs_ii == rhs_ii)

    # Verify component by component
    diff_ii = (lhs_ii - rhs_ii).mag()

    # ── Rule (iii): (A x B) x C = B(A.C) - A(B.C) ──────────
    lhs_iii = AxB.cross(C)
    BdotC = B.dot(C)
    rhs_iii = B*AdotC - A*BdotC
    rule_iii_ok = (lhs_iii == rhs_iii)

    # NOT equal: rule ii vs rule iii
    ii_vs_iii_different = not (lhs_ii == lhs_iii)

    # ── Physical examples ────────────────────────────────────
    # Poynting vector S = (1/mu0) * E x B
    mu0 = 4*math.pi*1e-7
    E_field = Vec3(1, 0, 0)   # E in x direction
    B_field = Vec3(0, 1, 0)   # B in y direction
    S_dir   = E_field.cross(B_field)   # should be z direction
    S_unit  = S_dir.unit()

    # Angular momentum L = r x p  (classical)
    r_vec = Vec3(0, 0, 1)   # r along z
    p_vec = Vec3(1, 0, 0)   # p along x
    L_ang = r_vec.cross(p_vec)   # L = y direction

    # Magnetic force F = q*(v x B)
    v_vec = Vec3(1, 0, 0)   # velocity in x
    B_ext = Vec3(0, 0, 1)   # B in z
    F_mag = v_vec.cross(B_ext)   # F = v x B = -y direction

    return {
        'A': A.to_list(), 'B': B.to_list(), 'C': C.to_list(),

        'rule_i_scalar_triple': {
            'A.(B x C)': float(stp_ABC),
            'B.(C x A)': float(stp_BCA),
            'C.(A x B)': float(stp_CAB),
            'all_equal': bool(rule_i_all),
            'det_agrees': bool(det_agrees),
            'det_value': float(det_val),
            'anti_symmetry': bool(anti_sym),
            'physical': 'Volume of parallelepiped = |A.(B x C)|',
            'formula': (
                'A.(B x C) = B.(C x A) = C.(A x B) = '
                'det | Ax Ay Az | = parallelepiped volume'
                '    | Bx By Bz |'
                '    | Cx Cy Cz |'
            ),
        },

        'rule_ii_bac_cab': {
            'LHS: A x (B x C)': lhs_ii.to_list(),
            'RHS: B(A.C) - C(A.B)': rhs_ii.to_list(),
            'equal': bool(rule_ii_ok),
            'error': float(diff_ii),
            'A.C': float(AdotC),
            'A.B': float(AdotB),
            'formula': 'A x (B x C) = B*(A.C) - C*(A.B)    [BAC - CAB]',
            'mnemonic': '"Back Cab": B*(A dot C) minus C*(A dot B)',
            'physical': (
                'Result lives in B-C plane. '
                'Used to expand curl(curl(A)) = grad(div A) - laplacian(A)'
            ),
        },

        'rule_iii_axb_c': {
            'LHS: (A x B) x C': lhs_iii.to_list(),
            'RHS: B(A.C) - A(B.C)': rhs_iii.to_list(),
            'equal': bool(rule_iii_ok),
            'formula': '(A x B) x C = B*(A.C) - A*(B.C)',
            'not_assoc': bool(ii_vs_iii_different),
            'note': 'Cross product NOT associative: A x (B x C) != (A x B) x C in general',
            'difference': (
                'Rule ii: A x (B x C) = B(A.C) - C(A.B)  [last term has C, A.B]\n'
                'Rule iii:(A x B) x C = B(A.C) - A(B.C)  [last term has A, B.C]'
            ),
        },

        'physical_examples': {
            'Poynting_S = E x B': {
                'E': E_field.to_list(), 'B': B_field.to_list(),
                'S_direction': S_dir.to_list(),
                'S_unit_hat': S_unit.to_list(),
                'note': 'E in x, B in y -> S in z: wave propagates in z direction',
            },
            'angular_momentum_L = r x p': {
                'r': r_vec.to_list(), 'p': p_vec.to_list(),
                'L': L_ang.to_list(),
                'note': 'r in z, p in x -> L in -y direction',
            },
            'magnetic_force_F = qv x B': {
                'v': v_vec.to_list(), 'B': B_ext.to_list(),
                'F = v x B': F_mag.to_list(),
                'note': 'v in x, B in z -> F in -y (Lorentz force)',
            },
        },

        'vector_derivative_rules': {
            'rule_i':   'grad(f*g) = f*grad(g) + g*grad(f)',
            'rule_ii':  'div(f*A)  = f*div(A) + A.grad(f)',
            'rule_iii': 'curl(f*A) = f*curl(A) - A x grad(f)',
            'rule_iv':  'div(A x B) = B.curl(A) - A.curl(B)',
            'rule_v':   'grad(A.B) = A x curl(B) + B x curl(A) + (A.grad)B + (B.grad)A',
            'rule_vi':  'curl(A x B) = (B.grad)A - (A.grad)B + A*div(B) - B*div(A)',
        },

        'fundamental_theorems': {
            'Divergence': 'closed_surface(A.da) = volume(div(A) dV)',
            'Stokes':     'closed_line(A.dl)    = surface(curl(A).da)',
            'Gradient':   'integral_a^b(grad(T).dl) = T(b) - T(a)',
        },
    }


# ============================================================
# Problem Set 3: Pressure + Statistics + Kinetic Theory
# ============================================================

def pressure_statistics(T_K=300.0, n_density=2.69e25):
    """
    Pressure from Statistical Mechanics -> Radiation Pressure in Photonics

    KINETIC THEORY (AP Physics C / AP Chem / AP Stats connection):
      P = (1/3) * n * m * <v^2>    [kinetic theory]
        = n * kB * T               [ideal gas: same result via equipartition]
        = (2/3) * (N/V) * KE_avg  [pressure = energy density * 2/3]

      Each molecule: KE_avg = (3/2)*kB*T  (equipartition, 3 translational DoF)
      rms speed: v_rms = sqrt(3*kB*T/m)

    MAXWELL-BOLTZMANN SPEED DISTRIBUTION:
      f(v) = 4*pi * (m/(2*pi*kB*T))^(3/2) * v^2 * exp(-m*v^2/(2*kB*T))
      Mean speed:  <v>  = sqrt(8*kB*T/(pi*m))
      Most probable: vp = sqrt(2*kB*T/m)
      RMS speed:   vrms = sqrt(3*kB*T/m)
      Order: vp < <v> < vrms

    RADIATION PRESSURE (PHOTONICS):
      Photon momentum: p_photon = hbar*k = h/lambda = h*f/c
      Force = dp/dt = (rate of photon arrival) * p_photon
      P_rad = I/c         [perfect absorber: momentum transferred once]
      P_rad = 2*I/c       [perfect reflector: momentum transferred twice]

      For laser: I = P_optical / A_beam
      At 1 mW into 1 mm^2: P_rad = 1e-3 / (3e8 * 1e-6) = 3.3 nPa  (tiny!)
      BUT in optical tweezers: focused to 1 micron^2 -> P_rad = 3.3 mPa -> traps cells

    STATISTICS CONNECTION:
      Boltzmann factor:     P(state i) ~ exp(-E_i/kB*T)
      Partition function:   Z = sum_i exp(-E_i/kB*T)
      Thermal average:      <E> = -d/d(beta) ln(Z),  beta=1/(kB*T)
      Entropy:              S = -kB * sum_i P_i * ln(P_i)  [Shannon = Boltzmann entropy]
      Pressure from Z:      P = kB*T * d(ln Z)/dV

    PHOTONIC ANALOG:
      Photon gas (blackbody): P_rad = U/(3V) = (4*sigma/3c)*T^4
      This is how stars are supported by radiation pressure!
    """
    kB = 1.381e-23; h_P = 6.626e-34; c_l = 2.998e8; sigma_SB = 5.67e-8
    m_N2 = 28e-3 / 6.022e23   # nitrogen molecule mass [kg]
    m_air = 29e-3 / 6.022e23  # air molecule mass

    # Kinetic theory pressure (matches ideal gas)
    v_rms = math.sqrt(3*kB*T_K/m_air)
    P_kinetic = (1/3) * n_density * m_air * v_rms**2
    P_ideal   = n_density * kB * T_K

    # Maxwell-Boltzmann speeds (nitrogen)
    v_mp   = math.sqrt(2*kB*T_K/m_N2)       # most probable
    v_mean = math.sqrt(8*kB*T_K/(math.pi*m_N2))  # mean
    v_rms_N2 = math.sqrt(3*kB*T_K/m_N2)     # rms
    order_ok = v_mp < v_mean < v_rms_N2

    # Speed distribution (numerical)
    v_arr = np.linspace(0, 4*v_rms_N2, 500)
    fv = (4*math.pi * (m_N2/(2*math.pi*kB*T_K))**1.5 *
          v_arr**2 * np.exp(-m_N2*v_arr**2/(2*kB*T_K)))
    # Normalize check
    dv = v_arr[1]-v_arr[0]
    norm = float(np.trapezoid(fv, v_arr))

    # Radiation pressure
    I_laser_W_m2 = 1e-3 / 1e-6   # 1 mW into 1 mm^2 = 1e-6 m^2
    P_rad_absorb = I_laser_W_m2 / c_l
    P_rad_reflect = 2 * I_laser_W_m2 / c_l
    I_tweezers = 1e-3 / 1e-12    # 1 mW into 1 micron^2 = 1e-12 m^2
    P_rad_tweezers = I_tweezers / c_l

    # Blackbody radiation pressure (T=5778 K for Sun)
    T_sun = 5778.0
    U_density = 4*sigma_SB/c_l * T_sun**4   # [J/m^3]
    P_rad_sun = U_density / 3               # radiation pressure [Pa]

    # Boltzmann statistics (3-level system: E0=0, E1=hf, E2=2hf)
    f_laser = c_l/1550e-9    # 193 THz
    E_levels = [0, h_P*f_laser, 2*h_P*f_laser]
    beta = 1/(kB*T_K)
    Z = sum(math.exp(-beta*E) for E in E_levels)
    P_states = [math.exp(-beta*E)/Z for E in E_levels]
    E_avg = sum(P*E for P,E in zip(P_states, E_levels))
    S_entropy = -kB * sum(P*math.log(max(P,1e-300)) for P in P_states)

    return {
        'kinetic_theory': {
            'T_K': float(T_K),
            'n_density_per_m3': float(n_density),
            'v_rms_air_m_s': float(v_rms),
            'P_kinetic_Pa': float(P_kinetic),
            'P_ideal_Pa': float(P_ideal),
            'agreement': bool(abs(P_kinetic - P_ideal)/max(P_ideal,1e-30) < 0.01),
            'formulas': {
                'kinetic': 'P = (1/3)*n*m*v_rms^2',
                'ideal': 'P = n*kB*T',
                'v_rms': 'v_rms = sqrt(3*kB*T/m)',
            },
        },
        'maxwell_boltzmann': {
            'v_mp_m_s': float(v_mp),
            'v_mean_m_s': float(v_mean),
            'v_rms_m_s': float(v_rms_N2),
            'order_vp_vmean_vrms': bool(order_ok),
            'distribution_norm': float(norm),
            'order_formula': 'vp < v_mean < vrms: sqrt(2) < sqrt(8/pi) < sqrt(3) in units of sqrt(kT/m)',
        },
        'radiation_pressure': {
            'I_1mW_1mm2_W_m2': float(I_laser_W_m2),
            'P_rad_absorb_Pa': float(P_rad_absorb),
            'P_rad_reflect_Pa': float(P_rad_reflect),
            'P_rad_tweezers_Pa': float(P_rad_tweezers),
            'P_rad_sun_Pa': float(P_rad_sun),
            'formula_absorb': 'P_rad = I/c',
            'formula_reflect': 'P_rad = 2*I/c',
            'photon_momentum': 'p_photon = hbar*k = h/lambda = h*f/c',
            'application': 'Optical tweezers trap cells at ~mPa; stellar structure at MPa',
        },
        'boltzmann_statistics': {
            'E_levels_eV': [float(E/1.602e-19) for E in E_levels],
            'P_states': [float(p) for p in P_states],
            'partition_Z': float(Z),
            'E_avg_eV': float(E_avg/1.602e-19),
            'entropy_J_K': float(S_entropy),
            'formula': 'P(state i) = exp(-E_i/kT) / Z',
            'connection': 'Shannon entropy H = -sum p*log(p) = Boltzmann entropy S/kB',
        },
    }


# ============================================================
# Problem Set 4: Vector Calculus (Griffiths Ch.1 continuation)
# ============================================================

def vector_calculus_ops(
    n_pts=20,
    r_max=2.0,
):
    """
    Gradient, Divergence, Curl, Laplacian -> Photonics

    GRADIENT of scalar f:
      grad(f) = (df/dx)*x + (df/dy)*y + (df/dz)*z
      Points in direction of steepest increase
      |grad(f)| = rate of change in that direction
      Photonics: grad(phi) = k-vector (wavevector = gradient of phase)

    DIVERGENCE of vector A:
      div(A) = dAx/dx + dAy/dy + dAz/dz
      Measures source/sink density
      Gauss: div(E) = rho/epsilon0  (charge is source of E)
      Photonics: div(S) = 0 in lossless medium (energy conserved)

    CURL of vector A:
      curl(A) = (dAz/dy - dAy/dz)*x
              + (dAx/dz - dAz/dx)*y
              + (dAy/dx - dAx/dy)*z
      Measures rotation / circulation
      Faraday: curl(E) = -dB/dt  (changing B creates circulating E)
      Photonics: curl(E) = -j*omega*mu*H  (phasor form)

    LAPLACIAN:
      laplacian(f) = div(grad(f)) = d^2f/dx^2 + d^2f/dy^2 + d^2f/dz^2
      Wave equation: laplacian(E) = mu*epsilon * d^2E/dt^2
      In phasor: laplacian(E) + k^2*E = 0  [Helmholtz equation]
      GS phase: laplacian(phi) = 0 in free space (phi is harmonic)

    IDENTITIES (always true):
      div(curl(A)) = 0    (curl has no divergence)
      curl(grad(f)) = 0   (gradient has no curl)
    """
    # Grid
    x1d = np.linspace(-r_max, r_max, n_pts)
    dx = x1d[1] - x1d[0]
    X, Y, Z3 = np.meshgrid(x1d, x1d, [0.0], indexing='ij')
    X = X[:,:,0]; Y = Y[:,:,0]

    # Scalar field: f(x,y) = x^2 + y^2  (parabola)
    f = X**2 + Y**2

    # Gradient (numerical)
    grad_fx = np.gradient(f, dx, axis=0)
    grad_fy = np.gradient(f, dx, axis=1)
    # Analytic: grad(x^2+y^2) = (2x, 2y, 0)
    grad_fx_analytic = 2*X; grad_fy_analytic = 2*Y
    grad_err = float(np.max(np.abs(grad_fx[2:-2,2:-2] - grad_fx_analytic[2:-2,2:-2])))

    # Vector field: A = (-y, x, 0) / (x^2+y^2)  -- circulation field
    r2 = X**2 + Y**2 + 1e-10
    Ax = -Y/r2; Ay = X/r2

    # Divergence: div(A) = dAx/dx + dAy/dy
    dAx_dx = np.gradient(Ax, dx, axis=0)
    dAy_dy = np.gradient(Ay, dx, axis=1)
    div_A = dAx_dx + dAy_dy
    # For A = (-y, x)/(x^2+y^2): div = 0 (away from origin)
    div_max_away = float(np.max(np.abs(div_A[3:-3,3:-3])))

    # Curl (z-component): curl(A)_z = dAy/dx - dAx/dy
    dAy_dx = np.gradient(Ay, dx, axis=0)
    dAx_dy = np.gradient(Ax, dx, axis=1)
    curl_Az = dAy_dx - dAx_dy
    # For A = (-y,x)/(x^2+y^2): curl = 2*pi*delta(r) (singular at origin)
    # Away from origin: curl ~ 0
    curl_max_away = float(np.max(np.abs(curl_Az[3:-3,3:-3])))

    # Laplacian of f = x^2 + y^2: laplacian = 4 (constant)
    d2f_dx2 = np.gradient(np.gradient(f, dx, axis=0), dx, axis=0)
    d2f_dy2 = np.gradient(np.gradient(f, dx, axis=1), dx, axis=1)
    laplacian_f = d2f_dx2 + d2f_dy2
    laplacian_expected = 4.0   # analytic: d^2(x^2+y^2)/dx^2 + d^2/dy^2 = 2+2=4
    laplacian_err = float(np.max(np.abs(laplacian_f[2:-2,2:-2] - laplacian_expected)))

    # Vector Laplacian identity: curl(curl(A)) = grad(div(A)) - laplacian(A)
    # For this A: divergence free, so curl(curl(A)) = -laplacian(A)

    # Identities
    # curl(grad(f)) = 0:  compute curl of (grad_fx, grad_fy)
    d_gradfx_dy = np.gradient(grad_fx, dx, axis=1)
    d_gradfy_dx = np.gradient(grad_fy, dx, axis=0)
    curl_grad_f = d_gradfy_dx - d_gradfx_dy
    curl_grad_err = float(np.max(np.abs(curl_grad_f[2:-2,2:-2])))

    return {
        'grid': {'n_pts': n_pts, 'r_max': float(r_max), 'dx': float(dx)},
        'gradient': {
            'f': 'x^2 + y^2',
            'analytic_grad': 'grad(f) = (2x, 2y)',
            'numerical_error': float(grad_err),
            'photonics': 'grad(phase) = k-vector (wavevector)',
        },
        'divergence': {
            'A': '(-y, x) / (x^2+y^2)',
            'div_A_analytic': '0 (away from origin)',
            'div_max_numerical': float(div_max_away),
            'satisfied': bool(div_max_away < 0.1),
            'Gauss_law': 'div(E) = rho/epsilon0',
            'photonics': 'div(S) = 0 in lossless: energy conserved',
        },
        'curl': {
            'curl_A_z_away': float(curl_max_away),
            'Faraday': 'curl(E) = -dB/dt',
            'Ampere': 'curl(B) = mu0*J + mu0*eps0*dE/dt',
            'phasor': 'curl(E) = -j*omega*mu*H',
        },
        'laplacian': {
            'f': 'x^2 + y^2',
            'laplacian_f_expected': float(laplacian_expected),
            'numerical_error': float(laplacian_err),
            'Helmholtz': 'laplacian(E) + k^2*E = 0  [wave equation in phasor]',
            'photonics': 'Phase phi satisfies laplacian(phi)=0 (harmonic) in source-free region',
        },
        'identities': {
            'curl_grad_f_error': float(curl_grad_err),
            'curl_grad_f': 'curl(grad(f)) = 0  [VERIFIED]',
            'div_curl_A': 'div(curl(A)) = 0  [always true]',
            'BAC_CAB': 'A x (B x C) = B(A.C) - C(A.B)  [verified in triple_product_rules]',
        },
    }


def demo():
    print("=== VECTOR ALGEBRA PROBLEM SETS ===\n")

    # Problem Set 1: Basic operations
    print("--- Problem Set 1: Basic Operations ---")
    A = Vec3(1, 2, 3); B = Vec3(4, 5, 6)
    print(f"  A = {A},  B = {B}")
    ps1 = problem_set_basic_operations(A, B, scalar_c=3.0)

    print(f"\n  (1) ADDITION:  A + B = {ps1['addition']['A_plus_B']}")
    print(f"      Commutative (A+B = B+A): {ps1['addition']['commutative']}")

    print(f"\n  (2) SCALAR MULT:  3*A = {ps1['scalar_mult']['c_times_A']}")
    print(f"      Distributive c*(A+B) = c*A+c*B: {ps1['scalar_mult']['distributive']}")

    print(f"\n  (3) DOT PRODUCT:  A.B = {ps1['dot_product']['A_dot_B']:.0f}")
    print(f"      Commutative (A.B = B.A): {ps1['dot_product']['commutative']}")
    print(f"      A.A = |A|^2 = {ps1['dot_product']['A_dot_A']:.4f}: {ps1['dot_product']['self_dot_check']}")
    print(f"      Angle theta = {ps1['dot_product']['theta_deg']:.2f} deg")

    print(f"\n  (4) CROSS PRODUCT:  A x B = {ps1['cross_product']['A_cross_B']}")
    print(f"      Anti-commutative (A x B = -(B x A)): {ps1['cross_product']['anti_commutative']}")
    print(f"      |A x B| formula={ps1['cross_product']['|A_cross_B|_formula']:.4f} direct={ps1['cross_product']['|A_cross_B|_direct']:.4f}: {ps1['cross_product']['cross_mag_check']}")
    print(f"      Perp to A: {ps1['cross_product']['perpendicular_to_A']},  Perp to B: {ps1['cross_product']['perpendicular_to_B']}")
    print(f"      A x A = {ps1['cross_product']['A_cross_A']} (zero): {ps1['cross_product']['self_cross_zero']}")
    print(f"      Unit crosses: x x y = {ps1['cross_product']['unit_cross_products']['x x y']}")

    # Triple product rules
    print("\n--- Problem Set 2: Triple Product Rules ---")
    A2 = Vec3(1, 2, 3); B2 = Vec3(0, 1, 4); C2 = Vec3(-1, 3, 2)
    print(f"  A={A2}, B={B2}, C={C2}")
    tpr = triple_product_rules(A2, B2, C2)

    r1 = tpr['rule_i_scalar_triple']
    print(f"\n  RULE (i): Scalar Triple Product")
    print(f"    A.(B x C) = {r1['A.(B x C)']:.4f}")
    print(f"    B.(C x A) = {r1['B.(C x A)']:.4f}")
    print(f"    C.(A x B) = {r1['C.(A x B)']:.4f}")
    print(f"    All equal: {r1['all_equal']},  det = {r1['det_value']:.4f},  agrees: {r1['det_agrees']}")
    print(f"    Anti-sym A.(BxC)=-A.(CxB): {r1['anti_symmetry']}")

    r2 = tpr['rule_ii_bac_cab']
    print(f"\n  RULE (ii): BAC-CAB  A x (B x C) = B(A.C) - C(A.B)")
    print(f"    LHS = {r2['LHS: A x (B x C)']}")
    print(f"    RHS = {r2['RHS: B(A.C) - C(A.B)']}")
    print(f"    Equal: {r2['equal']},  error={r2['error']:.2e}")

    r3 = tpr['rule_iii_axb_c']
    print(f"\n  RULE (iii): (A x B) x C = B(A.C) - A(B.C)")
    print(f"    LHS = {r3['LHS: (A x B) x C']}")
    print(f"    RHS = {r3['RHS: B(A.C) - A(B.C)']}")
    print(f"    Equal: {r3['equal']}")
    print(f"    Cross product NOT associative (ii != iii): {r3['not_assoc']}")

    print(f"\n  Physical examples:")
    pe = tpr['physical_examples']
    print(f"    Poynting S = E x B = {pe['Poynting_S = E x B']['S_direction']} ({pe['Poynting_S = E x B']['note']})")
    print(f"    Lorentz F = v x B = {pe['magnetic_force_F = qv x B']['F = v x B']} ({pe['magnetic_force_F = qv x B']['note']})")

    # Pressure + statistics
    print("\n--- Pressure + Statistics ---")
    ps = pressure_statistics()
    print(f"  Kinetic theory P = ideal gas P: {ps['kinetic_theory']['agreement']}")
    print(f"  P at 300K = {ps['kinetic_theory']['P_ideal_Pa']:.0f} Pa ({ps['kinetic_theory']['P_ideal_Pa']/101325:.3f} atm)")
    print(f"  Speed order vp<vmean<vrms: {ps['maxwell_boltzmann']['order_vp_vmean_vrms']}")
    print(f"  vp={ps['maxwell_boltzmann']['v_mp_m_s']:.0f}, vmean={ps['maxwell_boltzmann']['v_mean_m_s']:.0f}, vrms={ps['maxwell_boltzmann']['v_rms_m_s']:.0f} m/s")
    print(f"  Radiation pressure (1mW/mm^2, absorb): {ps['radiation_pressure']['P_rad_absorb_Pa']:.2e} Pa")
    print(f"  Optical tweezers (1mW/um^2): {ps['radiation_pressure']['P_rad_tweezers_Pa']:.2e} Pa")
    print(f"  Boltzmann P(ground)={ps['boltzmann_statistics']['P_states'][0]:.4f}")

    # Vector calculus
    print("\n--- Vector Calculus (grad/div/curl/laplacian) ---")
    vc = vector_calculus_ops(n_pts=30)
    print(f"  grad(x^2+y^2) numerical error: {vc['gradient']['numerical_error']:.4f}")
    print(f"  div(A)=0 away from origin: {vc['divergence']['satisfied']}")
    print(f"  laplacian(x^2+y^2)=4 error: {vc['laplacian']['numerical_error']:.4f}")
    print(f"  curl(grad(f))=0 error: {vc['identities']['curl_grad_f_error']:.2e}")
    print(f"  Identity: {vc['identities']['curl_grad_f']}")

    print("\n=== VECTOR ALGEBRA COMPLETE ===")


if __name__ == '__main__':
    demo()
