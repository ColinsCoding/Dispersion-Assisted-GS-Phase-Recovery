"""
Vector Calculus with Complex-Valued Entries

"Show the dark side" -- complex fields have a real part AND an imaginary part.
Most textbooks only show Re[E]. This module shows both.

The imaginary part is NOT fictional:
  Im[E] encodes phase information -- exactly what GS phase retrieval recovers.
  Im[n] = absorption (Beer-Lambert k from complex_refractive_index).
  Im[Z] = reactance (energy stored, not dissipated, in circuit).
  Im[chi] = dielectric loss (heat generated in microwave oven).
  Im[H(f)] = phase response of dispersion filter.

EC ENGR 279AS (UCLA, Jalali dept): "electromagnetics, microwave and millimeter wave
circuits, photonics and optoelectronics" -- ALL of these require complex vector calculus.
The complex Poynting vector, complex wavenumber, complex impedance.

OUSD priorities from annotated image:
  FutureG -> beamforming (complex weights), MIMO (complex channel matrix)
  Directed Energy -> complex field amplitude at target, Gouy phase
  Integrated Sensing -> complex radar return (I/Q), complex transfer function
  Trusted AI -> complex-valued neural networks (CVNNs) for radar/comms
"""
import numpy as np
import sympy as sp


# ===========================================================================
# Part 0: Pre-calculus foundation -- trig, unit circle, unit sphere
# ===========================================================================

def unit_circle_trig():
    """
    The unit circle: all of trig lives here.
    r=1: x=cos(theta), y=sin(theta), x^2+y^2=1.

    Every complex number z = r*exp(j*theta) = r*(cos(theta)+j*sin(theta)).
    Euler's formula: exp(j*theta) = cos(theta) + j*sin(theta)
    This IS the unit circle in the complex plane.

    Key values:
      theta=0:    (1, 0)     -> cos=1, sin=0
      theta=pi/6: (sqrt3/2, 1/2)   30 deg
      theta=pi/4: (1/sqrt2, 1/sqrt2) 45 deg
      theta=pi/3: (1/2, sqrt3/2)   60 deg
      theta=pi/2: (0, 1)     -> cos=0, sin=1
      theta=pi:   (-1, 0)    -> cos=-1, sin=0

    Metal salt emission connection:
      The quantum wave function is complex: psi(r,t) = R(r)*exp(j*m*phi)*exp(-j*E*t/hbar)
      The angular part Y_l^m(theta, phi) = functions of cos/sin on the unit SPHERE.
      Atomic orbitals = solutions to Laplacian on unit sphere (same math as below).
    """
    theta = np.linspace(0, 2*np.pi, 360)
    x = np.cos(theta); y = np.sin(theta)

    # Key angles
    key_angles_deg = [0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330, 360]
    key = {}
    for d in key_angles_deg:
        t = d * np.pi/180
        key[f'{d}deg'] = {
            'theta_rad': t,
            'cos': round(float(np.cos(t)), 6),
            'sin': round(float(np.sin(t)), 6),
            'exp_j_theta': complex(round(np.cos(t),4), round(np.sin(t),4)),
        }

    # Pythagorean identities
    t = sp.Symbol('t', real=True)
    identity_1 = sp.simplify(sp.cos(t)**2 + sp.sin(t)**2)
    identity_2 = sp.simplify(1 + sp.tan(t)**2 - 1/sp.cos(t)**2)

    return {
        'theta': theta,
        'x': x, 'y': y,
        'key_angles': key,
        'pythagorean': f'cos^2+sin^2={identity_1}  (=1 always)',
        'eulers_formula': 'exp(j*theta) = cos(theta) + j*sin(theta)',
        'DeMoivre': '(cos(theta)+j*sin(theta))^n = cos(n*theta)+j*sin(n*theta)',
        'trig_addition': {
            'cos(a+b)': 'cos(a)cos(b) - sin(a)sin(b)',
            'sin(a+b)': 'sin(a)cos(b) + cos(a)sin(b)',
            'double_angle_cos': '1 - 2*sin^2(theta) = 2*cos^2(theta) - 1 = cos(2theta)',
        },
        'quantum_connection': 'Y_lm(theta,phi) = spherical harmonics = trig on unit sphere = atomic orbitals',
    }


def unit_sphere_coordinates():
    """
    Unit sphere: all points at distance 1 from origin in 3D.
    Parameterized by (theta, phi): theta=polar, phi=azimuthal.

    Cartesian from spherical:
      x = sin(theta)*cos(phi)
      y = sin(theta)*sin(phi)
      z = cos(theta)
      r = 1 (unit sphere)

    Unit vectors in spherical coordinates:
      r_hat = (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
      theta_hat = (cos(theta)cos(phi), cos(theta)sin(phi), -sin(theta))
      phi_hat = (-sin(phi), cos(phi), 0)

    These are ORTHONORMAL: r_hat.theta_hat=0, r_hat.phi_hat=0, theta_hat.phi_hat=0.
    And they DEPEND ON POSITION -- unlike Cartesian x_hat, y_hat, z_hat.
    This is why vector calculus in spherical coords has extra terms (Christoffel symbols).

    Applications:
      Antenna radiation pattern: power(theta,phi) on unit sphere
      Atomic orbitals: |Y_lm(theta,phi)|^2 on unit sphere
      Radar cross section: how much power reflected toward (theta,phi)
      Directed energy: beam steering to (theta,phi) from phased array
    """
    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2*np.pi, 100)
    Theta, Phi = np.meshgrid(theta, phi)

    X = np.sin(Theta)*np.cos(Phi)
    Y = np.sin(Theta)*np.sin(Phi)
    Z = np.cos(Theta)

    # Dipole radiation pattern on sphere: P(theta) = sin^2(theta)
    P_dipole = np.sin(Theta)**2   # normalized

    # Verify orthonormality of basis vectors at one point
    th, ph = np.pi/4, np.pi/3
    r_hat     = np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])
    theta_hat = np.array([np.cos(th)*np.cos(ph), np.cos(th)*np.sin(ph), -np.sin(th)])
    phi_hat   = np.array([-np.sin(ph), np.cos(ph), 0.0])

    ortho = {
        'r_dot_theta': float(np.dot(r_hat, theta_hat)),
        'r_dot_phi':   float(np.dot(r_hat, phi_hat)),
        'theta_dot_phi': float(np.dot(theta_hat, phi_hat)),
        'r_norm': float(np.linalg.norm(r_hat)),
    }

    return {
        'X': X, 'Y': Y, 'Z': Z,
        'Theta': Theta, 'Phi': Phi,
        'P_dipole': P_dipole,
        'r_hat': r_hat,
        'theta_hat': theta_hat,
        'phi_hat': phi_hat,
        'orthonormality': ortho,
        'all_orthogonal': all(abs(v) < 1e-10 for k,v in ortho.items() if 'dot' in k),
        'position_dependent': (
            'CRITICAL: spherical unit vectors r_hat, theta_hat, phi_hat change direction '
            'as you move around the sphere. Cartesian x_hat never changes. '
            'This is why d(r_hat)/d(theta) = theta_hat (not zero). '
            'Causes extra terms in Laplacian, gradient, divergence in spherical coords.'
        ),
    }


# ===========================================================================
# Part 1: Gradient, Divergence, Curl (real and complex)
# ===========================================================================

def gradient_demo():
    """
    grad(f) = del(f) = (df/dx, df/dy, df/dz)

    Gradient points in direction of STEEPEST INCREASE of f.
    |grad(f)| = rate of change per unit length in that direction.
    E = -grad(V): electric field = negative gradient of potential.
    Force = -grad(U): force = negative gradient of potential energy.

    For complex scalar field f = f_R + j*f_I:
      grad(f) = grad(f_R) + j*grad(f_I)
      BOTH parts have gradients. E field of lossy medium has complex gradient.

    "Through time" -- time-varying gradient:
      d/dt[grad(V)] = grad(dV/dt)  [gradient commutes with time derivative]
      In photonics: phase gradient = local k-vector = direction of propagation.
      Group delay = d(phi)/d(omega) = time gradient of phase spectrum.
    """
    x, y, z = sp.symbols('x y z', real=True)
    omega, t, k = sp.symbols('omega t k', real=True)

    # Real potential: V = x^2 + 2*y^2 - z
    V_real = x**2 + 2*y**2 - z
    grad_V_real = [sp.diff(V_real, xi) for xi in [x, y, z]]
    E_from_V = [-g for g in grad_V_real]   # E = -grad(V)

    # Complex plane wave: f = exp(j*(k*x - omega*t))
    f_complex = sp.exp(sp.I*(k*x - omega*t))
    grad_f_x = sp.diff(f_complex, x)  # j*k * f  (complex!)
    # grad_f points in x direction with magnitude |j*k| = k, phase rotated by 90 deg

    # Gaussian beam phase: phi(x,y,z) = k*z + k*(x^2+y^2)/(2*R) - arctan(z/z_R)
    R, z_R = sp.symbols('R z_R', positive=True)
    phi_beam = k*z + k*(x**2 + y**2)/(2*R) - sp.atan(z/z_R)
    grad_phi = [sp.diff(phi_beam, xi) for xi in [x, y, z]]

    return {
        'V_real': str(V_real),
        'grad_V': [str(g) for g in grad_V_real],
        'E_field': [str(e) for e in E_from_V],
        'plane_wave_grad_x': str(sp.simplify(grad_f_x)),
        'plane_wave_lesson': 'grad(exp(j*k*x)) = j*k*exp(j*k*x): gradient of complex wave = j*k * wave (90 deg rotation + scale)',
        'beam_grad_phi': [str(g) for g in grad_phi],
        'photonics_lesson': (
            'Local k-vector = grad(phase). '
            'Gaussian beam phase gradient in (x,y): k*(x,y)/R -> rays curve toward axis (focusing). '
            'Gouy phase gradient in z: k - 1/(z+j*z_R) -> anomalous phase shift through focus.'
        ),
        'group_delay': 'tau(omega) = d(phi)/d(omega) = time derivative of phase spectrum [same math]',
    }


def divergence_demo():
    """
    div(F) = del.F = dFx/dx + dFy/dy + dFz/dz

    Divergence = source density.
    div(E) = rho/epsilon_0  [Gauss's law: charge density creates E field lines]
    div(B) = 0              [no magnetic monopoles: B field lines never end]
    div(J) = -d(rho)/dt     [continuity equation: charge conservation]

    For complex field F = F_R + j*F_I:
      div(F) = div(F_R) + j*div(F_I)
      In lossy media: complex div(E) != 0 in general (need careful definition).

    COMPLEX DIVERGENCE (dark side):
      E field in absorbing medium: E = E0 * exp(-k*x) * exp(j*(k_r*x - omega*t))
      dEx/dx = (-k + j*k_r) * Ex = complex number
      Real part: field amplitude decreasing (absorption)
      Imaginary part: phase changing (propagation)
      Both are real physics. Neither is fictional.

    Divergence theorem (Gauss's theorem):
      int_V div(F) dV = oint_S F.dA
      Volume integral of sources = flux through surface.
      Proof of Gauss's law integral form from differential form.
    """
    x, y, z, k_r, k_i = sp.symbols('x y z k_r k_i', real=True)
    omega, t = sp.symbols('omega t', real=True)

    # Point charge E field: E = q/(4*pi*eps0) * r_hat/r^2
    # In component form: E = q*r/(4*pi*eps0*r^3) = q*(x,y,z)/(4*pi*eps0*(x^2+y^2+z^2)^(3/2))
    r2 = x**2 + y**2 + z**2
    r3 = (x**2 + y**2 + z**2)**sp.Rational(3,2)
    Ex = x/r3; Ey = y/r3; Ez = z/r3
    div_E = sp.diff(Ex,x) + sp.diff(Ey,y) + sp.diff(Ez,z)
    div_E_simplified = sp.simplify(div_E)

    # Complex field in absorbing medium
    k_complex = k_r + sp.I * k_i   # complex wavenumber
    E_absorbing = sp.exp(sp.I * k_complex * x - sp.I*omega*t)
    dE_dx = sp.diff(E_absorbing, x)
    dE_dx_simplified = sp.simplify(dE_dx)

    return {
        'point_charge_div_E': f'div(E) for point charge = {div_E_simplified} (zero away from origin)',
        'Gauss_law': 'div(E) = rho/eps0: charge IS the source of E field lines',
        'no_monopoles': 'div(B) = 0: B field lines form closed loops, never start/end',
        'continuity': 'div(J) = -d(rho)/dt: current diverging from point = charge disappearing',
        'complex_dE_dx': str(dE_dx_simplified),
        'complex_lesson': (
            'dE/dx for E in absorbing medium = (j*k_r - k_i)*E.\n'
            'Real part (-k_i*E): amplitude decay = absorption.\n'
            'Imaginary part (j*k_r*E): phase rotation = propagation.\n'
            'BOTH are physical. The dark side (imaginary) = propagation.'
        ),
        'divergence_theorem': 'int_V div(F) dV = oint_S F.dA  [Gauss theorem]',
    }


def curl_demo():
    """
    curl(F) = del x F = (dFz/dy - dFy/dz, dFx/dz - dFz/dx, dFy/dx - dFx/dy)

    Curl = rotation density.
    curl(E) = -dB/dt     [Faraday's law: changing B creates E vortex]
    curl(B) = mu0*J + mu0*eps0*dE/dt  [Ampere-Maxwell: current + displacement current]
    curl(grad(f)) = 0    [gradient has no curl: conservative fields are irrotational]
    div(curl(F)) = 0     [curl has no divergence: always]

    These two identities are vector calc identities that simplify Maxwell's equations
    and let us derive wave equations. Same identities used in differential geometry
    and topology (de Rham cohomology -- but that's grad school material).

    Complex curl (dark side):
      In complex notation, Maxwell's equations become MUCH cleaner:
      curl(E) = -j*omega*mu*H
      curl(H) = J + j*omega*epsilon*E
      The j*omega = time derivative in phasor domain.
      This is why EC ENGR 279AS (UCLA) uses complex notation throughout.

    Stokes' theorem:
      oint_C F.dl = int_S (curl F).dA
      Line integral around curve = surface integral of curl.
      Faraday's law: oint E.dl = -d/dt int B.dA = -d(Phi_B)/dt [EMF = -d(flux)/dt]
    """
    x, y, z = sp.symbols('x y z', real=True)
    omega, mu, eps = sp.symbols('omega mu epsilon', positive=True)
    Ex, Ey, Ez, Hx, Hy, Hz = sp.symbols('Ex Ey Ez Hx Hy Hz', complex=True)

    # curl of B field from infinite wire: B = mu0*I/(2*pi*r) * phi_hat
    # In Cartesian: Bx = -mu0*I*y/(2*pi*(x^2+y^2)), By = mu0*I*x/(2*pi*(x^2+y^2))
    r2 = x**2 + y**2
    mu0, I = sp.symbols('mu0 I', positive=True)
    Bx = -mu0*I*y / (2*sp.pi*r2)
    By =  mu0*I*x / (2*sp.pi*r2)
    Bz = sp.Integer(0)
    # curl_z = dBy/dx - dBx/dy
    curl_z = sp.simplify(sp.diff(By,x) - sp.diff(Bx,y))

    # Check vector identity: curl(grad(f)) = 0
    f = sp.Function('f')(x, y, z)
    gf = [sp.diff(f, xi) for xi in [x,y,z]]
    curl_grad_x = sp.diff(gf[2],y) - sp.diff(gf[1],z)
    curl_grad_y = sp.diff(gf[0],z) - sp.diff(gf[2],x)
    curl_grad_z = sp.diff(gf[1],x) - sp.diff(gf[0],y)
    is_zero = (sp.simplify(curl_grad_x)==0 and
               sp.simplify(curl_grad_y)==0 and
               sp.simplify(curl_grad_z)==0)

    return {
        'curl_B_of_wire_z_component': str(curl_z),
        'curl_B_lesson': 'curl(B) = mu0*J (Ampere): B forms closed loops around current -- nonzero curl',
        'Faraday': 'curl(E) = -dB/dt  ->  EMF = -d(Phi)/dt via Stokes theorem',
        'Maxwell_complex_phasor': {
            'curl_E': 'curl(E) = -j*omega*mu*H  (time deriv -> multiply by j*omega)',
            'curl_H': 'curl(H) = J + j*omega*epsilon*E',
            'lesson': 'j*omega in phasor domain = d/dt in time domain. Complex notation eliminates cos/sin.',
        },
        'vector_identity_curl_grad_zero': f'curl(grad(f)) = 0: {is_zero}',
        'vector_identity_div_curl_zero': 'div(curl(F)) = 0 (always)',
        'Stokes_theorem': 'oint_C F.dl = int_S curl(F).dA  [line -> surface]',
        'dark_side': (
            'Complex Maxwell: fields are complex-valued. '
            'E(r,t) = Re[E_complex * exp(-j*omega*t)]. '
            'E_complex has both real and imaginary parts at each point in space. '
            'You cannot see or measure Im[E] directly -- you measure |E|^2 (intensity). '
            'GS phase retrieval RECOVERS Im[E] from intensity measurements. '
            'That is the entire point of this repository.'
        ),
    }


def laplacian_demo():
    """
    del^2 f = div(grad(f)) = d^2f/dx^2 + d^2f/dy^2 + d^2f/dz^2

    Laplacian = divergence of gradient = total second derivative.

    Key equations:
      Laplace equation:   del^2(V) = 0       [electrostatics in free space]
      Poisson equation:   del^2(V) = -rho/eps [with charge]
      Wave equation:      del^2(E) = (1/c^2) d^2E/dt^2
      Schrodinger:        del^2(psi) + (2m/hbar^2)(E-V)*psi = 0
      Heat equation:      del^2(T) = (1/alpha) dT/dt

    In spherical coordinates (for atoms, antennas, black holes):
      del^2(f) = (1/r^2) d/dr[r^2 df/dr] + (1/r^2 sin(theta)) d/d(theta)[sin(theta) df/d(theta)]
               + (1/r^2 sin^2(theta)) d^2f/d(phi)^2

    Solutions to del^2(psi)=0 on unit sphere = SPHERICAL HARMONICS Y_lm(theta,phi).
    These are the angular parts of hydrogen atom orbitals.
    Same Y_lm appear in: antenna radiation patterns, gravitational multipoles, MIMO beamforming.

    Complex Laplacian (wave equation with complex k):
      E(r,t) = E0 * exp(j*(k_r + j*k_i)*z - j*omega*t)
             = E0 * exp(-k_i*z) * exp(j*(k_r*z - omega*t))
      del^2(E) = -(k_r + j*k_i)^2 * E = -(k_r^2 - k_i^2 + 2*j*k_r*k_i) * E
      Dark side: imaginary part of k^2 = wave absorption in medium.
    """
    x, y, z = sp.symbols('x y z', real=True)
    r, theta, phi = sp.symbols('r theta phi', positive=True)

    # 1D wave: del^2(exp(j*k*z)) = d^2/dz^2 exp(j*k*z) = -k^2 * exp(j*k*z)
    k = sp.Symbol('k', complex=True)
    wave = sp.exp(sp.I*k*z)
    lap_wave = sp.diff(wave, z, 2)

    # Spherical Laplacian of 1/r:
    r_sym = sp.Symbol('r', positive=True)
    f = 1/r_sym
    # del^2(1/r) = 0 for r>0  (Laplace eq for point charge potential)
    d2f = sp.diff(r_sym**2 * sp.diff(f, r_sym), r_sym) / r_sym**2
    d2f_simplified = sp.simplify(d2f)

    # Spherical harmonics Y_00, Y_10, Y_11 (real form)
    Y_00 = 1/sp.sqrt(4*sp.pi)
    Y_10 = sp.sqrt(3/(4*sp.pi)) * sp.cos(theta)
    Y_11_real = sp.sqrt(3/(4*sp.pi)) * sp.sin(theta) * sp.cos(phi)

    return {
        'wave_laplacian': f'del^2(exp(j*k*z)) = {sp.simplify(lap_wave)}',
        'laplace_1_over_r': f'del^2(1/r) = {d2f_simplified} (for r>0)',
        'equations': {
            'Laplace':    'del^2(V) = 0      [free space, no charges]',
            'Poisson':    'del^2(V) = -rho/eps  [with charge density]',
            'wave':       'del^2(E) - (1/c^2)*d^2E/dt^2 = 0',
            'Schrodinger':'del^2(psi) + (2m/hbar^2)*(E-V)*psi = 0',
            'Heat':       'del^2(T) = (1/alpha)*dT/dt',
        },
        'spherical_harmonics': {
            'Y_00': str(Y_00),
            'Y_10': str(Y_10),
            'Y_11_real': str(Y_11_real),
            'note': '|Y_lm(theta,phi)|^2 = probability density on unit sphere = orbital shape',
        },
        'complex_k_dark_side': (
            'k = k_r + j*k_i (complex wavenumber).\n'
            'Real part k_r: propagation (oscillation in space).\n'
            'Imaginary part k_i: absorption (amplitude decay).\n'
            'k^2 = k_r^2 - k_i^2 + 2*j*k_r*k_i.\n'
            'Dark side Im[k^2] = 2*k_r*k_i = connects propagation to absorption.'
        ),
    }


# ===========================================================================
# Part 2: Complex Vector Fields (the dark side, visualized)
# ===========================================================================

def complex_field_2d(field_type='EM_wave', N=30):
    """
    Visualize complex vector fields in 2D -- both real AND imaginary parts.

    Most textbooks show Re[E] only. This shows:
      Re[E]: the bright side (what you measure at t=0)
      Im[E]: the dark side (what you measure at t=pi/(2*omega))
      |E|:   the amplitude (what a detector sees, time-averaged)
      angle(E): the phase (what GS recovers)

    For EM plane wave propagating in x:
      E(x,y,t) = E0 * y_hat * exp(j*(k*x - omega*t))
      At t=0: E = E0 * exp(j*k*x) = E0*(cos(k*x) + j*sin(k*x)) * y_hat
      Re[E] at t=0: E0*cos(k*x)*y_hat  [what you see]
      Im[E] at t=0: E0*sin(k*x)*y_hat  [the dark side -- same wave, quarter-period later]

    "Through time": t changes -> Re[E] and Im[E] exchange (they oscillate in quadrature).
    """
    x = np.linspace(-np.pi, np.pi, N)
    y = np.linspace(-np.pi, np.pi, N)
    X, Y = np.meshgrid(x, y)

    if field_type == 'EM_wave':
        k = 1.0
        E = np.exp(1j * k * X)   # plane wave in x direction (complex)
        # y-polarized: only Ey component
        Ereal_x = np.zeros_like(X); Ereal_y = np.real(E)
        Eimag_x = np.zeros_like(X); Eimag_y = np.imag(E)
        title = 'EM plane wave E_y = exp(j*k*x)'

    elif field_type == 'point_source':
        # Complex field from point source: E ~ exp(j*k*r)/r
        r = np.sqrt(X**2 + Y**2 + 0.1)
        k = 2.0
        E_scalar = np.exp(1j*k*r) / r
        # Radial field: E = E_scalar * r_hat
        Ereal_x = np.real(E_scalar) * X/r
        Ereal_y = np.real(E_scalar) * Y/r
        Eimag_x = np.imag(E_scalar) * X/r
        Eimag_y = np.imag(E_scalar) * Y/r
        title = 'Point source E = exp(j*k*r)/r * r_hat'

    elif field_type == 'vortex':
        # Optical vortex: exp(j*m*phi) * exp(-r^2/w^2) -- carries orbital angular momentum
        m = 1   # topological charge
        r = np.sqrt(X**2 + Y**2 + 0.01)
        phi_arr = np.arctan2(Y, X)
        w = 1.5
        E_scalar = np.exp(1j*m*phi_arr) * np.exp(-r**2/w**2)
        Ereal_x = np.real(E_scalar) * (-Y/r)   # phi_hat x-component = -sin(phi) = -y/r
        Ereal_y = np.real(E_scalar) * ( X/r)
        Eimag_x = np.imag(E_scalar) * (-Y/r)
        Eimag_y = np.imag(E_scalar) * ( X/r)
        title = f'Optical vortex m={m}: exp(j*phi) [orbital angular momentum]'

    E_complex = (Ereal_x + 1j*Eimag_x, Ereal_y + 1j*Eimag_y)
    amplitude = np.sqrt(Ereal_x**2 + Ereal_y**2 + Eimag_x**2 + Eimag_y**2)
    phase = np.arctan2(np.sqrt(Eimag_x**2 + Eimag_y**2),
                       np.sqrt(Ereal_x**2 + Ereal_y**2))

    return {
        'X': X, 'Y': Y,
        'Ereal_x': Ereal_x, 'Ereal_y': Ereal_y,
        'Eimag_x': Eimag_x, 'Eimag_y': Eimag_y,
        'amplitude': amplitude,
        'phase': phase,
        'title': title,
        'dark_side_lesson': (
            'Im[E] is NOT zero. It is the SAME wave shifted by 90 degrees in time. '
            'If you can only measure Re[E] (intensity measurements), '
            'you CANNOT distinguish a wave from its time-shifted copy. '
            'Phase retrieval (GS) uses BOTH amplitude measurements to pin down Im[E].'
        ),
    }


def complex_poynting_vector(E0=1.0, n_complex=complex(1.5, 0.01),
                             wavelength_nm=1550.0, z_arr=None):
    """
    Complex Poynting vector: S = E x H* (time-averaged power flow).

    For plane wave in medium with complex n = n_r + j*n_i:
      E = E0 * exp(j*(k_r + j*k_i)*z) * x_hat
      H = (E/eta) * y_hat  where eta = mu0*c/n (wave impedance)

      <S> = (1/2) Re[E x H*] = (E0^2 / (2*eta_r)) * exp(-2*k_i*z) * z_hat

    Time-averaged power: decays as exp(-alpha*z) where alpha=2*k_i=absorption.
    Complex power = Re[S] + j*Im[S]:
      Re[S]: active power (real energy flow, heats absorber)
      Im[S]: reactive power (oscillates between E and H, no net transport)

    Dark side: Im[S] != 0 even in lossless dielectric (n real).
    For lossless: eta is real, S is real -> Im[S]=0.
    For lossy: eta is complex -> Im[S]!=0 -> reactive stored energy.

    EC ENGR 279AS: complex Poynting theorem is the energy budget for microwave circuits.
    Same math as complex power in AC circuits: P = (1/2)*Re[V*I*].
    """
    c = 2.998e8; mu0 = 4*np.pi*1e-7; eps0 = 8.854e-12
    lam = wavelength_nm * 1e-9
    omega = 2*np.pi*c / lam

    n_r = n_complex.real; n_i = n_complex.imag
    k0 = omega / c
    k_complex = k0 * n_complex   # k = k0*(n_r + j*n_i)
    k_r = k_complex.real; k_i = k_complex.imag

    eta = (mu0*c / n_complex)   # complex wave impedance

    if z_arr is None:
        alpha = 2 * k_i
        if alpha > 0:
            z_arr = np.linspace(0, 5.0/alpha, 400)
        else:
            z_arr = np.linspace(0, 10*lam, 400)

    E_z = E0 * np.exp(1j*k_complex*z_arr)
    H_z = E_z / eta

    S_complex = 0.5 * E_z * np.conj(H_z)
    S_real = np.real(S_complex)
    S_imag = np.imag(S_complex)

    S0 = 0.5 * E0**2 / np.real(eta)   # peak Poynting at z=0

    return {
        'z_m': z_arr,
        'S_real': S_real,
        'S_imag': S_imag,
        'S_magnitude': np.abs(S_complex),
        'n_complex': n_complex,
        'k_complex': k_complex,
        'eta_complex': eta,
        'S0_W_per_m2': float(S0.real) if hasattr(S0, 'real') else float(S0),
        'alpha_per_m': 2*k_i,
        'skin_depth_m': 1.0/(2*k_i) if k_i > 0 else float('inf'),
        'lesson': (
            f'n = {n_r:.3f} + j*{n_i:.4f}  '
            f'k = {k_r:.2f} + j*{k_i:.4f} /m  '
            f'alpha = {2*k_i:.4f} /m  '
            f'skin depth = {1/(2*k_i)*1e6:.1f} um'
            if k_i > 0 else
            f'n = {n_r:.3f} (lossless): Im[S] = reactive stored energy'
        ),
        'AC_analogy': (
            'Complex Poynting = complex power in AC circuit.\n'
            'S = (1/2)*E x H* = P + j*Q  [active + reactive power]\n'
            'P = (1/2)*Re[V*I*] = (1/2)*|I|^2*R  (heat dissipated)\n'
            'Q = (1/2)*Im[V*I*] = (1/2)*|I|^2*(X_L-X_C)  (reactive, stored)\n'
            'Power factor: cos(angle(S)) = P/|S|.'
        ),
    }


def ray_tracing_bvh():
    """
    Binary tree in light tracing: Bounding Volume Hierarchy (BVH).

    Ray tracing: for each pixel, cast a ray, test against all objects.
    Naive: O(N) per ray * M rays = O(N*M) total  [too slow for N=10^6 triangles]
    BVH:   O(log N) per ray  [tree traversal]

    BVH construction:
      1. Compute bounding box of all objects
      2. Split along longest axis (surface area heuristic)
      3. Left child = objects on left, right child = objects on right
      4. Recurse until leaf = 1-4 triangles
      -> Binary tree with O(N) nodes, O(N log N) build time

    Ray-box intersection test:
      For axis-aligned box [x_min,x_max] x [y_min,y_max] x [z_min,z_max]:
      t_x_min = (x_min - ray_origin.x) / ray_dir.x
      t_x_max = (x_max - ray_origin.x) / ray_dir.x
      ... similarly for y, z
      t_enter = max(t_x_min, t_y_min, t_z_min)
      t_exit  = min(t_x_max, t_y_max, t_z_max)
      Hit iff t_enter < t_exit and t_exit > 0

    Mirror reflection (the "mirror" in binary tree in light tracing):
      r = d - 2*(d.n)*n  [reflect direction d about normal n]
      Same as Householder reflection matrix: I - 2*n*n^T

    Refraction (Snell's law in vector form):
      n1*(d x n) = n2*(t x n)  [component parallel to surface conserved]
      t = (n1/n2)*d + (n1/n2*cos(theta_i) - cos(theta_t))*n

    Complex angle: when n1 > n2 and theta_i > critical angle:
      cos(theta_t) = sqrt(1 - (n1/n2)^2 * sin^2(theta_i))  <- COMPLEX!
      Evanescent wave: exp(j*k_t*z) where k_t is imaginary -> exp(-|k_t|*z)
      This is the dark side of ray tracing: beyond critical angle,
      'refracted ray' becomes exponentially decaying evanescent wave.
      Used in: TIRF microscopy, fiber optic total internal reflection, tunnel junctions.
    """
    # Ray-AABB intersection
    def ray_box_intersect(origin, direction, box_min, box_max):
        t_mins = (np.array(box_min) - np.array(origin)) / (np.array(direction) + 1e-20)
        t_maxs = (np.array(box_max) - np.array(origin)) / (np.array(direction) + 1e-20)
        t1 = np.minimum(t_mins, t_maxs)
        t2 = np.maximum(t_mins, t_maxs)
        t_enter = np.max(t1); t_exit = np.min(t2)
        return t_exit > 0 and t_enter < t_exit

    # Mirror reflection
    def reflect(d, n):
        d = np.array(d, dtype=float); n = np.array(n, dtype=float)
        n = n / np.linalg.norm(n)
        return d - 2*np.dot(d, n)*n

    # Snell's law vector form
    def refract(d, n_hat, n1, n2):
        d = np.array(d, dtype=float); n_hat = np.array(n_hat, dtype=float)
        n_hat = n_hat / np.linalg.norm(n_hat)
        ratio = n1 / n2
        cos_i = -np.dot(d, n_hat)
        discriminant = 1 - ratio**2 * (1 - cos_i**2)
        if discriminant < 0:
            # Total internal reflection: return None (evanescent wave)
            return None, complex(0, -np.sqrt(-discriminant))
        cos_t = np.sqrt(discriminant)
        t_ray = ratio*d + (ratio*cos_i - cos_t)*n_hat
        return t_ray, cos_t

    # Test cases
    hit = ray_box_intersect([0,0,-5], [0,0,1], [-1,-1,0], [1,1,2])
    miss = ray_box_intersect([2,2,-5], [0,0,1], [-1,-1,0], [1,1,2])
    r_vec = reflect([0, -1, 0], [0, 1, 0])  # downward ray, upward normal -> reflects up
    t_normal, _ = refract([0, 0, 1], [0, 0, -1], 1.0, 1.5)  # air -> glass, normal incidence
    t_tir, k_evanescent = refract([0.9, 0, np.sqrt(1-0.81)], [0, 0, -1], 1.5, 1.0)  # glass -> air, steep

    return {
        'ray_box_hit': hit,
        'ray_box_miss': miss,
        'reflected_direction': r_vec.tolist(),
        'refracted_normal_incidence': t_normal.tolist() if t_normal is not None else None,
        'total_internal_reflection': t_tir is None,
        'evanescent_k': k_evanescent,
        'evanescent_lesson': (
            'Total internal reflection: refracted ray becomes evanescent wave.\n'
            'cos(theta_t) = sqrt(1 - (n1/n2)^2*sin^2(theta_i)) is IMAGINARY.\n'
            'Wave: exp(j*k_t*z) = exp(-|k_t|*z) -- exponential decay, no propagation.\n'
            'This IS the dark side of refraction. Same math as quantum tunneling.\n'
            'TIRF microscopy: excite only within ~100nm of surface using evanescent field.'
        ),
        'bvh_lesson': (
            'BVH = binary tree of bounding boxes.\n'
            'Each node: left child + right child (two sub-volumes).\n'
            'Traversal: test parent box first, only recurse if hit.\n'
            'O(log N) per ray vs O(N) naive. 1000x speedup for N=10^6 triangles.\n'
            'Same binary tree as BST (algorithms.py) but built on spatial data.'
        ),
    }


def ousd_vector_calc_alignment():
    """
    How vector calculus with complex entries directly enables OUSD priority CTAs.
    From the annotated image: FutureG, Trusted AI, Advanced Computing,
    Integrated Sensing & Cyber, Directed Energy, Human-Machine Interfaces.
    """
    return {
        'FutureG': {
            'tech': '5G/6G beamforming, MIMO, massive antenna arrays',
            'math': 'Complex beamforming weights w = a*exp(j*phi_i). Beam = sum w_i * signal.',
            'vector_calc': 'Array manifold vector a(theta) on unit sphere. Optimize phi_i to steer.',
            'dark_side': 'Phase phi_i = the dark side. |w_i| = amplitude. You need both.',
        },
        'Directed_Energy': {
            'tech': 'High-power laser, RF weapons (HELLADS, HISAR)',
            'math': 'Complex E field at target: E = E0*exp(j*phi)*exp(-alpha*z).',
            'vector_calc': 'Poynting vector S = (1/2)*E x H*. Power at target = Re[S]*area.',
            'dark_side': 'Atmospheric absorption: k_i != 0. Must account for Im[k] in propagation.',
        },
        'Integrated_Sensing': {
            'tech': 'Radar, SAR (synthetic aperture radar), multistatic sensors',
            'math': 'Radar return: s(t) = A*exp(j*(omega*t + phi_Doppler)). IQ = real + j*imag.',
            'vector_calc': 'Complex signal = I(t) + j*Q(t). Envelope = |s|. Phase = angle(s).',
            'dark_side': 'Q channel IS the dark side. Discarding it = losing Doppler/phase info.',
        },
        'Trusted_AI': {
            'tech': 'Complex-valued neural networks (CVNNs) for radar/comms/photonics',
            'math': 'Complex weight W = |W|*exp(j*angle(W)). Complex activation functions.',
            'vector_calc': 'Complex gradient: dL/dW* (Wirtinger derivative). Same chain rule, complex.',
            'dark_side': 'Wirtinger calculus: dL/dW* != conj(dL/dW) in general. Need both.',
        },
        'HMI': {
            'tech': 'AR/VR headsets, neural interfaces, haptic feedback',
            'math': 'Sensor fusion: complex transfer function H(f) for audio/tactile filtering.',
            'vector_calc': 'Kalman filter: complex state vector x = [pos, vel, phase].',
            'dark_side': 'Phase of signals from IMU + camera + audio must all be tracked.',
        },
        'this_repo': (
            'GS phase retrieval IS applied complex vector calculus:\n'
            '  E(t) in C^N: complex-valued field vector\n'
            '  FT: unitary operator on C^N (complex linear map)\n'
            '  Constraint: project onto {E: |E|^2 = I_meas} -- complex sphere in C^N\n'
            '  Gradient: dL/dE* (Wirtinger derivative of loss w.r.t. complex field)\n'
            '  GS = alternating projections on two complex spheres.\n'
            '  The "dark side" (Im[E]) is what we are trying to recover.'
        ),
    }


def demo():
    print("=== VECTOR CALCULUS: REAL AND COMPLEX (THE DARK SIDE) ===\n")

    print("--- Unit Circle ---")
    uc = unit_circle_trig()
    for d in [0, 30, 45, 90, 180]:
        a = uc['key_angles'][f'{d}deg']
        print(f"  {d:3d} deg: cos={a['cos']:7.4f}  sin={a['sin']:7.4f}  "
              f"exp(j*theta)={a['exp_j_theta']}")
    print(f"  Euler: {uc['eulers_formula']}")

    print("\n--- Unit Sphere ---")
    us = unit_sphere_coordinates()
    print(f"  All orthogonal: {us['all_orthogonal']}")
    for pair, val in us['orthonormality'].items():
        print(f"  {pair} = {val:.2e}")

    print("\n--- Gradient (real + complex) ---")
    gr = gradient_demo()
    print(f"  grad(V={gr['V_real']}) = {gr['grad_V']}")
    print(f"  Complex: {gr['plane_wave_lesson']}")

    print("\n--- Divergence ---")
    dv = divergence_demo()
    print(f"  {dv['point_charge_div_E']}")
    print(f"  Dark side: {dv['complex_lesson'][:100]}...")

    print("\n--- Curl ---")
    cr = curl_demo()
    print(f"  curl(B) of wire z-component: {cr['curl_B_of_wire_z_component']}")
    print(f"  curl(grad(f))=0: {cr['vector_identity_curl_grad_zero']}")
    print(f"  Maxwell phasor: {cr['Maxwell_complex_phasor']['curl_E']}")
    print(f"  Dark side: {cr['dark_side'][:100]}...")

    print("\n--- Laplacian + Spherical Harmonics ---")
    lp = laplacian_demo()
    print(f"  del^2(exp(j*k*z)) = {lp['wave_laplacian']}")
    print(f"  del^2(1/r) = {lp['laplace_1_over_r']}")

    print("\n--- Complex Poynting Vector ---")
    pv = complex_poynting_vector()
    print(f"  {pv['lesson']}")
    print(f"  AC analogy: {pv['AC_analogy'][:80]}...")

    print("\n--- Ray Tracing BVH ---")
    rt = ray_tracing_bvh()
    print(f"  Ray-box hit: {rt['ray_box_hit']}")
    print(f"  Ray-box miss: {rt['ray_box_miss']}")
    print(f"  Reflected direction: {[round(x,3) for x in rt['reflected_direction']]}")
    print(f"  Total internal reflection: {rt['total_internal_reflection']}")
    print(f"  Evanescent k: {rt['evanescent_k']:.4f}j")

    print("\n--- OUSD Alignment ---")
    ousd = ousd_vector_calc_alignment()
    for cta in ['FutureG', 'Directed_Energy', 'Integrated_Sensing']:
        print(f"  {cta}: {ousd[cta]['dark_side'][:80]}...")

    print("\n=== DONE ===")


if __name__ == '__main__':
    demo()
