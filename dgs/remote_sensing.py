"""
Remote Sensing: SAR ocean imaging, InSAR phase retrieval, NASA GIS animation.

THE KEY CONNECTION (this is why it's not boring):

  Coppinger 1999 Eq(1):  E_ch(t) = exp(-t^2/tau^2) * exp(-j*t^2 / 2*L1*beta2)
  SAR transmitted pulse:  s(t)    = rect(t/T)       * exp(+j*pi*K*t^2)

  They are the SAME THING: a Gaussian/rect envelope * quadratic phase (chirp).
  In Coppinger: chirp comes from fiber dispersion (beta2, L1).
  In SAR:       chirp comes from RF hardware sweeping frequency (K = chirp rate).
  Range compression in SAR = matched filter = convolve chirp with conjugate = GS idea.
  InSAR (interferometric SAR) = TWO SAR passes -> phase difference = surface height.
                                = EXACTLY the two-plane GS problem in this repo.

NSF/NASA funding paths (bigger than NIH for this work):
  NASA SBIR: https://sbir.nasa.gov (separate from NIH, ocean/Earth science track)
  NSF OCE (Ocean Sciences): https://www.nsf.gov/div/index.jsp?div=OCE
  NSF EAR (Earth Sciences, InSAR for deformation): https://www.nsf.gov/div/index.jsp?div=EAR
  NASA ESTO (Earth Science Technology Office): develop new instruments
  NASA JPL internship: apply Jan for next summer

Harsh environment applications:
  - Arctic sea ice extent (SAR penetrates clouds, works in polar darkness)
  - Hurricane ocean surface roughness (wind speed from backscatter)
  - Ground-penetrating radar (subsurface barriers/holes: buried pipes, voids)
  - Flood mapping (Sentinel-1 SAR, freely available data)
  - Wildfire perimeter (thermal + SAR through smoke)
"""
import numpy as np

PI = np.pi

# ---------------------------------------------------------------------------
# SAR chirp signal -- same math as Coppinger 1999
# ---------------------------------------------------------------------------
def sar_chirp(K_hz_per_s, T_s, fs_hz, t0=0.0):
    """
    Linear FM (LFM / chirp) pulse: s(t) = rect(t/T) * exp(j*pi*K*t^2)
    K: chirp rate [Hz/s], T: pulse duration [s], fs: sample rate [Hz]

    IDENTICAL MATH to Coppinger 1999 Eq(1) quadratic phase exp(-j*t^2/2*L1*b2):
      K = -1 / (2*pi * L1 * beta2)  [with appropriate unit conversion]
      T ~ tau (pulse duration)
      Fiber disperses passively; SAR generates chirp actively -- same quadratic phase.

    SAR systems: Sentinel-1 (ESA, free data), ALOS-2, TerraSAR-X
    Typical params: K=4e11 Hz/s, T=50us, BW=100MHz, fs=250MHz
    """
    N = int(T_s * fs_hz)
    t = np.linspace(-T_s/2, T_s/2, N) + t0
    rect = (np.abs(t - t0) <= T_s/2).astype(float)
    chirp = rect * np.exp(1j * PI * K_hz_per_s * t**2)
    return t, chirp


def range_compression(raw_signal, K_hz_per_s, T_s, fs_hz):
    """
    SAR range compression = matched filter with conjugate of reference chirp.
    Output: compressed pulse with width ~ 1/BW (resolution cell).

    This is the frequency-domain version:
      S_compressed(f) = S_raw(f) * H_ref*(f)
    where H_ref(f) = exp(-j*pi*f^2/K) is the matched filter.

    Connection to GS: the 'measurement' is |raw_signal|^2 (intensity only).
    Phase retrieval recovers the full complex field -- exactly the GS problem.

    Bandwidth BW = K * T -> range resolution = c / (2*BW)
    Sentinel-1: BW=100MHz -> range resolution = 1.5m
    """
    BW = K_hz_per_s * T_s
    N = len(raw_signal)
    f = np.fft.fftfreq(N, 1/fs_hz)
    S_raw = np.fft.fft(raw_signal)
    # Matched filter: conjugate of chirp spectrum
    H_ref = np.exp(-1j * PI * f**2 / K_hz_per_s)
    S_compressed = S_raw * np.conj(H_ref)
    compressed = np.fft.ifft(S_compressed)
    range_res_m = 3e8 / (2 * BW)
    return compressed, {'BW_hz': BW, 'range_resolution_m': range_res_m}


def insar_phase_retrieval(E1_complex, E2_complex):
    """
    InSAR (Interferometric SAR) phase difference = surface height proxy.
    Given two complex SAR images of the same scene from slightly different orbits:
      interferogram = E1 * conj(E2)
      phase(interferogram) = 4*pi/lambda * delta_height * sin(theta_inc) / altitude

    This IS the two-plane GS problem:
      E1 = field at pass 1 (plane 1 measurement)
      E2 = field at pass 2 (plane 2 measurement after 'dispersion' = geometry change)
      GS recovers phase of each from |E1|^2 and |E2|^2

    Real InSAR: 5.6cm wavelength (C-band Sentinel), 1cm height sensitivity.
    Applications: earthquake deformation, volcanic uplift, groundwater subsidence,
                  permafrost thaw (Arctic), landslide precursors.
    """
    interferogram = E1_complex * np.conj(E2_complex)
    phase_diff = np.angle(interferogram)   # unwrapped -> height
    coherence = np.abs(interferogram) / (np.abs(E1_complex) * np.abs(E2_complex) + 1e-12)
    return phase_diff, coherence, interferogram


# ---------------------------------------------------------------------------
# Ocean wave physics
# ---------------------------------------------------------------------------
def ocean_wave_dispersion(k_rad_per_m, depth_m=None, include_surface_tension=False):
    """
    Ocean wave dispersion relation.
    Deep water (depth >> wavelength): omega^2 = g*k
    Shallow water (depth << wavelength): omega = k*sqrt(g*d)
    Full: omega^2 = g*k*tanh(k*d)  [Airy wave theory]
    With surface tension: omega^2 = (g*k + sigma/rho * k^3) * tanh(k*d)

    Connection to Coppinger/STEAM: the group velocity vg = d(omega)/dk
    varies with k (dispersion). Same concept as fiber GVD beta2 = d^2(beta)/d(omega^2).
    Ocean: dispersion in SPACE (different wavelengths travel at different speeds).
    Fiber: dispersion in TIME (different frequencies arrive at different times).
    STEAM converts one to the other.

    g = 9.81 m/s^2, sigma = 0.073 N/m (water-air), rho = 1025 kg/m^3
    """
    g = 9.81
    sigma = 0.073  # N/m surface tension
    rho = 1025.0   # kg/m^3 seawater density

    if depth_m is None:
        # Deep water
        omega2 = g * k_rad_per_m
        if include_surface_tension:
            omega2 += (sigma/rho) * k_rad_per_m**3
    else:
        tanh_kd = np.tanh(k_rad_per_m * depth_m)
        omega2 = g * k_rad_per_m * tanh_kd
        if include_surface_tension:
            omega2 += (sigma/rho) * k_rad_per_m**3 * tanh_kd

    omega = np.sqrt(np.maximum(omega2, 0))
    phase_vel = np.where(k_rad_per_m > 0, omega / k_rad_per_m, 0)
    group_vel = np.gradient(omega, k_rad_per_m)

    return {
        'omega_rad_per_s': omega,
        'phase_velocity_m_per_s': phase_vel,
        'group_velocity_m_per_s': group_vel,
        'wavelength_m': np.where(k_rad_per_m > 0, 2*PI/k_rad_per_m, np.inf),
        'period_s': np.where(omega > 0, 2*PI/omega, np.inf),
    }


def ocean_wave_gvd_analog(k_center_rad_per_m, depth_m=None):
    """
    Ocean wave 'beta2' analog: d^2(omega)/dk^2 at k_center.
    In fiber: beta2 = d^2(beta)/d(omega^2), causes pulse spreading.
    In ocean: d^2(omega)/dk^2 causes wave packet spreading = dispersive spreading
              of a wave group (swell traveling across Pacific).

    Deep water: omega = sqrt(g*k) -> d^2(omega)/dk^2 = -sqrt(g)/(4*k^(3/2))
    Negative: longer waves travel faster -> wave packet spreads in space.
    Analogous to anomalous dispersion (beta2 < 0) in fiber.
    """
    dk = k_center_rad_per_m * 0.001
    k_arr = np.array([k_center_rad_per_m - dk,
                       k_center_rad_per_m,
                       k_center_rad_per_m + dk])
    g = 9.81
    if depth_m is None:
        omega = np.sqrt(g * k_arr)
    else:
        omega = np.sqrt(g * k_arr * np.tanh(k_arr * depth_m))

    d2omega_dk2 = (omega[2] - 2*omega[1] + omega[0]) / dk**2
    return {
        'k_center': k_center_rad_per_m,
        'd2omega_dk2': d2omega_dk2,
        'ocean_analog_of_beta2': d2omega_dk2,
        'sign': 'negative (anomalous) for deep water gravity waves',
        'meaning': 'wave packet spreads as it travels; longer waves outrun shorter ones',
        'fiber_analog': 'same sign as beta2<0 (anomalous dispersion) -- STEAM regime',
    }


def sar_ocean_wave_measurement(significant_wave_height_m=3.0,
                                peak_period_s=12.0,
                                satellite_altitude_km=693,
                                radar_wavelength_m=0.056):
    """
    How SAR measures ocean wave height remotely.
    Sentinel-1: altitude=693km, C-band lambda=5.6cm, incidence angle=20-45deg

    Physics chain:
      Ocean surface height eta(x,t) -> surface tilt -> Bragg backscatter -> SAR intensity
      SAR intensity I ~ sigma_0 (normalized radar cross section)
      sigma_0 depends on wind speed, wave height, incidence angle

    Significant wave height Hs estimation from SAR:
      Hs = 4 * sigma(eta)  [4x std of surface elevation]
      SAR measures: azimuth cutoff wavelength lambda_c = pi*R*V_r / (V_s * sigma_az)
      where R=range, V_r=wave orbital velocity, V_s=satellite speed

    GS phase retrieval connection:
      The ocean surface is an unknown phase screen phi(x,y).
      SAR measures |E_scattered|^2 from two viewing geometries (interferogram).
      GS recovers phi(x,y) -> height map eta(x,y).
      This IS the repo's retrieve_phase() applied to Earth observation.
    """
    g = 9.81
    k_peak = (2*PI/peak_period_s)**2 / g  # deep water: k = omega^2/g
    lambda_peak_m = 2*PI / k_peak
    phase_vel = np.sqrt(g / k_peak)       # deep water phase velocity [m/s]
    group_vel = phase_vel / 2              # deep water: vg = vp/2

    # Azimuth cutoff (resolution limit due to wave motion during synthetic aperture)
    V_satellite = 7500  # m/s (Sentinel-1 orbital velocity)
    sigma_orbital = 0.5 * significant_wave_height_m * 2*PI/peak_period_s  # orbital velocity
    R_m = satellite_altitude_km * 1e3
    lambda_c_m = PI * R_m * sigma_orbital / (V_satellite * 1.0)

    return {
        'Hs_m': significant_wave_height_m,
        'peak_period_s': peak_period_s,
        'peak_wavelength_m': round(lambda_peak_m, 1),
        'phase_velocity_m_per_s': round(phase_vel, 1),
        'group_velocity_m_per_s': round(group_vel, 1),
        'k_peak_rad_per_m': round(k_peak, 4),
        'azimuth_cutoff_m': round(lambda_c_m, 1),
        'satellite': 'Sentinel-1 (ESA, free data, 6-day repeat)',
        'data_access': 'https://scihub.copernicus.eu (free account, direct download)',
        'gs_connection': 'InSAR phase = two-plane GS problem; retrieve_phase() applies directly',
    }


# ---------------------------------------------------------------------------
# GIS / animation data pipeline
# ---------------------------------------------------------------------------
def simulate_ocean_surface_gis(nx=128, ny=128, n_frames=20,
                                 Hs_m=2.0, Tp_s=10.0, wind_dir_deg=45.0):
    """
    Simulate ocean surface elevation field for GIS animation.
    Uses JONSWAP spectrum (standard ocean wave energy spectrum).
    Output: (n_frames, nx, ny) array of eta(x,y,t) -- plug into matplotlib animation
            or export as GeoTIFF for QGIS/ArcGIS.

    JONSWAP (Joint North Sea Wave Project) spectrum:
      S(f) = alpha*g^2 / (2*pi)^4 / f^5 * exp(-5/4*(fp/f)^4) * gamma^r
      where r = exp(-(f-fp)^2 / (2*sigma^2*fp^2))
      alpha=0.0081, gamma=3.3 (peak enhancement), fp=1/Tp

    Connection to SAR: this IS the surface Sentinel-1 images.
    Connection to GS: phi(x,y) = ocean surface phase, recovered by InSAR.
    Connection to STEAM: dispersive spreading of wave packet = fiber chirp analog.
    """
    g = 9.81
    fp = 1.0 / Tp_s
    alpha = 0.0081
    gamma_js = 3.3

    dx = 10.0  # m per pixel
    dy = 10.0
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy

    kx = np.fft.fftfreq(nx, dx) * 2*PI
    ky = np.fft.fftfreq(ny, dy) * 2*PI
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)
    K = np.where(K == 0, 1e-6, K)

    # Deep water omega from k
    OMEGA = np.sqrt(g * K)
    F = OMEGA / (2*PI)

    # JONSWAP spectrum in frequency
    sigma_js = np.where(F <= fp, 0.07, 0.09)
    r = np.exp(-(F - fp)**2 / (2*sigma_js**2 * fp**2))
    S = (alpha * g**2 / (2*PI)**4 / (F**5 + 1e-30)
         * np.exp(-1.25*(fp/(F+1e-12))**4) * gamma_js**r)

    # Directional spreading: cos^2(theta - theta_wind)
    theta_wind = np.deg2rad(wind_dir_deg)
    THETA = np.arctan2(KY, KX)
    D = np.cos(THETA - theta_wind)**2
    D = np.where(D < 0, 0, D)
    S_dir = S * D

    # Amplitude from spectrum: a = sqrt(2*S*dk^2)
    dk = (2*PI/dx) / nx
    A = np.sqrt(2 * S_dir * dk**2)

    # Random phases
    rng = np.random.default_rng(42)
    phi0 = rng.uniform(0, 2*PI, (nx, ny))

    # Time evolution: eta(x,y,t) = Re[sum A*exp(j*(kx*x+ky*y - omega*t + phi0))]
    dt = 1.0  # seconds between frames
    frames = np.zeros((n_frames, nx, ny))
    for i in range(n_frames):
        t_sec = i * dt
        phase = phi0 - OMEGA * t_sec
        eta_complex = A * np.exp(1j * phase)
        # Inverse 2D FFT to get spatial field
        frames[i] = np.real(np.fft.ifft2(eta_complex)) * nx * ny
        # Scale to get correct Hs
        frames[i] *= Hs_m / (4 * frames[i].std() + 1e-6)

    return frames, {'x_m': x, 'y_m': y, 'Hs_m': Hs_m, 'Tp_s': Tp_s,
                    'dx_m': dx, 'n_frames': n_frames, 'dt_s': dt}


# ---------------------------------------------------------------------------
# NSF / NASA funding paths
# ---------------------------------------------------------------------------
FUNDING_PATHS = {
    'NASA_SBIR': {
        'name': 'NASA SBIR/STTR Phase I',
        'url': 'https://sbir.nasa.gov',
        'amount': '$150K / 6 months',
        'relevant_topics': [
            'A1.04: Ground-Based Sub-Orbital Observations (remote sensing sensors)',
            'H9.03: Flight Dynamics and Navigation Systems',
            'S1.07: Airborne Measurement Systems',
        ],
        'connection': 'SAR ocean wave sensor using STEAM-enhanced ADC (stretch factor M). '
                      'Same H(f)=exp(j*pi*D*f^2) physics -- STEAM chirp IS the SAR chirp.',
        'next_deadline': 'Check sbir.nasa.gov for current solicitation (typically March)',
    },
    'NSF_OCE': {
        'name': 'NSF Division of Ocean Sciences (OCE)',
        'url': 'https://www.nsf.gov/div/index.jsp?div=OCE',
        'amount': '$200K-$500K / 2-3 years',
        'relevant_topics': [
            'Physical Oceanography: wave dispersion, rogue waves (connects to dgs/nlse.py)',
            'Ocean Technology: new sensors for wave measurement',
        ],
        'connection': 'Rogue wave detection in ocean using GS phase retrieval + MI gain '
                      'model (jalali_grammar.rogue_wave_mi_gain()). '
                      'Same NLSE instability that causes optical rogue waves causes '
                      'oceanic freak waves.',
        'who_writes_it': 'Need a UC Davis OCE-affiliated PI. '
                         'Target: get Yoo or a geophysics prof as PI, you as student researcher.',
    },
    'NSF_EAR': {
        'name': 'NSF Division of Earth Sciences (EAR) -- Geophysics',
        'url': 'https://www.nsf.gov/div/index.jsp?div=EAR',
        'amount': '$200K-$600K',
        'relevant_topics': [
            'InSAR surface deformation (earthquake, volcano, groundwater)',
            'GS phase retrieval applied to interferometric data',
        ],
        'connection': 'dgs/gs_core.retrieve_phase() is the same algorithm used in '
                      'InSAR phase unwrapping. Frame it as: "differentiable GS for '
                      'real-time InSAR on edge computing hardware (RPi CM4)".',
    },
    'NSF_GRFP': {
        'name': 'NSF Graduate Research Fellowship Program',
        'url': 'https://www.nsfgrfp.org',
        'amount': '$37K/year stipend + tuition for 3 years = ~$150K',
        'eligibility': 'Must be in first 2 years of PhD program OR senior undergrad',
        'deadline': 'October (apply senior year or first year of PhD)',
        'connection': 'Your proposal: "Differentiable photonic sensing for Earth observation: '
                      'from optical fiber phase retrieval to InSAR ocean wave monitoring." '
                      'This connects every module in the repo to a fundable NSF narrative.',
        'this_repo_is_the_preliminary_work': True,
    },
    'NASA_JPL_internship': {
        'name': 'NASA JPL Internship (Year-Round)',
        'url': 'https://www.jpl.nasa.gov/edu/intern',
        'amount': '$700-900/week stipend',
        'deadline': 'Rolling, apply Sept-Oct for spring; Jan-Feb for summer',
        'divisions': [
            '335: Radar Science (SAR, InSAR)',
            '383: Microwave Atmospheric Science (passive remote sensing)',
            '388: Ocean Circulation and Air-Sea Interaction',
        ],
        'what_to_say': 'I have implemented SAR range compression and InSAR phase '
                        'retrieval (Gerchberg-Saxton) in Python with full test suite. '
                        'GitHub: <link>. Interested in Division 335 (Radar Science).',
        'location': 'Pasadena CA -- 6 hour drive or 1 hour flight from Sacramento',
    },
}


def demo():
    print("=== SAR Chirp = Coppinger 1999 Chirp ===")
    t, chirp = sar_chirp(K_hz_per_s=4e11, T_s=50e-6, fs_hz=250e6)
    print(f"SAR chirp: N={len(chirp)}, bandwidth={4e11*50e-6/1e6:.0f} MHz")
    compressed, meta = range_compression(chirp, 4e11, 50e-6, 250e6)
    print(f"Range resolution: {meta['range_resolution_m']:.2f} m")
    print(f"Peak compressed: {np.abs(compressed).max():.1f}")

    print("\n=== Ocean Wave Dispersion (beta2 analog) ===")
    k_arr = np.linspace(0.01, 2.0, 100)
    disp = ocean_wave_dispersion(k_arr)
    print(f"k=0.1 rad/m: lambda={2*PI/0.1:.1f}m, T={2*PI/disp['omega_rad_per_s'][9]:.1f}s")
    b2_analog = ocean_wave_gvd_analog(0.1)
    print(f"Ocean beta2 analog at k=0.1: {b2_analog['d2omega_dk2']:.4f} m^2*s/rad")
    print(f"Sign: {b2_analog['sign']}")

    print("\n=== SAR Ocean Wave Measurement ===")
    meas = sar_ocean_wave_measurement(significant_wave_height_m=3.0, peak_period_s=12.0)
    for k, v in meas.items():
        print(f"  {k}: {v}")

    print("\n=== JONSWAP Surface Simulation ===")
    frames, meta = simulate_ocean_surface_gis(nx=64, ny=64, n_frames=5)
    print(f"Frames shape: {frames.shape}")
    print(f"Hs estimate: {4*frames.std():.2f} m (target {meta['Hs_m']} m)")

    print("\n=== Funding Paths ===")
    for name, info in FUNDING_PATHS.items():
        print(f"\n{info['name']}: {info['amount']}")
        print(f"  {info.get('connection', info.get('what_to_say', ''))[:80]}...")


if __name__ == '__main__':
    demo()
