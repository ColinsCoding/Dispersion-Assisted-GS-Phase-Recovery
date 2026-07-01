"""
Soft Matter Manufacturing and Microfluidics

Lab-on-chip: squeeze biology and chemistry into channels 10-200 um wide.
Physics completely different from macro scale: viscosity dominates inertia.
Manufacturing: PDMS soft lithography -- photomask + spin coating + replica molding.

Apple products in the wet lab:
  iPhone LiDAR: measures channel depth to +/-1mm (not enough for um-scale chips)
  iPhone camera + OpenCV: droplet counting, colony sizing, cell tracking
  iPad: real-time data display, remote instrument control via Bluetooth
  Vision Pro: overlay microscope feed with analysis in AR (research labs 2024+)

Jalali group connection:
  STEAM camera (serial time-encoded amplified microscopy) images flowing cells
  at 36.7 million frames/second -- impossible with CCD/CMOS.
  Same H(f)=exp(j*pi*D*f^2) dispersion as this repo: time-stretch maps
  spatial information onto a stretched waveform then digitizes.
  Microfluidics + STEAM = ultrafast cell imaging for cancer screening.
  This IS the direct application of Jalali's lab and this codebase.
"""
import numpy as np
import sympy as sp


# Physical constants
MU_WATER = 1.002e-3    # Pa*s dynamic viscosity at 20C
RHO_WATER = 998.0      # kg/m^3
GAMMA_WATER_AIR = 0.072  # N/m surface tension


def reynolds_number(rho, v, L, mu):
    """
    Re = rho*v*L/mu -- ratio of inertial to viscous forces.

    Re << 1: viscous dominates -> laminar, reversible, Stokes flow
    Re >> 1: inertia dominates -> turbulent, chaotic
    Re_crit ~ 2300 for pipe flow (laminar-turbulent transition)

    Microfluidics: Re ~ 0.001 to 10 (ALWAYS laminar).
    Consequence: NO turbulent mixing. Mixing is purely diffusive.
    This is both a challenge (need to design mixing) and advantage (precise control).

    Same dimensionless group in photonics:
      Fresnel number N_F = a^2/(lambda*L) -- ratio of aperture^2 to wavelength*distance
      N_F >> 1: geometric optics (ray tracing)
      N_F << 1: diffraction dominates (Fraunhofer = far field = FT regime)
    """
    Re = rho * v * L / mu
    regime = ('turbulent' if Re > 4000
              else 'transitional' if Re > 2300
              else 'laminar')
    return {
        'Re': Re,
        'regime': regime,
        'mixing': 'diffusive only (no turbulence)' if Re < 1 else 'convective possible',
        'reversible': Re < 1,
        'lesson': (f'Re={Re:.3f}: {regime}. '
                   f'Microfluidics Re~0.001-10 -> always laminar. '
                   f'Mixing requires design (herringbone mixer, droplets, electroosmosis).'),
    }


def stokes_flow_channel(w_um, h_um, mu=MU_WATER, dP_Pa_per_m=1e5):
    """
    Pressure-driven flow in rectangular channel (Stokes / Poiseuille).

    For w >> h (wide channel): parabolic profile, max velocity at center.
      u_max = h^2 * dP / (8 * mu)    [circular cross-section: h=R, factor 8]
      u_avg = h^2 * dP / (12 * mu)   [rectangular, wide limit]
      Q = u_avg * w * h               [volumetric flow rate]

    For general rectangle: exact series solution.
    We use the approximate formula valid for h < w/3.

    Stokes equation: mu * d^2u/dy^2 = dP/dx  (no inertia term rho*Du/Dt)
    ODE: d^2u/dy^2 = (1/mu) * dP/dx  -> parabola.
    Boundary conditions: u=0 at y=0 and y=h (no-slip walls).

    Hydraulic resistance: R_hyd = dP / Q  [Pa*s/m^3]
    Electrical analogy: dP <-> V, Q <-> I, R_hyd <-> R.
    Networks of channels = resistor networks. Same KVL/KCL.
    """
    if w_um <= 0 or h_um <= 0:
        raise ValueError("Channel dimensions must be positive")
    w = w_um * 1e-6; h = h_um * 1e-6
    dP = dP_Pa_per_m   # Pa/m

    # Wide-channel (h < w) Poiseuille approximation
    u_max = h**2 * dP / (8 * mu)
    u_avg = h**2 * dP / (12 * mu)
    Q = u_avg * w * h

    # Hydraulic resistance per unit length [Pa*s/m^4]
    R_hyd_per_m = dP / Q if Q > 0 else float('inf')

    # Reynolds number
    L_hyd = 2*w*h / (w + h)   # hydraulic diameter
    rho_water = RHO_WATER
    Re = rho_water * u_avg * L_hyd / mu

    # Velocity profile: parabola u(y) = u_max * 4y(h-y)/h^2
    y = np.linspace(0, h, 100)
    u_profile = u_max * 4*y*(h-y) / h**2

    return {
        'u_max_mm_per_s': u_max * 1e3,
        'u_avg_mm_per_s': u_avg * 1e3,
        'Q_nL_per_s': Q * 1e12,   # nL/s
        'R_hyd_per_m': R_hyd_per_m,
        'Re': Re,
        'regime': 'laminar (Stokes)' if Re < 100 else 'laminar-inertial',
        'y_um': y * 1e6,
        'u_profile_mm_per_s': u_profile * 1e3,
        'electrical_analogy': (
            'dP <-> Voltage, Q <-> Current, R_hyd <-> Resistance. '
            'Series channels: R_total = sum(R_i). '
            'Parallel: 1/R_total = sum(1/R_i). Same KCL/KVL.'
        ),
    }


def capillary_number(mu, v, gamma=GAMMA_WATER_AIR):
    """
    Ca = mu*v/gamma -- ratio of viscous to surface tension forces.

    Ca << 1: surface tension dominates -> droplets form (microfluidic droplet generator)
    Ca >> 1: viscous dominates -> continuous flow (no droplets)
    Ca ~ 0.01-0.1: droplet formation transition

    Droplet microfluidics:
      Oil phase (continuous) + water phase (dispersed) -> droplets
      Each droplet = isolated reaction vessel, picoliter to nanoliter
      Application: single-cell sequencing (10x Genomics, Drop-seq), drug screening

    Weber number: We = rho*v^2*L/gamma (inertia vs surface tension)
      We << 1 and Ca << 1: surface tension controls everything.
    """
    Ca = mu * v / gamma
    regime = ('droplet_formation' if Ca < 0.1
              else 'transitional' if Ca < 1.0
              else 'continuous_flow')
    return {
        'Ca': Ca,
        'regime': regime,
        'application': ('picoliter_droplets -> single-cell-seq' if Ca < 0.1
                        else 'continuous flow chemistry'),
        'lesson': (f'Ca={Ca:.4f}. '
                   f'Droplet microfluidics needs Ca<0.1. '
                   f'10x Genomics GemCode = 10^5 droplets/sec at Ca~0.001.'),
    }


def diffusion_timescale(D_m2_per_s, L_um):
    """
    Diffusion time: tau = L^2 / (2*D)
    Diffusion length: x_rms = sqrt(2*D*t)

    In microchannels, mixing ONLY by diffusion (Re << 1).
    Design rule: channel must be short enough that tau << residence time.
    OR use chaotic advection (herringbone grooves) to enhance mixing.

    Diffusion coefficients (water, 25C):
      Small molecule (MW~100):  D ~ 5e-10 m^2/s  -> tau(10um) = 0.1ms
      Protein (MW~50kDa):       D ~ 8e-11 m^2/s  -> tau(10um) = 0.6ms
      DNA (10kbp):              D ~ 1e-12 m^2/s  -> tau(10um) = 50ms
      Microparticle (1um):      D ~ 5e-13 m^2/s  -> tau(10um) = 100ms

    Peclet number: Pe = v*L/D
      Pe >> 1: convection dominates (fluid carries species faster than diffusion)
      Pe << 1: diffusion dominates (rapidly mixed)
    """
    if D_m2_per_s <= 0 or L_um <= 0:
        raise ValueError("D and L must be positive")
    L = L_um * 1e-6
    tau = L**2 / (2*D_m2_per_s)

    # Diffusion length vs time
    t = np.logspace(-6, 2, 200)   # 1 us to 100 s
    x_rms = np.sqrt(2*D_m2_per_s*t) * 1e6   # in um

    species = {
        'small_molecule_100Da':  5e-10,
        'protein_50kDa':        8e-11,
        'DNA_10kbp':            1e-12,
        'microparticle_1um':    5e-13,
    }
    timescales = {name: L**2/(2*D) for name, D in species.items()}

    return {
        'tau_s': tau,
        'tau_ms': tau * 1e3,
        'x_rms_um_at_1ms': float(np.sqrt(2*D_m2_per_s*1e-3) * 1e6),
        'x_rms_um_at_1s':  float(np.sqrt(2*D_m2_per_s*1.0)  * 1e6),
        't_s': t,
        'x_rms_um': x_rms,
        'species_timescales_s': timescales,
        'lesson': (
            'tau = L^2/(2D): quadratic in size. '
            '10x smaller channel = 100x faster mixing. '
            'Microfluidics exploits this: 10um channel, 1ms mixing vs 1s in a test tube.'
        ),
    }


def pdms_fabrication_protocol():
    """
    PDMS (polydimethylsiloxane) soft lithography -- the standard microfluidics fab process.

    PDMS: silicone rubber, transparent, biocompatible, oxygen-permeable (good for cells),
          bonds to glass via O2 plasma treatment, castable from molds.

    Steps:
    1. DESIGN: Draw channel layout in AutoCAD / KLayout / Inkscape (DXF/SVG).
               Design rules: min feature 10um (standard), 5um (high-res litho).
               Account for PDMS shrinkage: ~1-2% linear.

    2. PHOTOMASK: Print design on transparency (>50um features) or chrome mask (<10um).
               Cost: transparency ~$5, chrome mask ~$200.

    3. SU-8 MOLD (Silicon wafer + SU-8 photoresist):
       a. Spin coat SU-8 on 3-inch Si wafer: thickness = channel height (10-200um)
          Spin speed: 3000 rpm -> ~30um; 500 rpm -> ~200um
       b. Soft bake: 65C 2min, 95C 10min (evaporate solvent)
       c. UV expose through mask: ~200 mJ/cm^2 (crosslinks exposed SU-8)
       d. Post-exposure bake: 65C 1min, 95C 5min (completes crosslinking)
       e. Develop in SU-8 developer (PGMEA): 5-10 min, isopropanol rinse
       f. Hard bake: 150C 30min (optional, increases durability)
       Result: SU-8 channels on Si wafer (positive mold = channel walls are raised)

    4. PDMS CASTING:
       Mix 10:1 base:curing agent by weight (Sylgard 184, Dow Corning)
       Degas in desiccator/vacuum: 30-60 min (removes bubbles)
       Pour over SU-8 mold: ~5mm thick
       Cure: 65-80C for 1-4 hours (chemical crosslinking)
       Peel off: razor blade around perimeter, slow peel
       Punch inlets: 0.75mm or 1mm biopsy punch

    5. BONDING to glass:
       O2 plasma: 30W, 30s (activates Si-OH groups on both surfaces)
       Press together within 60s: irreversible covalent Si-O-Si bond
       Bake: 80C 30min to strengthen bond

    Cost estimate (UC lab or CSUS fab):
       SU-8 mold: $50-200 (wafer + chemicals, one-time per design)
       Per chip: ~$0.50 (PDMS, glass slide)
       Photomask: $5 (transparency) or $200 (chrome for <10um)

    Apple products:
       iPhone camera + Python/OpenCV: image chips, measure channel dimensions
       iPad + LabVIEW Mobile: control syringe pump, log pressure data
       iPhone LiDAR: rough wafer topography (~1mm res, not useful for chips)
    """
    return {
        'material': 'PDMS (Sylgard 184), 10:1 base:curing agent',
        'transparency': True,
        'biocompatible': True,
        'O2_permeable': True,
        'steps': [
            '1. Design in AutoCAD/Inkscape (min feature ~10um)',
            '2. Print photomask ($5 transparency or $200 chrome)',
            '3. SU-8 mold: spin coat, UV expose, develop (1 day)',
            '4. Cast PDMS 10:1, degas, cure 65C 1hr, peel',
            '5. O2 plasma bond to glass slide, bake 80C 30min',
        ],
        'cost_per_chip_usd': 0.50,
        'cost_mold_usd': 150.0,
        'min_feature_um': 10.0,
        'channel_height_range_um': (10, 200),
        'steam_camera_connection': (
            'Jalali group STEAM camera: microfluidic channel flows cells past a '
            'single-pixel detector at 36.7 Mfps. '
            'Impossible with CCD. Only possible because time-stretch (H(f)=exp(j*pi*D*f^2)) '
            'maps each spatial position to a different time delay. '
            'This codebase = the phase retrieval inverse of that forward process.'
        ),
        'apple_lab_use': {
            'iPhone_camera': 'OpenCV cell counting, droplet sizing, colony area',
            'iPad': 'Syringe pump control, real-time pressure/flow logging',
            'Vision_Pro': 'AR overlay of analysis on microscope field (research, 2024)',
            'iPhone_LiDAR': 'Rough wafer topography, NOT useful for um-scale chips',
        },
    }


def droplet_generator(Q_oil_uL_per_min, Q_water_uL_per_min,
                       w_channel_um=100.0, h_um=50.0,
                       gamma=GAMMA_WATER_AIR, mu_oil=5e-3):
    """
    T-junction droplet generator: oil (continuous) + water (dispersed).
    Water channel meets oil channel at 90 degrees -> periodic droplet pinch-off.

    Droplet size scaling: d/w ~ (Q_water/Q_oil)^(1/3)  [approximate]
    Droplet frequency: f = Q_total / V_droplet

    Used in:
      - Single-cell RNA sequencing (10x Genomics): each cell encapsulated in a droplet
      - Drug screening: 10^6 reactions in 1 hour (vs 10^4 in 96-well plate)
      - Digital PCR: count DNA molecules (one per droplet)
      - Synthesis: millifluidic synthesis of nanoparticles with tight size distribution
    """
    if Q_oil_uL_per_min <= 0 or Q_water_uL_per_min <= 0:
        raise ValueError("Flow rates must be positive")
    Q_oil   = Q_oil_uL_per_min   * 1e-6/60   # m^3/s
    Q_water = Q_water_uL_per_min * 1e-6/60

    w = w_channel_um * 1e-6; h = h_um * 1e-6
    v_oil = Q_oil / (w * h)
    Ca_oil = mu_oil * v_oil / gamma

    # Empirical droplet size (Garstecki model)
    flow_ratio = Q_water / Q_oil
    d_over_w = 1 + flow_ratio   # Garstecki 2006: d/w ~ 1 + Q_w/Q_c (squeezing regime)
    d_um = d_over_w * w_channel_um
    V_droplet_pL = (4/3) * np.pi * (d_um/2*1e-6)**3 * 1e15   # pL

    Q_total = Q_oil + Q_water
    f_Hz = Q_total / (V_droplet_pL * 1e-15) if V_droplet_pL > 0 else 0

    return {
        'Ca_oil': Ca_oil,
        'flow_ratio_water_to_oil': flow_ratio,
        'droplet_diameter_um': d_um,
        'droplet_volume_pL': V_droplet_pL,
        'frequency_Hz': f_Hz,
        'regime': 'squeezing' if Ca_oil < 0.01 else 'dripping' if Ca_oil < 0.1 else 'jetting',
        'throughput': f'{f_Hz:.0f} droplets/sec = {f_Hz*3600:.0f}/hour',
        'application': (
            '10x Genomics: each droplet = 1 cell + 1 bead + lysis buffer. '
            'Cell barcode + UMI -> single-cell transcriptome. '
            '~10^4 cells per run. Same chip design as this function.'
        ),
    }


def atomic_emission_spectra():
    """
    Fireworks and atomic spectroscopy: same physics as neon signs, lasers, LEDs.

    Mechanism:
      1. Chemical energy heats atoms to excited electronic states (n=2,3,4...)
      2. Spontaneous emission: E_upper -> E_lower + hv
         E_photon = E_upper - E_lower = hf = hc/lambda
      3. Each element has unique lambda -> unique color (spectral fingerprint)

    Fireworks color chart:
      Red:    Sr salts   (SrCO3, SrCl2)    lambda ~ 640-700nm
      Orange: Ca salts   (CaCl2, CaCO3)    lambda ~ 600-640nm
      Yellow: Na salts   (NaCl, NaNO3)     lambda ~ 589nm (D-line doublet)
      Green:  Ba salts   (BaCl2, BaNO3)    lambda ~ 500-540nm
      Blue:   Cu salts   (CuCl, CuSO4)     lambda ~ 430-480nm
      White:  Mg, Al, Ti (blackbody + line) broadband

    Sodium D-line (589nm):
      3p -> 3s transition in Na atom.
      Same transition used in Na street lamps.
      Same physics as: laser gain (stimulated emission), LED (electroluminescence).
      Connection to photonics: narrow-linewidth Na laser = frequency reference.

    Nuclear connection:
      Fireworks use CHEMICAL (electron) transitions: eV energy scale.
      Nuclear transitions: MeV energy scale (10^6x more energetic) -> gamma rays.
      Nuclear fireworks = gamma-ray bursts (astrophysics), not visible.
      Neutron activation: stable nucleus + n -> excited nucleus -> gamma + new isotope.
      Same photon physics, different energy scale.
    """
    h = 6.626e-34; c = 2.998e8; eV = 1.602e-19

    colors = {
        'red_Sr':    {'lambda_nm': 670, 'element': 'Sr', 'transition': '5p->5s'},
        'orange_Ca': {'lambda_nm': 622, 'element': 'Ca', 'transition': '4p->4s'},
        'yellow_Na': {'lambda_nm': 589, 'element': 'Na', 'transition': '3p->3s (D-line)'},
        'green_Ba':  {'lambda_nm': 524, 'element': 'Ba', 'transition': '6p->6s'},
        'blue_Cu':   {'lambda_nm': 450, 'element': 'Cu', 'transition': '4p->4s'},
    }
    for name, data in colors.items():
        lam = data['lambda_nm'] * 1e-9
        data['E_photon_eV'] = h*c/lam / eV
        data['frequency_THz'] = c/lam / 1e12

    return {
        'colors': colors,
        'na_D_line_note': (
            'Na 589nm = most efficient emitter (high oscillator strength). '
            'Used as frequency reference for optical clocks. '
            'Same 3p->3s quantum transition in NaD laser.'
        ),
        'nuclear_vs_chemical': (
            'Chemical (electron) transition: eV range -> UV/Vis/IR photons.\n'
            'Nuclear (proton/neutron) transition: MeV range -> gamma rays, X-rays.\n'
            'Radioactive decay: nucleus emits alpha/beta/gamma to reach lower energy state.\n'
            'Same Bohr-model intuition: quantized energy levels, emission when falling down.'
        ),
        'laser_connection': (
            'Laser = stimulated emission (forced), fireworks = spontaneous emission.\n'
            'Einstein A coefficient (spontaneous) and B coefficient (stimulated):\n'
            '  A/B = 8*pi*h*f^3/c^3  (spontaneous/stimulated ratio)\n'
            'At optical frequencies A >> B -> lasers need population inversion to win.\n'
            'Microwave masers: f lower -> B >> A -> easy population inversion.'
        ),
        'jalali_ultrafast': (
            'STEAM camera images 36.7 Mfps via time-stretch.\n'
            'Can image individual laser pulses hitting a screen.\n'
            'Or individual droplets in a microfluidic channel (cell screening).\n'
            'Or individual firework sparks at ~microsecond timescale.\n'
            'H(f)=exp(j*pi*D*f^2) maps each moment in time to a different wavelength.'
        ),
    }


def linux_lab_server_setup():
    """
    Linux boot process and headless lab server setup.
    Running this repo on a headless Linux box (Raspberry Pi / NUC / GPU server).

    BOOT SEQUENCE:
      1. Power on -> BIOS/UEFI POST (hardware check)
      2. Bootloader: GRUB2 loads /boot/vmlinuz (kernel) + /boot/initrd.img (initramfs)
      3. Kernel: hardware init, mount root filesystem
      4. systemd (PID 1): reads unit files, brings up services
      5. Login prompt (getty) or SSH daemon (sshd) -> your session

    KEY FILES:
      /boot/vmlinuz-*     kernel binary
      /boot/initrd.img-*  initial ramdisk (minimal filesystem for boot)
      /etc/systemd/       unit files
      /etc/fstab          mount points
      /etc/hosts          local DNS (where adblock_hosts.py writes)

    FOR THIS REPO (headless GPU server):
      sudo apt install python3 python3-pip jupyter-lab
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
      pip install -e .  (installs dgs/ + griffiths/ packages via pyproject.toml)
      jupyter lab --no-browser --port=8888 --ip=0.0.0.0
      # On your Mac/Windows: ssh -L 8888:localhost:8888 user@server-ip
      # Then open http://localhost:8888 in browser

    SYSTEMD SERVICE (auto-start Jupyter on boot):
      /etc/systemd/system/jupyter.service
      ExecStart=/usr/bin/jupyter lab --no-browser --port=8888 --ip=0.0.0.0
      sudo systemctl enable jupyter && sudo systemctl start jupyter

    NETWORKING:
      ip addr show       # see all network interfaces
      ssh-keygen         # generate SSH key
      ssh-copy-id user@server  # copy public key -> passwordless login
      ufw allow 22/tcp   # firewall: allow SSH only
      ufw allow 8888/tcp # allow Jupyter (or tunnel via SSH instead)

    RPi (RogueGuard hardware, same board):
      sudo apt install raspi-config
      raspi-config -> enable SSH, SPI, I2C (for ADC boards)
      gpio readall   # check pin states
      pigpio + Python: precise timing for ADC sampling
    """
    return {
        'boot_sequence': [
            'BIOS/UEFI POST -> hardware check',
            'GRUB2 -> loads vmlinuz + initrd.img',
            'Kernel -> hardware init, root fs mount',
            'systemd (PID 1) -> services, daemons',
            'sshd / getty -> your login',
        ],
        'loading_screen': 'Loading screen = initramfs + systemd targets completing in sequence',
        'jupyter_setup': [
            'pip install -e .  (dgs/griffiths packages)',
            'jupyter lab --no-browser --ip=0.0.0.0 --port=8888',
            'ssh -L 8888:localhost:8888 user@server  (tunnel)',
        ],
        'systemd_service': (
            '[Unit] Description=Jupyter Lab\n'
            '[Service] ExecStart=jupyter lab --no-browser --ip=0.0.0.0\n'
            '[Install] WantedBy=multi-user.target\n'
            'sudo systemctl enable jupyter'
        ),
        'rpi_for_rogueguard': (
            'RPi CM4: same board as RogueGuard optical rogue wave monitor.\n'
            'ADC via SPI: pigpio + spidev for 1MHz+ sampling.\n'
            'Run TD-GS: dgs/gs_core.py on ARM64 -- no GPU needed for 1D GS.\n'
            'CNN inference: ONNX Runtime on RPi CPU (~10ms per frame).'
        ),
    }


def group_z2_binary():
    """
    The group Z_2 = {0, 1} under XOR (addition mod 2).
    This is the algebraic foundation of binary computing.

    Group axioms:
      Closure:     0 XOR 0 = 0, 0 XOR 1 = 1, 1 XOR 0 = 1, 1 XOR 1 = 0  CHECK
      Associative: (a XOR b) XOR c = a XOR (b XOR c)                      CHECK
      Identity:    a XOR 0 = a  (0 is the identity)                        CHECK
      Inverse:     a XOR a = 0  (every element is its own inverse)         CHECK

    This is also Z/2Z: integers mod 2.

    APPLICATIONS:
      1. Parity check codes (Hamming code):
         Append XOR of all bits -> detect 1-bit errors.
         Parity bit = sum mod 2 = XOR of all message bits.
         If received parity != computed parity -> bit flip detected.

      2. CRC (Cyclic Redundancy Check):
         Polynomial ring Z_2[x] / (generator polynomial)
         Generator poly: x^16 + x^15 + x^2 + 1 (CRC-16)
         Used in: Ethernet, USB, storage devices.

      3. Exclusive OR in cryptography:
         One-time pad: ciphertext = message XOR key
         XOR with same key twice recovers message: (m XOR k) XOR k = m XOR (k XOR k) = m XOR 0 = m
         Perfect secrecy (Shannon) if key is truly random and used once.

      4. GF(2^n) -- Galois field:
         Z_2 extended to n-bit words. Used in:
         AES encryption, RAID parity, Reed-Solomon error correction, QR codes.

      5. Quantum: Pauli X gate = bit flip. Z gate = phase flip.
         Z_2 symmetry: X^2 = Z^2 = I (applying twice = identity).
         Qubit stabilizer codes use Z_2 groups extensively.

    Connection to photonics / Fourier:
      Symmetry group of FT: F^4 = I, so FT has order 4 in the group of linear operators.
      Parity operator P: P*f(x) = f(-x). P^2 = I -> Z_2 symmetry.
      Even functions: eigenvalue +1. Odd functions: eigenvalue -1.
      This is why cosines and sines are the FT basis: they are eigenfunctions of P.
    """
    # Z_2 Cayley table
    table = np.array([[0, 1], [1, 0]])   # XOR table

    # Parity check example
    msg = np.array([1, 0, 1, 1, 0, 1, 0])
    parity = int(np.sum(msg) % 2)
    msg_with_parity = np.append(msg, parity)

    # Simulate 1-bit error
    err_pos = 3
    received = msg_with_parity.copy(); received[err_pos] ^= 1
    check_parity = int(np.sum(received) % 2)
    error_detected = (check_parity != 0)

    # XOR encryption demo
    plaintext = np.array([1,0,1,1,0,1,0,0], dtype=np.uint8)  # ASCII 'B' = 0x42 = 01000010... approx
    key = np.array([1,1,0,1,0,0,1,1], dtype=np.uint8)
    ciphertext = plaintext ^ key
    decrypted = ciphertext ^ key
    recovered = np.array_equal(plaintext, decrypted)

    return {
        'group': 'Z_2 = {0,1} under XOR (addition mod 2)',
        'cayley_table': table,
        'axioms': {
            'closure': True,
            'associative': True,
            'identity': 0,
            'every_element_self_inverse': True,
        },
        'parity_check': {
            'message': msg.tolist(),
            'parity_bit': parity,
            'transmitted': msg_with_parity.tolist(),
            'error_at_bit': err_pos,
            'received': received.tolist(),
            'error_detected': error_detected,
        },
        'xor_encryption': {
            'plaintext': plaintext.tolist(),
            'key': key.tolist(),
            'ciphertext': ciphertext.tolist(),
            'decrypted': decrypted.tolist(),
            'recovered': recovered,
        },
        'applications': [
            'Hamming code: parity check bits detect/correct 1-bit errors',
            'CRC: polynomial in Z_2[x] for burst error detection',
            'AES: GF(2^8) field arithmetic',
            'QKD: BB84 key sifting uses XOR to verify shared key',
        ],
        'FT_connection': (
            'Parity operator P: P*f(x)=f(-x). P^2=I -> Z_2 symmetry.\n'
            'FT eigenvalues: +1 (even functions), -1 (odd functions).\n'
            'F^4=I: FT applied 4 times = identity. '
            'Order-4 element in operator group (bigger than Z_2, but Z_2 is subgroup).\n'
            'GS algorithm: two FT projections per iteration. F*P*F*P = convergence.'
        ),
    }


def demo():
    print("=== MICROFLUIDICS + SOFT MATTER + SPECTROSCOPY + GROUP THEORY ===\n")

    print("--- Reynolds Number (water, 10um channel) ---")
    v = 1e-3   # 1 mm/s
    r = reynolds_number(RHO_WATER, v, 10e-6, MU_WATER)
    print(f"  Re={r['Re']:.4f}: {r['regime']}. {r['mixing']}")

    print("\n--- Stokes Flow in 100x50um channel ---")
    flow = stokes_flow_channel(w_um=100, h_um=50, dP_Pa_per_m=5e4)
    print(f"  u_max={flow['u_max_mm_per_s']:.2f} mm/s  "
          f"Q={flow['Q_nL_per_s']:.2f} nL/s  Re={flow['Re']:.4f}")

    print("\n--- Capillary Number ---")
    ca = capillary_number(mu=5e-3, v=1e-3)
    print(f"  Ca={ca['Ca']:.5f}: {ca['regime']} -> {ca['application']}")

    print("\n--- Diffusion Timescale (protein in 50um channel) ---")
    dt = diffusion_timescale(D_m2_per_s=8e-11, L_um=50)
    print(f"  tau={dt['tau_ms']:.2f} ms")
    print(f"  Species timescales (s): { {k: f'{v:.3f}' for k,v in dt['species_timescales_s'].items()} }")

    print("\n--- Droplet Generator ---")
    dg = droplet_generator(Q_oil_uL_per_min=10, Q_water_uL_per_min=2)
    print(f"  d={dg['droplet_diameter_um']:.0f} um  "
          f"V={dg['droplet_volume_pL']:.2f} pL  "
          f"{dg['throughput']}")

    print("\n--- Atomic Emission / Fireworks ---")
    ae = atomic_emission_spectra()
    for name, d in ae['colors'].items():
        print(f"  {name:12s}: {d['lambda_nm']}nm  {d['E_photon_eV']:.2f}eV  {d['transition']}")

    print("\n--- Z_2 Group {0,1} ---")
    g = group_z2_binary()
    print(f"  Group: {g['group']}")
    print(f"  Parity error detected: {g['parity_check']['error_detected']}")
    print(f"  XOR decryption recovered: {g['xor_encryption']['recovered']}")
    print(f"  Applications: {g['applications'][:2]}")

    print("\n--- PDMS Protocol ---")
    pdms = pdms_fabrication_protocol()
    print(f"  Steps: {len(pdms['steps'])}")
    print(f"  Cost per chip: ${pdms['cost_per_chip_usd']}")
    print(f"  STEAM connection: {pdms['steam_camera_connection'][:80]}...")

    print("\n=== DONE ===")


if __name__ == '__main__':
    demo()
