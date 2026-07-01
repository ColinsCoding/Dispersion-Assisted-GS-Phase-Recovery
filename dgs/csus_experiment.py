"""
Cost-effective Jalali time-stretch experiment for CSUS Computer Engineering.
Physics minor context: Cal State Sacramento -> UC Davis lab access pipeline.

Goal: Reproduce the core Coppinger 1999 result (stretch factor M=1+L2/L1)
with ~$800-2500 in telecom surplus components.
The GitHub repo IS the theoretical lab notebook. This module computes
what you expect to measure before you touch any hardware.

SBIR angle: A reproducible $1500 benchtop version is proof-of-concept
for Phase I. NIH reviewers want to see the student can actually build it.
"""
import numpy as np
import sympy as sp

# ---------------------------------------------------------------------------
# Bill of Materials — what to buy and where
# ---------------------------------------------------------------------------
BILL_OF_MATERIALS = [
    {
        'item': '1550nm DFB laser module',
        'purpose': 'CW optical carrier (seed for time-stretch)',
        'spec': '1549-1551nm, <1MHz linewidth, +10dBm output, butterfly package',
        'source': 'eBay surplus / alibaba',
        'price_usd': 80,
        'part_example': 'Sumitomo SLT5411-CC or Nortel AA1419019',
        'notes': 'Any 1550nm DFB works. Do not buy pulsed laser yet.'
    },
    {
        'item': 'LiNbO3 intensity modulator (MZM)',
        'purpose': 'Encode RF signal onto optical carrier (Coppinger Eq 3)',
        'spec': '10GHz bandwidth, V_pi < 5V, fiber-coupled, 1550nm',
        'source': 'eBay surplus (JDS Uniphase, EOSpace)',
        'price_usd': 200,
        'part_example': 'JDSU 21011295 or EOSpace AX-0MSS-10-PFU',
        'notes': 'V_pi*L ~ 10 V*cm. Quadrature bias = V_pi/2. See dgs/lnbo3.py.'
    },
    {
        'item': 'Single-mode fiber SMF-28 spools',
        'purpose': 'Dispersive elements L1 and L2',
        'spec': 'Corning SMF-28, beta2=-21.7ps^2/km at 1550nm, FC/APC connectors',
        'source': 'eBay surplus, FiberFin, Fiber Instrument Sales',
        'price_usd': 40,
        'part_example': '1km + 5km spools (gives L1=1km, L2=5km, M=6)',
        'notes': 'Stretch factor M = 1+L2/L1 = 6. See eq8_stretch_factor() in coppinger1999.py'
    },
    {
        'item': 'InGaAs PIN photodetector',
        'purpose': 'Convert optical -> electrical (Coppinger Eq 7, |E_out|^2)',
        'spec': '>1GHz bandwidth, 1550nm, FC/APC input',
        'source': 'eBay surplus (New Focus, Thorlabs)',
        'price_usd': 150,
        'part_example': 'New Focus 1811 (125MHz) or 1611 (1GHz)',
        'notes': 'New Focus 1811 is <$100 used and enough for 1GHz demo.'
    },
    {
        'item': 'RF signal generator',
        'purpose': 'Drive MZM at fm (the signal you want to stretch)',
        'spec': '100MHz-3GHz output, 0dBm, SMA output',
        'source': 'eBay surplus / Amazon',
        'price_usd': 80,
        'part_example': 'Rigol DSG815 or ADF4351 eval board ($25)',
        'notes': 'ADF4351 covers 35MHz-4GHz for $25. Needs 3.3V USB power.'
    },
    {
        'item': 'Digital oscilloscope',
        'purpose': 'Capture stretched waveform, measure fm_out = fm/M',
        'spec': '>=500MHz analog bandwidth, 2 channels, USB export',
        'source': 'Rigol / Amazon',
        'price_usd': 350,
        'part_example': 'Rigol DS1054Z (50MHz free, unlockable to 100MHz) or DS1104Z',
        'notes': 'You already need this for any EE lab. Dual purpose.'
    },
    {
        'item': 'FC/APC patch cables and optical power meter',
        'purpose': 'Fiber connections, insertion loss measurement',
        'spec': 'SMF-28 compatible, 1m lengths',
        'source': 'Amazon / FS.com',
        'price_usd': 60,
        'part_example': 'FS.com FC-APC patch cables, Grandway FHP02B power meter',
        'notes': 'FS.com ships from China, cheap and good quality.'
    },
    {
        'item': 'Bias-T and SMA cables',
        'purpose': 'DC bias + RF drive to MZM electrodes',
        'spec': '0-20V DC bias range, 50-ohm RF path',
        'source': 'Mini-Circuits / eBay',
        'price_usd': 40,
        'part_example': 'Mini-Circuits ZFBT-4R2G+ bias-T',
        'notes': 'Bias MZM at V_pi/2 for quadrature (max linear response, see lnbo3.py)'
    },
]

TOTAL_BOM_USD = sum(item['price_usd'] for item in BILL_OF_MATERIALS)

# ---------------------------------------------------------------------------
# Predicted experimental measurements
# ---------------------------------------------------------------------------
def predict_measurements(L1_km=1.0, L2_km=5.0, beta2_ps2km=-21.7,
                          fm_input_ghz=1.0, fm_sweep_ghz=None):
    """
    Compute what you EXPECT to see on the oscilloscope before touching hardware.
    Use this to verify experiment is working correctly.

    Parameters
    ----------
    L1_km : first spool length [km]
    L2_km : second spool length [km]
    beta2_ps2km : GVD of SMF-28 at 1550nm [ps^2/km]
    fm_input_ghz : modulation frequency you set on signal generator [GHz]

    Returns dict of expected measurements.
    """
    M = 1 + L2_km / L1_km
    fm_out_ghz = fm_input_ghz / M

    # Time-domain: input pulse (none — CW laser with MZM gives periodic signal)
    # Output period T_out = M * T_in
    T_in_ns = 1.0 / fm_input_ghz
    T_out_ns = M * T_in_ns

    # Dispersion penalty (Eq 9 of Coppinger)
    try:
        from dgs.coppinger1999 import eq9_dispersion_penalty
    except ModuleNotFoundError:
        from coppinger1999 import eq9_dispersion_penalty
    penalty_dB = 10 * np.log10(
        eq9_dispersion_penalty(L2_km, beta2_ps2km, fm_input_ghz, M) + 1e-30
    )

    # Time delay at fm (frequency-to-time mapping)
    # t(f) = 2*pi*beta2*(L1+L2)*f  [ps]
    t_delay_ps = 2*np.pi*(beta2_ps2km*1e-27)*((L1_km+L2_km)*1e3)*(fm_input_ghz*1e9)*1e12

    # Sweep prediction: fm_out vs fm_in
    if fm_sweep_ghz is None:
        fm_sweep_ghz = np.linspace(0.1, 3.0, 30)
    fm_out_sweep = np.array(fm_sweep_ghz) / M

    return {
        'stretch_factor_M': round(M, 2),
        'input_freq_GHz': fm_input_ghz,
        'output_freq_GHz': round(fm_out_ghz, 4),
        'input_period_ns': round(T_in_ns, 3),
        'output_period_ns': round(T_out_ns, 3),
        'dispersion_penalty_dB': round(penalty_dB, 2),
        'time_delay_ps': round(t_delay_ps, 1),
        'fm_sweep_in': fm_sweep_ghz,
        'fm_sweep_out': fm_out_sweep,
        'verification': f"Set scope timebase to {T_out_ns*2:.1f}ns/div to see 2 cycles",
        'pass_criterion': f"Output frequency = {fm_out_ghz:.4f} GHz +/- 5%"
    }


def verification_checklist():
    """Step-by-step procedure to verify the experiment is working."""
    return [
        "STEP 1: Power on laser. Measure optical power at detector output. "
        "Expect: +7 to +10 dBm after modulator, -3 to -10 dBm after fiber.",

        "STEP 2: Set MZM bias to V_pi/2 (quadrature). Measure DC photocurrent. "
        "Expect: I_DC = 0.5 * I_max. Vary bias to find I_min and I_max first.",

        "STEP 3: Apply RF at fm=100MHz, 0dBm from signal generator. "
        "Observe scope. You should see 100MHz oscillation at detector output. "
        "If not: check V_bias, check RF cable, check fiber connections.",

        "STEP 4: With L1=1km only (no L2): measure output frequency. "
        "Expect: fm_out = fm_in = 100MHz (no stretch, M=1).",

        "STEP 5: Add L2=5km fiber spool. Measure output frequency. "
        "Expect: fm_out = 100MHz/6 = 16.67MHz. This IS the Coppinger Eq(8) result.",

        "STEP 6: Repeat at fm=500MHz, 1GHz, 2GHz. Plot fm_out vs fm_in. "
        "Expect: linear relationship fm_out = fm_in/M (slope = 1/M = 1/6).",

        "STEP 7: Try L2=1km (M=2), 2km (M=3), 5km (M=6). "
        "Verify M = 1 + L2/L1 for all cases. THIS is Fig.2 of Coppinger 1999.",

        "STEP 8: Photo + oscilloscope screenshot for SBIR application. "
        "Caption: 'Benchtop reproduction of Coppinger 1999, M=6, cost ~$800'",
    ]


# ---------------------------------------------------------------------------
# UC Davis professor contact strategy (Fall 2026)
# ---------------------------------------------------------------------------
UC_DAVIS_CONTACTS = [
    {
        'name': 'Prof. S.J.B. Yoo',
        'dept': 'Electrical and Computer Engineering',
        'email': 'sbyoo@ucdavis.edu',
        'research': 'Silicon photonics, optical networks, WDM systems',
        'relevance': 9,  # /10
        'connection': 'Works with integrated photonic devices -- the chip-scale version '
                      'of the time-stretch components in dgs/coppinger1999.py',
        'what_to_say': 'I am implementing the Coppinger 1999 time-stretch equations '
                       'in Python (GitHub: <your-link>) and want to run the benchtop '
                       'experiment. Your group has the fiber and photodetector '
                       'infrastructure I need.',
        'apply_for': 'EEC 199 (special studies/research units)',
    },
    {
        'name': 'Prof. Weijian Yang',
        'dept': 'Electrical and Computer Engineering',
        'email': 'wejyang@ucdavis.edu',
        'research': 'Two-photon microscopy, neural imaging, ultrafast optics',
        'relevance': 8,
        'connection': 'STEAM microscopy for cell imaging (crispr_steam_theory.ipynb) '
                      'directly overlaps with his neural imaging work. CRISPR angle.',
        'what_to_say': 'I built a theoretical model of STEAM microscopy for CRISPR '
                       'gene edit detection (Jupyter notebook, SymPy derivation of '
                       'Jalali equations). Looking for lab access to validate '
                       'the optical phase measurement.',
        'apply_for': 'EEC 199 or volunteer research assistant',
    },
    {
        'name': 'Prof. Xiaoyi Bao',
        'dept': 'Physics (visiting / adjunct)',
        'email': 'physics@ucdavis.edu',
        'research': 'Fiber sensors, distributed sensing, Brillouin scattering',
        'relevance': 6,
        'connection': 'Fiber optics overlap; knows dispersive fiber well.',
        'what_to_say': 'CSUS CE student, physics minor, implementing dispersive '
                       'fiber equations from Coppinger 1999 for SBIR proposal.',
        'apply_for': 'Physics 199',
    },
    {
        'name': 'Prof. Bahram Jalali (UCLA, not UCD)',
        'dept': 'Electrical Engineering, UCLA',
        'email': 'jalali@ucla.edu',
        'research': 'STEAM microscopy, photonic time stretch, rogue waves',
        'relevance': 10,
        'connection': 'The source. Email in September with CRISPR notebook + GitHub.',
        'what_to_say': 'See crispr_steam_theory.ipynb Section 7 email template. '
                       'Lead with the CRISPR delta_phi = 4e-4 rad result. '
                       'Ask: which paper should I read next after Coppinger 1999?',
        'apply_for': 'N/A -- relationship building, not lab access',
    },
]


def cold_email_template(professor_name, professor_email, your_github,
                         your_measurement_result=None):
    """
    Generate cold email to UC Davis professor for lab access.
    Uses the GS derivation + Coppinger notebook as the hook.
    DO NOT send before August 15 (professors return from summer).
    """
    if your_measurement_result is None:
        measurement_line = (
            "I have fully derived the Coppinger 1999 Eq(1) chirp phase "
            "from the convolution theorem and implemented all equations symbolically "
            "in SymPy (20 passing tests)."
        )
    else:
        measurement_line = (
            f"I reproduced the Coppinger 1999 stretch factor M={your_measurement_result['stretch_factor_M']} "
            f"on a benchtop setup, verifying fm_out = {your_measurement_result['output_freq_GHz']} GHz "
            f"(expected {your_measurement_result['input_freq_GHz']/your_measurement_result['stretch_factor_M']:.4f} GHz)."
        )

    return f"""Subject: CSUS CE Student -- Photonic Time-Stretch Research, Lab Access Request

Dear Professor {professor_name.split()[-1]},

I am Colin Casey, a Computer Engineering student at CSUS with a physics minor.
I am independently reproducing the Coppinger, Bhushan & Jalali (1999) IEEE MTT
photonic time-stretch experiment and applying for SBIR Phase I funding for a
STEAM microscopy variant targeting CRISPR gene-edit optical detection.

{measurement_line}

GitHub repository: {your_github}
Key file: notebooks/coppinger1999_sympy.ipynb (full SymPy derivation of paper)
Key file: dgs/coppinger1999.py (35+ functions, derive_eq1_from_convolution)

I am writing to ask whether I could have supervised access to your fiber optics
equipment (1550nm source, SMF fiber spools, photodetector, oscilloscope) to
run the stretch-factor verification experiment. I would need approximately
4 hours of bench time and would provide my own laptop + Python simulation
to pre-calculate expected measurements before touching any hardware.

I am available any week in September. I can commute from Sacramento.
I am happy to work toward EEC 199 research units if that is the appropriate path.

Thank you for your time.

Colin Casey
CSUS Computer Engineering, Physics Minor
colincas37@gmail.com
GitHub: {your_github}
"""


# ---------------------------------------------------------------------------
# Fall 2026 application timeline
# ---------------------------------------------------------------------------
FALL_2026_TIMELINE = [
    {'date': '2026-07-01', 'action': 'Build BOM list, order components',
     'cost': '$800 total',
     'deliverable': 'Hardware in hand by July 15'},

    {'date': '2026-07-15', 'action': 'Run benchtop M verification (Steps 1-5)',
     'cost': '$0 (use CSUS EE lab or borrow scope)',
     'deliverable': 'Photo + scope screenshot showing fm_out = fm_in/M'},

    {'date': '2026-08-01', 'action': 'Draft SBIR Phase I Specific Aims (1 page)',
     'cost': '$0',
     'deliverable': 'Aims page: STEAM cell imaging, P2 from sbir_portfolio.py'},

    {'date': '2026-08-15', 'action': 'Email Yoo + Yang at UC Davis',
     'cost': '$0',
     'deliverable': 'Send cold_email_template() with GitHub link + scope photo'},

    {'date': '2026-09-01', 'action': 'UC Davis fall semester begins',
     'cost': '$0',
     'deliverable': 'Follow up email, schedule visit'},

    {'date': '2026-09-15', 'action': 'Enroll EEC 199 (special research units)',
     'cost': '$CSUS concurrent enrollment fee (~$150/unit)',
     'deliverable': 'Official lab access, faculty supervisor'},

    {'date': '2026-10-01', 'action': 'Run full Coppinger Fig.2 reproduction',
     'cost': '$0 (using UCD equipment)',
     'deliverable': 'M=3,6,8 verified; data in notebooks/benchtop_fig2.ipynb'},

    {'date': '2026-11-01', 'action': 'Submit SBIR Phase I (NIH deadline Nov 5)',
     'cost': '$0 (free to apply)',
     'deliverable': 'P2 STEAM microscopy proposal submitted'},
]


def ride_logistics():
    """
    Sacramento -> UC Davis commute options.
    Distance: 17 miles, ~25 min by car, ~45 min by bus/train.
    """
    return {
        'drive': {'time_min': 25, 'cost_usd': 3.50, 'route': 'I-80W to UCD'},
        'yolobus_42B': {'time_min': 55, 'cost_usd': 1.50,
                        'notes': 'Yolo Bus route 42B: Downtown Sacramento -> UC Davis'},
        'amtrak_capitol_corridor': {'time_min': 20, 'cost_usd': 8,
                                     'notes': 'Sacramento Valley Station -> Davis Station, '
                                              'then walk/bike 1 mile to ECE building'},
        'best_option': 'Amtrak + bike share (Lime scooter at Davis station) = '
                       '$8 + $2 = $10 round trip, 35 min door to door',
        'lab_visit_strategy': 'Schedule 2-hour blocks on Tu/Th when you have no CSUS class. '
                              'Amtrak has 8am and 10am trains from Sacramento.',
    }


def demo():
    print("=== CSUS Cost-Effective Jalali Experiment ===")
    print(f"\nTotal BOM: ${TOTAL_BOM_USD}")
    print("\nComponents:")
    for item in BILL_OF_MATERIALS:
        print(f"  ${item['price_usd']:4d}  {item['item']}")

    print("\n=== Predicted Measurements (L1=1km, L2=5km) ===")
    pred = predict_measurements(L1_km=1.0, L2_km=5.0, fm_input_ghz=1.0)
    for k, v in pred.items():
        if not hasattr(v, '__len__') or isinstance(v, str):
            print(f"  {k}: {v}")

    print("\n=== Verification Checklist ===")
    for step in verification_checklist():
        print(f"  {step[:80]}...")

    print("\n=== UC Davis Contacts ===")
    for c in UC_DAVIS_CONTACTS[:2]:
        print(f"  {c['name']} ({c['dept']})")
        print(f"    Relevance: {c['relevance']}/10")
        print(f"    What to say: {c['what_to_say'][:70]}...")

    print("\n=== Fall 2026 Timeline ===")
    for item in FALL_2026_TIMELINE:
        print(f"  {item['date']}: {item['action']}")
        print(f"    -> {item['deliverable']}")

    print("\n=== Ride Logistics ===")
    log = ride_logistics()
    print(f"  Best: {log['best_option']}")

    print("\n=== Cold Email Preview ===")
    email = cold_email_template("S.J.B. Yoo", "sbyoo@ucdavis.edu",
                                 "https://github.com/YOUR_USERNAME/Dispersion-Assisted-GS-Phase-Recovery")
    print(email[:600] + "...")


if __name__ == '__main__':
    demo()
