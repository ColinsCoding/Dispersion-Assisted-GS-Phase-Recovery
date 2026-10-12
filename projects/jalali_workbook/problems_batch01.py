"""Problem data for Batch 1 (JW-001 ... JW-050). Each entry: metadata fields + a SymPy solution."""

S = {
    'NIST': 'NIST CODATA / SI 2019 exact constants (c, h, e)',
    'AGR': 'G. P. Agrawal, *Nonlinear Fiber Optics*, Academic Press, Ch. 3 (dispersion, Gaussian and chirped Gaussian pulses)',
    'ST': 'B. E. A. Saleh and M. C. Teich, *Fundamentals of Photonics*, Wiley (interference, Jones matrices, coupled-mode theory, fibers)',
    'HJ03': 'Y. Han and B. Jalali, "Photonic time-stretched analog-to-digital converter: fundamental concepts and practical considerations," *J. Lightw. Technol.*, vol. 21, no. 12, pp. 3085-3103, 2003',
    'GJ13': 'K. Goda and B. Jalali, "Dispersive Fourier transformation for fast continuous single-shot measurements," *Nat. Photonics*, vol. 7, pp. 102-112, 2013',
    'GTJ09': 'K. Goda, K. K. Tsia, and B. Jalali, "Serial time-encoded amplified imaging for real-time observation of fast dynamic phenomena," *Nature*, vol. 458, pp. 1145-1149, 2009',
    'SRKJ07': 'D. R. Solli, C. Ropers, P. Koonath, and B. Jalali, "Optical rogue waves," *Nature*, vol. 450, pp. 1054-1057, 2007',
    'BJ04': 'O. Boyraz and B. Jalali, "Demonstration of a silicon Raman laser," *Opt. Express*, vol. 12, no. 21, pp. 5269-5273, 2004',
    'DBGJ13': 'E. D. Diebold, B. W. Buckley, D. R. Gossett, and B. Jalali, "Digitally synthesized beat frequency multiplexing for sub-millisecond fluorescence microscopy," *Nat. Photonics*, vol. 7, pp. 806-810, 2013',
    'MAH17': 'A. Mahjoubfar *et al.*, "Time stretch and its applications," *Nat. Photonics*, vol. 11, pp. 341-351, 2017',
    'GOOD': 'J. W. Goodman, *Speckle Phenomena in Optics*, Roberts & Co. (exponential intensity statistics of fully developed speckle)',
    'WIL': 'E. B. Wilson, "Probable inference, the law of succession, and statistical inference," *J. Amer. Statist. Assoc.*, vol. 22, pp. 209-212, 1927',
    'KP': 'C. Kharif and E. Pelinovsky, "Physical mechanisms of the rogue wave phenomenon," *Eur. J. Mech. B/Fluids*, vol. 22, pp. 603-634, 2003 (the 2x significant-height convention)',
    'MAL': 'I. H. Malitson, *J. Opt. Soc. Am.*, vol. 55, pp. 1205-1209, 1965 (fused-silica Sellmeier; evaluated with the repo library `dgs/materials.py`)',
    'SYN': 'synthetic educational model',
}

# Common header every SymPy section starts with (workbook requirement), plus exact SI constants.
HDR = '''import sympy as sp
sp.init_printing()
c = sp.Integer(299792458)                       # speed of light, m/s (exact, SI 2019)
h = sp.Rational(662607015, 10**42)              # Planck constant, J s (exact)
qe = sp.Rational(1602176634, 10**28)            # elementary charge, C (exact)
def say(label, val, unit='', digits=6):
    v = sp.nsimplify(val) if isinstance(val, (int, float)) else val
    print(f'{label} = {v}  ~=  {sp.N(v, digits)} {unit}'.rstrip())
'''

P = []
def prob(**kw):
    P.append(kw)

prob(id=1, theme='Optical communication (carrier frequency)', cat='A', pre=[], diff=1,
     obj='Convert a free-space wavelength to an optical frequency with exact constants.',
     ctx='Telecom-band light used in time-stretch systems is specified by wavelength; RF engineers think in frequency.',
     given='scalar wavelength λ = 1550 nm = 1.550×10⁻⁶ m; scalar c = 299 792 458 m/s (exact).',
     unknown='scalar frequency f (Hz).', assume='vacuum wavelength; f > 0, λ > 0.',
     model='c = λ f  (c: m/s, λ: m, f: Hz = 1/s); dimension check: m · s⁻¹ = m · s⁻¹.',
     q='Find f exactly as a rational number, then in THz.', atype='exact rational, then decimal in THz',
     mistake='Using λ in nm without converting to m (off by 10⁹).', verify='f·λ reproduces c exactly; Lean `JW.freq_of_wavelength`.',
     src='NIST',
     code='lam = sp.Rational(1550, 10**9)\nf = c/lam\nsay("f (Hz)", f)\nsay("f (THz)", f/10**12)\nassert f*lam == c')

prob(id=2, theme='Photonics (photon energy)', cat='A', pre=[1], diff=1,
     obj='Compute photon energy in joules and electron-volts.',
     ctx='Photon energy sets detector bandgap requirements (InGaAs vs Si) for 1550 nm light.',
     given='λ = 1550 nm; h = 6.62607015×10⁻³⁴ J·s (exact); e = 1.602176634×10⁻¹⁹ C (exact).',
     unknown='scalar E (J) and E (eV).', assume='single photon in vacuum.',
     model='E = h c / λ (J); E_eV = E / e. Dimension: J·s · m/s / m = J.',
     q='Compute E in J and eV. Can a silicon detector (bandgap ≈ 1.12 eV) absorb it by band-to-band absorption?',
     atype='exact rational + decimal; yes/no', mistake='Dividing by e twice or using h-bar instead of h.',
     verify='E_eV · λ(µm) ≈ 1.23984 (the 1240 eV·nm rule).', src='NIST',
     code='lam = sp.Rational(1550, 10**9)\nE = h*c/lam\nsay("E (J)", E)\nsay("E (eV)", E/qe)\nsay("check E_eV * lambda_um", E/qe*sp.Rational(155,100))\nprint("absorbed by Si band-to-band?", bool(E/qe > sp.Rational(112,100)))')

prob(id=3, theme='Photonics (photon flux)', cat='A', pre=[2], diff=1,
     obj='Convert optical power to photon arrival rate.',
     ctx='Shot-noise-limited detection (JW-036) depends on how many photons arrive per pixel.',
     given='P = 1.0 mW = 10⁻³ W; λ = 1550 nm.', unknown='scalar photon rate R (photons/s).',
     assume='monochromatic light, constant power.', model='R = P / E = P λ / (h c). Dimension: W / J = s⁻¹.',
     q='Find R. How many photons arrive in a 100 ps time-stretch pixel?', atype='decimal in scientific notation',
     mistake='Forgetting that E is per photon, so R = P/E, not P·E.', verify='R·E = P exactly.', src='NIST',
     code='lam = sp.Rational(1550, 10**9); Pw = sp.Rational(1, 1000)\nE = h*c/lam; R = Pw/E\nsay("R (photons/s)", R)\nsay("photons per 100 ps", R*sp.Rational(1, 10**10))\nassert sp.simplify(R*E - Pw) == 0')

prob(id=4, theme='Dispersion units', cat='A', pre=[], diff=2,
     obj='Perform unit conversion of the fiber dispersion parameter D to SI.',
     ctx='Fiber datasheets give D in ps/(nm·km); equations need s/m².',
     given='scalar D = 1 ps/(nm·km).', unknown='D in s/m².', assume='none.',
     model='1 ps = 10⁻¹² s, 1 nm = 10⁻⁹ m, 1 km = 10³ m; D has dimension time / (length · length).',
     q='Convert 1 ps/(nm·km) to s/m², using SymPy units.', atype='exact power of ten',
     mistake='Treating nm and km as cancelling (they are both lengths but enter as a product).',
     verify='sympy.physics.units.convert_to.', src='SYN',
     code='from sympy.physics import units as u\nD = u.picosecond/(u.nanometer*u.kilometer)\nDsi = u.convert_to(D, [u.second, u.meter])\nprint("1 ps/(nm km) =", Dsi)')

prob(id=5, theme='Dispersion (β₂ from D)', cat='J', pre=[1, 4], diff=2,
     obj='Relate the group-velocity dispersion β₂ to D, with signs.',
     ctx='Pulse-propagation equations use β₂; datasheets give D.',
     given='D = 17 ps/(nm·km) (typical standard single-mode fiber value at 1550 nm, used here as a textbook number); λ = 1550 nm.',
     unknown='scalar β₂ in ps²/km.', assume='β₂ and D refer to the same wavelength; D > 0 means anomalous dispersion.',
     model='β₂ = −λ² D / (2π c)  (β₂: s²/m; λ: m; D: s/m²; c: m/s). Dimension: m²·s/m² / (m/s) = s²/m.',
     q='Compute β₂ in ps²/km. Is the fiber normal or anomalous at 1550 nm?', atype='signed decimal',
     mistake='Dropping the minus sign, or mixing ps/(nm km) with SI.', verify='D = −2πcβ₂/λ² round trip.', src='AGR',
     code='lam = sp.Rational(1550, 10**9)\nD = 17*sp.Rational(1, 10**6)          # s/m^2 (JW-004)\nb2 = -lam**2*D/(2*sp.pi*c)              # s^2/m\nsay("beta2 (ps^2/km)", b2*sp.Integer(10)**24*1000)\nassert sp.simplify(-2*sp.pi*c*b2/lam**2 - D) == 0\nprint("anomalous (beta2 < 0):", bool(b2 < 0))')

prob(id=6, theme='Time-stretch imaging (frame timing)', cat='A', pre=[], diff=1,
     obj='Use scientific notation to relate frame rate, period and samples per frame.',
     ctx='In a time-stretch camera each laser pulse is one frame (line scan); a digitizer samples the stretched pulse.',
     given='laser repetition rate f_rep = 6×10⁶ Hz (synthetic value, same order as reported systems [GTJ09]); digitizer rate f_s = 50×10⁹ samples/s (synthetic).',
     unknown='frame period T_rep (s); samples per frame N (integer).', assume='one frame per pulse; no dead time.',
     model='T_rep = 1/f_rep; N = floor(f_s · T_rep). Dimension: (1/s)·s = 1.',
     q='Compute T_rep in ns and N.', atype='exact rational, then integer', mistake='Rounding N up (a partial sample does not exist).',
     verify='N·(1/f_s) ≤ T_rep < (N+1)/f_s.', src='SYN',
     code='frep = 6*10**6; fs = 50*10**9\nT = sp.Rational(1, frep)\nsay("T_rep (ns)", T*10**9)\nN = sp.floor(fs*T)\nprint("samples per frame N =", N, "(exact ratio", sp.Rational(fs, frep), ")")\nassert N*sp.Rational(1, fs) <= T < (N+1)*sp.Rational(1, fs)')

prob(id=7, theme='Real-time data (big data from time stretch)', cat='A', pre=[6], diff=1,
     obj='Estimate raw data rates and storage, with units.',
     ctx='Continuous time-stretch instruments produce very large data streams [MAH17].',
     given='f_s = 50×10⁹ samples/s; b = 8 bits/sample (synthetic).', unknown='bit rate (bit/s), byte rate (B/s), storage for 1 h (TB).',
     assume='continuous recording, no compression; 1 TB = 10¹² B.', model='R_bit = f_s b; R_byte = R_bit/8; S = R_byte · t.',
     q='Compute the three quantities.', atype='decimals with SI prefixes', mistake='Confusing bits and bytes (factor 8) or TB with TiB.',
     verify='dimension: samples/s · bit/sample = bit/s.', src='SYN',
     code='fs = 50*10**9; b = 8\nRbit = fs*b; Rbyte = sp.Rational(Rbit, 8)\nsay("bit rate (Gbit/s)", sp.Rational(Rbit, 10**9))\nsay("byte rate (GB/s)", Rbyte/10**9)\nsay("1 hour (TB)", Rbyte*3600/10**12)')

prob(id=8, theme='Optical communication (link loss)', cat='A', pre=[], diff=1,
     obj='Add losses in dB and convert to a linear transmission.',
     ctx='Dispersive fiber spools for time stretch are kilometers long; their loss must be budgeted.',
     given='attenuation a = 0.2 dB/km; length L = 50 km; two connectors at 0.5 dB each.',
     unknown='total loss (dB); transmission T (dimensionless).', assume='losses independent, so dB values add.',
     model='Loss_dB = a L + 2 · 0.5; T = 10^(−Loss_dB/10).', q='Find the loss and T.', atype='exact power of ten + decimal',
     mistake='Using 20 log for optical power (that is for field amplitude).', verify='Lean `JW.dB_mul`: multiplying gains adds dB.', src='SYN',
     code='a = sp.Rational(2, 10); L = 50\nloss = a*L + 2*sp.Rational(1, 2)\nT = sp.Integer(10)**(-loss/10)\nsay("loss (dB)", loss); say("T", T)\nsay("fiber alone", sp.Integer(10)**(-a*L/10))')

prob(id=9, theme='Optical power levels (dBm)', cat='A', pre=[8], diff=1,
     obj='Work with absolute (dBm) and relative (dB) power units.',
     ctx='Optical amplifiers (e.g. Raman amplification in STEAM [GTJ09]) are specified by dB gain.',
     given='P_in = −30 dBm; amplifier gain G = 20 dB.', unknown='P_in (W), P_out (dBm and W).',
     assume='linear amplifier, no saturation.', model='P_dBm = 10 log₁₀(P / 1 mW); P_out,dBm = P_in,dBm + G_dB.',
     q='Compute P_in in µW and P_out in dBm and µW. What is wrong with adding two dBm values?', atype='decimals; short explanation',
     mistake='Adding dBm + dBm (that multiplies two powers, which has units W²).', verify='P_out/P_in = 10^(G/10) = 100.', src='SYN',
     code='Pin_dBm = -30; G = 20\nPin = sp.Integer(10)**sp.Rational(Pin_dBm, 10)/1000\nPout_dBm = Pin_dBm + G\nPout = sp.Integer(10)**sp.Rational(Pout_dBm, 10)/1000\nsay("P_in (uW)", Pin*10**6); print("P_out =", Pout_dBm, "dBm"); say("P_out (uW)", Pout*10**6)\nassert Pout/Pin == 100')

prob(id=10, theme='Phasors (complex optical field)', cat='B', pre=[], diff=1,
     obj='Show that detected intensity does not depend on the absolute optical phase.',
     ctx='Photodetectors measure |E|²; phase must be recovered indirectly (interference, dispersion).',
     given='scalar complex field E(t) = A e^{i(ωt+φ)} with real A ≥ 0 (V/m), ω (rad/s), φ (rad).',
     unknown='intensity proxy |E|² (V²/m²).', assume='scalar, monochromatic field.', model='|E|² = E · E*.',
     q='Simplify |E|² symbolically.', atype='symbolic expression', mistake='Writing |E|² = E² (forgets the conjugate).',
     verify='SymPy simplification (this is algebra, not a formal proof).', src='ST',
     code='A = sp.symbols("A", nonnegative=True); w, t, phi = sp.symbols("omega t phi", real=True)\nE = A*sp.exp(sp.I*(w*t + phi))\nprint("|E|^2 =", sp.simplify(E*sp.conjugate(E)))\nprint("E^2 (wrong) =", sp.simplify(E**2))')

prob(id=11, theme='Interference (fringe visibility)', cat='B', pre=[10], diff=2,
     obj='Derive two-beam interference and fringe visibility.',
     ctx='Spectral interferometry is a standard way to measure phase in time-stretch systems [GJ13].',
     given='intensities I₁ = 4, I₂ = 1 (arbitrary units); phase difference Δφ (rad).',
     unknown='I(Δφ), I_max, I_min, visibility V.', assume='mutually coherent, same polarization.',
     model='I = I₁ + I₂ + 2√(I₁I₂) cos Δφ; V = (I_max − I_min)/(I_max + I_min).',
     q='Find I_max, I_min and V. Edge case: what is V when I₂ = 0?', atype='exact numbers', mistake='Adding fields\' intensities without the cross term.',
     verify='V = 2√(I₁I₂)/(I₁+I₂).', src='ST',
     code='I1, I2 = sp.Integer(4), sp.Integer(1)\nImax = I1 + I2 + 2*sp.sqrt(I1*I2); Imin = I1 + I2 - 2*sp.sqrt(I1*I2)\nV = (Imax - Imin)/(Imax + Imin)\nprint("Imax =", Imax, " Imin =", Imin, " V =", V)\nx = sp.symbols("I_1", positive=True)\nprint("edge I2 = 0: V =", sp.simplify((2*sp.sqrt(x*0))/(x + 0)))')

prob(id=12, theme='RF photonics (detector frequency response)', cat='B', pre=[10], diff=2,
     obj='Find magnitude and phase of a first-order transfer function.',
     ctx='A photodiode with load resistance behaves as a single-pole low-pass filter.',
     given='H(ω) = 1/(1 + iωτ), τ > 0 (s), ω ≥ 0 (rad/s).', unknown='|H| and arg H at ω = 1/τ.',
     assume='single-pole model.', model='|H| = 1/√(1+ω²τ²); arg H = −atan(ωτ).',
     q='Evaluate at ω = 1/τ and convert |H|² to dB.', atype='exact surds and angle', mistake='Reporting −3 dB as |H| = 1/2 (it is |H|² = 1/2).',
     verify='|H(1/τ)|² = 1/2 exactly.', src='ST',
     code='w, tau = sp.symbols("omega tau", positive=True)\nH = 1/(1 + sp.I*w*tau)\nH0 = H.subs(w, 1/tau)\nprint("|H| =", sp.simplify(sp.Abs(H0)), " arg H =", sp.simplify(sp.arg(H0)))\nsay("|H|^2 in dB", 10*sp.log(sp.Abs(H0)**2, 10))')

prob(id=13, theme='Polar coordinates (Gaussian beam power)', cat='B', pre=[], diff=2,
     obj='Integrate a radially symmetric intensity in polar coordinates.',
     ctx='Laser beams illuminating cells in a flow channel are approximately Gaussian.',
     given='I(r) = I₀ exp(−2r²/w²) (W/m²), I₀ > 0, beam radius w > 0 (m), r ≥ 0 (m).',
     unknown='total power P (W); fraction inside r = w.', assume='ideal TEM₀₀ beam.',
     model='P = ∫₀^{2π} ∫₀^∞ I(r) r dr dθ (the Jacobian r from polar coordinates).',
     q='Compute P and the fraction of power inside r = w.', atype='closed form; decimal', mistake='Forgetting the Jacobian r.',
     verify='fraction → 1 as the radius → ∞.', src='ST',
     code='r, w, I0, th = sp.symbols("r w I_0 theta", positive=True)\nI = I0*sp.exp(-2*r**2/w**2)\nPtot = sp.integrate(sp.integrate(I*r, (r, 0, sp.oo)), (th, 0, 2*sp.pi))\nfrac = sp.integrate(I*r, (r, 0, w))*2*sp.pi/Ptot\nprint("P =", sp.simplify(Ptot)); say("fraction inside w", sp.simplify(frac))')

prob(id=14, theme='Dispersion (group index from a Cauchy model)', cat='C', pre=[], diff=2,
     obj='Differentiate a refractive-index model to get the group index.',
     ctx='Group index sets pulse delay (JW-046); it differs from the phase index.',
     given='n(λ) = A + B/λ² with A = 1.4580, B = 0.00354 µm² (synthetic Cauchy coefficients); λ = 1.55 µm.',
     unknown='n_g(λ) = n − λ dn/dλ.', assume='Cauchy model valid in the band (simplified educational model).',
     model='n_g = n − λ n′(λ); here n_g = A + 3B/λ².', q='Derive n_g symbolically and evaluate at 1.55 µm.', atype='expression + decimal',
     mistake='Sign error: n_g = n − λ n′, and n′ < 0 here, so n_g > n.', verify='compare with the closed form A + 3B/λ².', src='SYN',
     code='lam, A, B = sp.symbols("lambda A B", positive=True)\nn = A + B/lam**2\nng = sp.simplify(n - lam*sp.diff(n, lam))\nprint("n_g =", ng)\nvals = {A: sp.Rational(14580, 10000), B: sp.Rational(354, 100000), lam: sp.Rational(155, 100)}\nsay("n at 1.55 um", n.subs(vals)); say("n_g at 1.55 um", ng.subs(vals))')

prob(id=15, theme='Limits (sinc spectra)', cat='C', pre=[], diff=1,
     obj='Evaluate a removable singularity with a limit.',
     ctx='Rectangular pulses have sinc-shaped spectra; the center value needs a limit.',
     given='g(x) = sin(x)/x, x ∈ ℝ \\ {0}.', unknown='lim_{x→0} g(x) and lim_{x→0} g(x)².', assume='none.',
     model='Taylor: sin x = x − x³/6 + O(x⁵).', q='Compute both limits. Edge case: what does floating-point evaluation give at x = 0 exactly?',
     atype='exact numbers; failure description', mistake='Plugging in x = 0 (0/0 is undefined).', verify='sp.limit and the series.', src='SYN',
     code='x = sp.symbols("x")\nprint("lim sin(x)/x =", sp.limit(sp.sin(x)/x, x, 0), "  lim (sin(x)/x)^2 =", sp.limit((sp.sin(x)/x)**2, x, 0))\nprint("series:", sp.series(sp.sin(x)/x, x, 0, 5))\nimport math\ntry:\n    math.sin(0.0)/0.0\nexcept ZeroDivisionError as e:\n    print("float at x = 0:", type(e).__name__, "(use numpy.sinc or a limit)")')

prob(id=16, theme='Taylor series (early pulse broadening)', cat='C', pre=[], diff=2,
     obj='Use a Taylor approximation and bound its error.',
     ctx='For short propagation distances, pulse broadening is quadratic in distance.',
     given='g(u) = √(1+u), u = (z/L_D)² ≥ 0.', unknown='second-order Taylor polynomial; error at z = L_D/2.',
     assume='u small.', model='√(1+u) = 1 + u/2 − u²/8 + O(u³).', q='Find the polynomial and its absolute error at u = 1/4.',
     atype='polynomial; exact error', mistake='Expanding in z/L_D but truncating inconsistently (mixing orders).',
     verify='compare exact √(5/4) with the polynomial.', src='AGR',
     code='u = sp.symbols("u", nonnegative=True)\np2 = sp.series(sp.sqrt(1 + u), u, 0, 3).removeO()\nprint("Taylor:", p2)\nerr = sp.sqrt(1 + sp.Rational(1, 4)) - p2.subs(u, sp.Rational(1, 4))\nsay("error at u = 1/4", err)')

prob(id=17, theme='Integrals (pulse energy)', cat='C', pre=[], diff=1,
     obj='Integrate a Gaussian power profile.',
     ctx='Pulse energy determines photon budget per frame.',
     given='P(t) = P₀ exp(−t²/T₀²); P₀ = 1 W; T₀ = 1 ps.', unknown='energy E_p (J).', assume='Gaussian intensity profile.',
     model='E_p = ∫_{−∞}^{∞} P(t) dt = √π P₀ T₀.', q='Compute E_p in pJ.', atype='closed form + decimal',
     mistake='Using P₀ × FWHM (only approximate).', verify='SymPy integral equals √π P₀T₀.', src='AGR',
     code='t, P0, T0 = sp.symbols("t P_0 T_0", positive=True)\nE = sp.integrate(P0*sp.exp(-t**2/T0**2), (t, -sp.oo, sp.oo))\nprint("E_p =", E)\nsay("E_p (pJ) for 1 W, 1 ps", E.subs({P0: 1, T0: sp.Rational(1, 10**12)})*10**12)')

prob(id=18, theme='Pulse width conventions (FWHM)', cat='C', pre=[17], diff=2,
     obj='Relate the 1/e half-width T₀ to the full width at half maximum.',
     ctx='Papers quote FWHM; equations use T₀ [AGR].', given='intensity profile P(t) = P₀ exp(−t²/T₀²).',
     unknown='T_FWHM / T₀.', assume='Gaussian.', model='solve P(t) = P₀/2.', q='Find the exact ratio.', atype='exact surd',
     mistake='Using the field profile exp(−t²/2T₀²) and the intensity profile interchangeably.', verify='plug the half-width back in.', src='AGR',
     code='t, T0 = sp.symbols("t T_0", positive=True)\nthalf = sp.solve(sp.Eq(sp.exp(-t**2/T0**2), sp.Rational(1, 2)), t)[0]\nratio = sp.simplify(2*thalf/T0)\nsay("T_FWHM / T0", ratio)')

prob(id=19, theme='Dispersion length', cat='J', pre=[5], diff=2,
     obj='Compute the characteristic distance over which dispersion matters.',
     ctx='Whether a fiber spool "stretches" a pulse depends on L compared with L_D.',
     given='T₀ = 1 ps; β₂ from JW-005 (≈ −21.7 ps²/km).', unknown='L_D (m).', assume='Gaussian pulse; β₂ constant.',
     model='L_D = T₀² / |β₂| (m).', q='Compute L_D in meters.', atype='decimal', mistake='Using β₂ with its sign (L_D must be positive).',
     verify='dimension: ps² / (ps²/km) = km.', src='AGR',
     code='lam = sp.Rational(1550, 10**9); D = 17*sp.Rational(1, 10**6)\nb2 = -lam**2*D/(2*sp.pi*c)\nT0 = sp.Rational(1, 10**12)\nLD = T0**2/sp.Abs(b2)\nsay("L_D (m)", LD)')

prob(id=20, theme='Pulse broadening', cat='J', pre=[19, 16], diff=2,
     obj='Find where an unchirped Gaussian pulse doubles in width.',
     ctx='Time stretch deliberately broadens pulses by many orders of magnitude.',
     given='T(z) = T₀ √(1 + (z/L_D)²), with L_D from JW-019.', unknown='z₂ such that T(z₂) = 2T₀.',
     assume='unchirped Gaussian input; second-order dispersion only.', model='[AGR] Eq. for unchirped Gaussian broadening.',
     q='Solve for z₂ in terms of L_D and in meters. Show T(z) ≥ T₀ for all z.', atype='exact surd; decimal',
     mistake='Taking z₂ = 2L_D (broadening is not linear for z ≲ L_D).', verify='Lean `JW.broadening_ge`, `JW.broadening_doubles`.', src='AGR',
     code='z, LD = sp.symbols("z L_D", positive=True)\nz2 = sp.solve(sp.Eq(sp.sqrt(1 + (z/LD)**2), 2), z)[0]\nprint("z2 =", z2)\nlam = sp.Rational(1550, 10**9); b2 = -lam**2*17*sp.Rational(1, 10**6)/(2*sp.pi*c)\nsay("z2 (m)", z2.subs(LD, sp.Rational(1, 10**24)/sp.Abs(b2)))')

prob(id=21, theme='Gradients (beam edges)', cat='D', pre=[13], diff=3,
     obj='Compute a 2D gradient and find where it is steepest.',
     ctx='Edge-sensitive image processing responds to intensity gradients.',
     given='I(x, y) = I₀ exp(−2(x²+y²)/w²); I₀, w > 0.', unknown='vector ∇I at (w/2, 0); radius where |∇I| is maximal.',
     assume='ideal Gaussian.', model='∇I = (∂I/∂x, ∂I/∂y); radial slope dI/dr.',
     q='Compute ∇I at (w/2, 0) and the radius of maximum slope.', atype='vector expression; radius', mistake='Assuming the steepest point is at r = w.',
     verify='second derivative test on |dI/dr|.', src='SYN',
     code='x, y, r = sp.symbols("x y r", real=True); I0, w = sp.symbols("I_0 w", positive=True)\nI = I0*sp.exp(-2*(x**2 + y**2)/w**2)\ngrad = sp.Matrix([sp.diff(I, x), sp.diff(I, y)])\nprint("grad I at (w/2, 0) =", sp.simplify(grad.subs({x: w/2, y: 0})).T)\nslope = sp.diff(I0*sp.exp(-2*r**2/w**2), r)\nprint("max |slope| at r =", [s for s in sp.solve(sp.diff(slope, r), r) if s.is_positive])')

prob(id=22, theme='Jacobians (wavelength to frequency bandwidth)', cat='D', pre=[1], diff=2,
     obj='Transform a bandwidth between wavelength and frequency.',
     ctx='Optical bandwidth is quoted in nm; RF bandwidth after time stretch in GHz.',
     given='λ = 1550 nm; Δλ = 1 nm and 20 nm.', unknown='Δf (GHz).', assume='Δλ ≪ λ (local linearization).',
     model='f = c/λ ⇒ |df/dλ| = c/λ²; Δf ≈ (c/λ²) Δλ.', q='Compute Δf for both bandwidths and the error of the linearization for 20 nm.',
     atype='decimals', mistake='Using Δf = c/Δλ.', verify='exact difference c/λ₁ − c/λ₂.', src='SYN',
     code='lam = sp.Rational(1550, 10**9)\nfor dl in (1, 20):\n    d = sp.Rational(dl, 10**9)\n    lin = c/lam**2*d; exact = c/(lam - d/2) - c/(lam + d/2)\n    say(f"df for {dl} nm (GHz, linear)", lin/10**9); say("   exact (GHz)", exact/10**9)')

prob(id=23, theme='Optimization (chirped pulse compression)', cat='D', pre=[20], diff=3,
     obj='Minimize pulse width over propagation distance for a chirped input.',
     ctx='A chirped pulse first compresses, then broadens, when β₂C < 0 [AGR].',
     given='T(s)/T₀ = √((1 + C s)² + s²), with normalized distance s = β₂z/T₀² and chirp parameter C = −2.',
     unknown='s* minimizing T; T_min/T₀.', assume='linearly chirped Gaussian; second-order dispersion.',
     model='[AGR] chirped Gaussian broadening factor.', q='Find s* and T_min/T₀ in general and for C = −2.', atype='closed forms; exact numbers',
     mistake='Assuming a chirped pulse can be compressed for either sign of β₂.', verify='second derivative > 0.', src='AGR',
     code='s, C = sp.symbols("s C", real=True)\nF = (1 + C*s)**2 + s**2\nsstar = sp.solve(sp.diff(F, s), s)[0]\nTmin = sp.sqrt(sp.simplify(F.subs(s, sstar)))\nprint("s* =", sstar, "  Tmin/T0 =", sp.simplify(Tmin))\nprint("C = -2:", sstar.subs(C, -2), sp.simplify(Tmin.subs(C, -2)))\nprint("F\'\' =", sp.diff(F, s, 2), "(> 0, so a minimum)")')

prob(id=24, theme='Differential equations (fiber attenuation)', cat='E', pre=[8], diff=1,
     obj='Solve a first-order linear ODE and connect it to dB.',
     ctx='Power decays exponentially along a fiber.', given='dP/dz = −α P, P(0) = P₀; 0.2 dB/km.',
     unknown='P(z); α in 1/km.', assume='constant α.', model='α [1/km] = (ln 10 / 10) × a [dB/km].',
     q='Solve the ODE with dsolve, convert 0.2 dB/km to α, and check P(50 km)/P₀ against JW-008.',
     atype='function; decimal', mistake='Using α = 0.2 /km directly (that is dB/km, not 1/km).',
     verify='Lean `JW.attenuation_solves_ode`; JW-008 gives 0.1.', src='SYN',
     code='z = sp.symbols("z", nonnegative=True); al, P0 = sp.symbols("alpha P_0", positive=True)\nP = sp.Function("P")\nsol = sp.dsolve(sp.Eq(P(z).diff(z), -al*P(z)), ics={P(0): P0})\nprint(sol)\nalpha = sp.Rational(2, 10)*sp.log(10)/10\nsay("alpha (1/km)", alpha)\nprint("P(50)/P0 =", sp.simplify((sol.rhs/P0).subs({al: alpha, z: 50})))')

prob(id=25, theme='RF photonics (photodetector bandwidth)', cat='E', pre=[12], diff=2,
     obj='Solve an RC step response and relate rise time to bandwidth.',
     ctx='Time-stretch reduces the detector bandwidth needed; this sets what "fast enough" means.',
     given='R = 50 Ω; C = 1 pF.', unknown='τ (ps); f_3dB (GHz); 10–90% rise time t_r (ps); t_r · f_3dB.',
     assume='single-pole RC; ideal step.', model='v(t) = V(1 − e^{−t/RC}); f_3dB = 1/(2πRC).',
     q='Compute all four.', atype='decimals', mistake='Confusing 1/(RC) (rad/s) with 1/(2πRC) (Hz).',
     verify='t_r f_3dB = ln 9 / (2π) ≈ 0.35, independent of R and C.', src='SYN',
     code='R = 50; Cc = sp.Rational(1, 10**12)\ntau = R*Cc; f3 = 1/(2*sp.pi*tau); tr = tau*sp.log(9)\nsay("tau (ps)", tau*10**12); say("f3dB (GHz)", f3/10**9); say("t_r (ps)", tr*10**12); say("t_r*f3dB", sp.simplify(tr*f3))')

prob(id=26, theme='Dynamical systems (carrier rate equation)', cat='E', pre=[24], diff=2,
     obj='Solve a driven first-order system and find its settling time.',
     ctx='Simplified carrier dynamics in a detector or amplifier.', given='dN/dt = G − N/τ, N(0) = 0, τ = 1 ns (synthetic).',
     unknown='N(t); steady state N_ss; time to reach 90% of N_ss.', assume='constant generation G > 0.',
     model='linear ODE with constant forcing.', q='Solve and compute t₉₀.', atype='function; decimal (ns)',
     mistake='Thinking t₉₀ = 0.9τ.', verify='t₉₀ = τ ln 10.', src='SYN',
     code='t = sp.symbols("t", nonnegative=True); G, tau = sp.symbols("G tau", positive=True)\nN = sp.Function("N")\nsol = sp.dsolve(sp.Eq(N(t).diff(t), G - N(t)/tau), ics={N(0): 0}).rhs\nprint("N(t) =", sp.simplify(sol), "; N_ss =", sp.limit(sol, t, sp.oo))\nt90 = sp.solve(sp.Eq(sol, sp.Rational(9, 10)*G*tau), t)[0]\nprint("t90 =", sp.simplify(t90)); say("t90 (ns) for tau = 1 ns", t90.subs(tau, 1))')

prob(id=27, theme='Linear algebra (Jones matrices)', cat='F', pre=[], diff=2,
     obj='Multiply Jones matrices and compute transmitted intensity.',
     ctx='Polarization control appears in modulators and fiber systems.', given='input Jones vector (1, 1)/√2; ideal polarizers at 0°, 45°, 90°.',
     unknown='intensity after x-polarizer; after x then y; after x, 45°, y.', assume='ideal lossless polarizers.',
     model='P(θ) = [[cos²θ, sinθcosθ],[sinθcosθ, sin²θ]]; intensity = |E|².', q='Compute the three intensities.', atype='exact fractions',
     mistake='Multiplying matrices in the wrong order (the first element hit by light is the rightmost).', verify='Malus law cos²θ at each step.', src='ST',
     code='def pol(t): return sp.Matrix([[sp.cos(t)**2, sp.sin(t)*sp.cos(t)], [sp.sin(t)*sp.cos(t), sp.sin(t)**2]])\nE = sp.Matrix([1, 1])/sp.sqrt(2)\nI = lambda v: sp.simplify((v.H*v)[0])\nprint("x only:", I(pol(0)*E), "| x then y:", I(pol(sp.pi/2)*pol(0)*E), "| x, 45, y:", I(pol(sp.pi/2)*pol(sp.pi/4)*pol(0)*E))')

prob(id=28, theme='Least squares (wavelength-to-time calibration)', cat='F', pre=[], diff=3,
     obj='Fit a line by least squares with exact arithmetic and interpret the slope.',
     ctx='Dispersive Fourier transform maps wavelength to time, t ≈ t₀ + (DL)(λ − λ₀) [GJ13].',
     given='synthetic measured data (λ in nm, t in ns): (1545, 0.02), (1550, 0.98), (1555, 2.01), (1560, 2.99).',
     unknown='intercept a (ns), slope b (ns/nm) = total dispersion DL; residuals.', assume='linear map (first-order dispersion only); independent equal-variance errors.',
     model='minimize ‖y − Xβ‖², X = [1, λ]; normal equations XᵀXβ = Xᵀy.', q='Find b exactly and in ps/nm. Edge case: why is one data point not enough?',
     atype='exact rational; decimal', mistake='Fitting λ as a function of t and inverting the slope (not the same estimator).',
     verify='residuals sum to zero (intercept in model); rank of X.', src='SYN',
     code='lam = sp.Matrix([1545, 1550, 1555, 1560]); tt = sp.Matrix([sp.Rational(2, 100), sp.Rational(98, 100), sp.Rational(201, 100), sp.Rational(299, 100)])\nX = sp.Matrix.hstack(sp.ones(4, 1), lam)\nbeta = (X.T*X).inv()*X.T*tt\nprint("a, b =", list(beta)); say("DL (ps/nm)", beta[1]*1000)\nprint("residual sum =", sum(tt - X*beta), "; rank X =", X.rank(), "; rank with one point =", X[:1, :].rank(), "< 2 unknowns")')

prob(id=29, theme='Eigenvalues (silicon-photonic directional coupler)', cat='F', pre=[], diff=3,
     obj='Diagonalize a coupled-mode matrix and find the coupling length.',
     ctx='Directional couplers are basic silicon-photonic building blocks.', given='M = [[β, κ],[κ, β]] (rad/m), κ = 1 mm⁻¹.',
     unknown='supermode eigenvalues and eigenvectors; full-transfer length L_c.', assume='identical lossless waveguides, weak coupling.',
     model='da/dz = −iMa; P₂(z) = sin²(κz) for light launched in guide 1.', q='Find the eigenpairs and L_c = π/(2κ) in mm.',
     atype='symbolic eigenpairs; decimal', mistake='Taking L_c = π/κ (that returns the light to guide 1).', verify='solve the ODE with the matrix exponential.', src='ST',
     code='b, k, z = sp.symbols("beta kappa z", positive=True)\nM = sp.Matrix([[b, k], [k, b]])\nprint("eigenpairs:", [(ev, mult, [list(v) for v in vecs]) for ev, mult, vecs in M.eigenvects()])\na = sp.simplify((sp.exp(-sp.I*M*z)*sp.Matrix([1, 0])))\nprint("P2(z) =", sp.simplify(sp.Abs(a[1])**2).rewrite(sp.sin))\nsay("L_c (mm) for kappa = 1/mm", sp.pi/2)')

prob(id=30, theme='Linear systems (frequency-tagged pixel unmixing)', cat='F', pre=[], diff=3,
     obj='Invert a crosstalk matrix and recognize ill-conditioning.',
     ctx='In radio-frequency-tagged fluorescence imaging, pixels are encoded on distinct beat frequencies [DBGJ13]; here a 2-pixel crosstalk model.',
     given='crosstalk matrix A = [[1, 1/10],[1/5, 1]]; synthetic measurement y = (13/10, 11/10).', unknown='pixel values x = A⁻¹y; det A; condition number.',
     assume='linear, noiseless, simplified educational model (not the published processing pipeline).', model='y = A x.',
     q='Solve for x. Edge case: what happens as both off-diagonal entries → 1?', atype='exact vector; scalar', mistake='Assuming any measured mixture can be unmixed.',
     verify='A x = y; det → 0.', src='SYN',
     code='A = sp.Matrix([[1, sp.Rational(1, 10)], [sp.Rational(1, 5), 1]]); y = sp.Matrix([sp.Rational(13, 10), sp.Rational(11, 10)])\nx = A.solve(y); print("x =", list(x), " det A =", A.det()); assert A*x == y\nimport numpy as np\nprint("condition number =", round(float(np.linalg.cond(np.array(A, dtype=float))), 4))\ne = sp.symbols("epsilon"); print("det [[1, e],[e, 1]] =", sp.Matrix([[1, e], [e, 1]]).det(), "-> 0 at e = 1: singular")')

prob(id=31, theme='Sampling (photonic time-stretch ADC)', cat='G', pre=[], diff=2,
     obj='Apply the Nyquist criterion before and after time stretch.',
     ctx='Stretching an RF waveform in time divides its bandwidth by the stretch factor M [HJ03].',
     given='RF bandwidth B = 10 GHz; stretch factor M = 10.', unknown='minimum sample rate before and after stretch.',
     assume='ideal stretch; band-limited signal.', model='f_s > 2B; B′ = B/M.', q='Compute both rates. What does the stretch NOT change?',
     atype='decimals; short explanation', mistake='Thinking stretch increases bandwidth.',
     verify='samples per stretched window equal samples per original window × 1 (same information).', src='HJ03',
     code='B = 10*10**9; M = 10\nprint("before: f_s >", 2*B/10**9, "GS/s; after:", sp.Rational(2*B, M)/10**9, "GS/s")')

prob(id=32, theme='Time stretch (stretch factor)', cat='K', pre=[31], diff=2,
     obj='Compute the stretch factor from two dispersive stages.',
     ctx='Two-stage dispersion: first stage chirps the pulse, second stretches it [HJ03].', given='D₁L₁ = 100 ps/nm; D₂L₂ = 900 ps/nm; input time window 1 ns.',
     unknown='M; output window.', assume='same-sign dispersion; first-order model.', model='M = 1 + D₂L₂/(D₁L₁) [HJ03].',
     q='Compute M and the output window. Edge case: what if D₂L₂ = 0?', atype='integers', mistake='Using M = D₂L₂/(D₁L₁) (drops the 1).',
     verify='Lean `JW.stretch_factor_ge_one`.', src='HJ03',
     code='d1, d2 = 100, 900\nM = 1 + sp.Rational(d2, d1)\nprint("M =", M, "; output window =", M*1, "ns; edge D2L2 = 0: M =", 1 + sp.Rational(0, d1))')

prob(id=33, theme='Bayes (rare-cell screening)', cat='H', pre=[], diff=2,
     obj='Compute positive predictive value with a rare condition.',
     ctx='High-throughput imaging flow cytometry screens many cells for rare targets [GTJ09, MAH17].',
     given='prevalence p = 10⁻⁵; sensitivity s = 0.99; specificity q = 0.999 (synthetic).', unknown='PPV = P(target | positive).',
     assume='independent cells; fixed s, q.', model='PPV = s p / (s p + (1 − q)(1 − p)).', q='Compute PPV. What specificity gives PPV = 0.5?',
     atype='exact rational; decimal', mistake='Equating PPV with sensitivity.', verify='Lean `JW.ppv_le_one`.', src='SYN',
     code='p, s, q = sp.Rational(1, 10**5), sp.Rational(99, 100), sp.Rational(999, 1000)\nppv = s*p/(s*p + (1 - q)*(1 - p)); say("PPV", ppv)\nQ = sp.symbols("q"); qh = sp.solve(sp.Eq(s*p/(s*p + (1 - Q)*(1 - p)), sp.Rational(1, 2)), Q)[0]\nsay("specificity for PPV = 1/2", qh, digits=10)')

prob(id=34, theme='Binomial (seeing at least one rare cell)', cat='H', pre=[], diff=2,
     obj='Compute P(at least one) and compare with the union bound.',
     ctx='How many cells must be imaged to see one rare cell?', given='n = 10⁶ cells; p = 10⁻⁶ per cell.',
     unknown='P(K ≥ 1); the bound n p; n needed for P ≥ 0.95.', assume='independent Bernoulli trials.',
     model='P(K ≥ 1) = 1 − (1 − p)ⁿ ≤ n p.', q='Compute all three.', atype='decimals; integer',
     mistake='Saying P = n p = 1 (a probability bound, not the probability).', verify='Lean `JW.at_least_one_le_np`.', src='SYN',
     code='p = sp.Rational(1, 10**6); n = 10**6\nprob1 = 1 - sp.Float(1 - p, 40)**n          # 40-digit float power: the exact fraction would have millions of digits\nsay("P(K>=1)", sp.N(prob1, 15)); print("union bound n p =", n*p)\nN = sp.symbols("N", positive=True)\nNreq = sp.ceiling(sp.log(sp.Float(sp.Rational(1, 20), 40))/sp.log(sp.Float(1 - p, 40))); print("n for P >= 0.95:", Nreq)')

prob(id=35, theme='Poisson approximation', cat='H', pre=[34], diff=2,
     obj='Quantify the error of the Poisson approximation to the binomial.',
     ctx='Rare counts are modeled as Poisson in imaging and photon counting.', given='n = 10⁶, p = 10⁻⁶, λ = np = 1.',
     unknown='P(K = 0) exact and Poisson; the difference.', assume='independent trials.', model='Binomial(n, p) ≈ Poisson(np).',
     q='Compute both and their difference.', atype='decimals (high precision)', mistake='Using the approximation when p is not small.',
     verify='high-precision evaluation.', src='SYN',
     code='p = sp.Rational(1, 10**6); n = 10**6\nexact = sp.Float(1 - p, 40)**n                # exact value to 40 digits (no million-digit fraction)\npois = sp.N(sp.exp(-1), 40); pois = sp.N(sp.exp(-1), 30)\nprint("binomial P(0) =", exact); print("Poisson  P(0) =", pois); print("difference   =", sp.N(exact - pois, 5))')

prob(id=36, theme='Shot noise and SNR', cat='H', pre=[3], diff=2,
     obj='Relate photon number to shot-noise-limited SNR and dB.',
     ctx='Each time-stretch pixel receives few photons at high frame rates (JW-003).', given='mean photon count N̄ per pixel.',
     unknown='amplitude SNR; N̄ needed for amplitude SNR = 100; power SNR in dB.', assume='Poisson photon statistics, no other noise.',
     model='mean N̄, std √N̄ ⇒ SNR_amp = √N̄; SNR_pow = N̄.', q='For N̄ = 100 find SNR_amp; find N̄ for SNR_amp = 100; express both in dB (20 log for amplitude).',
     atype='integers; dB', mistake='Using 10 log on an amplitude ratio.', verify='SNR scaling ∝ √N̄.', src='SYN',
     code='N = sp.symbols("N", positive=True)\nsnr = N/sp.sqrt(N); print("SNR_amp =", sp.simplify(snr))\nprint("N = 100:", snr.subs(N, 100), "=", 20*sp.log(10, 10), "dB")\nprint("N for SNR 100:", sp.solve(sp.Eq(snr, 100), N)[0], "->", 20*sp.log(100, 10), "dB")')

prob(id=37, theme='Gaussian noise (false alarms)', cat='H', pre=[], diff=2,
     obj='Compute Gaussian tail probabilities and expected false alarms.',
     ctx='Thresholding detector noise at 3σ to find events.', given='X ~ N(0, σ²); threshold 3σ; 10⁶ independent samples.',
     unknown='P(|X| > 3σ); expected exceedances.', assume='independent, identically distributed Gaussian noise.',
     model='P(|X| > kσ) = erfc(k/√2).', q='Compute both.', atype='decimal; count', mistake='Using one tail when the detector is two-sided.',
     verify='SymPy stats cdf.', src='SYN',
     code='from sympy.stats import Normal, cdf\nX = Normal("X", 0, 1)\np2 = 2*(1 - cdf(X)(3)); say("P(|X|>3)", sp.simplify(p2), digits=8)\nsay("expected exceedances in 1e6", p2*10**6, digits=6)')

prob(id=38, theme='Exponential arrivals (cell coincidences in flow)', cat='H', pre=[], diff=2,
     obj='Use the exponential distribution for inter-arrival times.',
     ctx='Cells arrive at random in a flow channel; two cells too close together overlap in one frame.',
     given='arrival rate r = 10⁵ cells/s (synthetic); window τ = 1 µs.', unknown='P(gap < τ); memoryless property.',
     assume='Poisson arrivals.', model='gap G ~ Exp(r): P(G < τ) = 1 − e^{−rτ}.', q='Compute P(G < τ) and verify memorylessness symbolically.',
     atype='decimal; identity', mistake='Using rτ = 0.1 as the probability (only a first-order approximation).', verify='Lean `JW.exp_memoryless`.', src='SYN',
     code='r = 10**5; tau = sp.Rational(1, 10**6)\nsay("P(G < 1 us)", 1 - sp.exp(-r*tau)); print("first-order approx r*tau =", r*tau)\nR, S, T = sp.symbols("r s t", positive=True)\nprint("memoryless:", sp.simplify(sp.exp(-R*(S + T))/sp.exp(-R*S) - sp.exp(-R*T)) == 0)')

prob(id=39, theme='Gamma distribution (waiting for k events)', cat='H', pre=[38], diff=3,
     obj='Use the Gamma (Erlang) distribution for the k-th arrival.',
     ctx='Time until the 3rd rare cell arrives.', given='rate r = 10⁵ s⁻¹; k = 3.', unknown='mean of T₃; P(T₃ ≤ 30 µs).',
     assume='Poisson arrivals.', model='T₃ ~ Gamma(3, 1/r); P(T₃ ≤ t) = 1 − e^{−rt}(1 + rt + (rt)²/2).', q='Compute both.',
     atype='decimal', mistake='Thinking P(T₃ ≤ mean) = 1/2 (the Gamma is skewed).', verify='SymPy stats Gamma cdf vs closed form.', src='SYN',
     code='from sympy.stats import Gamma, cdf, E\nr = 10**5; T3 = Gamma("T3", 3, sp.Rational(1, r))\nsay("mean (us)", E(T3)*10**6)\nP = cdf(T3)(sp.Rational(30, 10**6)); say("P(T3 <= 30 us)", sp.simplify(P))\nx = 3; print("closed form check:", sp.simplify(P - (1 - sp.exp(-x)*(1 + x + sp.Rational(x**2, 2)))) == 0)')

prob(id=40, theme='Lognormal (cell size)', cat='H', pre=[], diff=2,
     obj='Distinguish mean and median of a lognormal distribution.',
     ctx='Cell diameters are often modeled as lognormal (synthetic model).', given='ln(d/1 µm) ~ N(μ, σ²) with μ = ln 10, σ = 0.2.',
     unknown='median, mean, P(d > 12 µm).', assume='lognormal model.', model='median = e^μ; mean = e^{μ+σ²/2}.', q='Compute all three.',
     atype='decimals (µm)', mistake='Reporting e^μ as the mean.', verify='SymPy stats LogNormal.', src='SYN',
     code='mu, sig = sp.log(10), sp.Rational(1, 5)\nsay("median (um)", sp.exp(mu)); say("mean (um)", sp.exp(mu + sig**2/2))\npt = sp.erfc((sp.log(12) - mu)/(sig*sp.sqrt(2)))/2       # P(d > 12) = 1 - Phi((ln 12 - mu)/sigma)\nsay("P(d > 12 um)", pt)\nzz = sp.symbols("z", real=True)                            # d = exp(mu + sigma Z), Z standard normal\nm_int = sp.integrate(sp.exp(mu + sig*zz)*sp.exp(-zz**2/2)/sp.sqrt(2*sp.pi), (zz, -sp.oo, sp.oo))\nprint("mean by integration agrees:", sp.simplify(m_int - sp.exp(mu + sig**2/2)) == 0)')

prob(id=41, theme='Monte Carlo (rare-event estimation)', cat='N', pre=[37], diff=3,
     obj='Estimate a small probability by simulation with a reproducible seed and a confidence interval.',
     ctx='Rare events need many samples; this motivates the importance-sampling problems in later batches.',
     given='Z ~ N(0, 1); target p = P(Z > 3); n = 10⁶ samples; seed 2026.', unknown='estimate p̂, 95% CI, relative standard error.',
     assume='pseudo-random numbers adequate (simulated data, not experimental).', model='p̂ = (1/n)Σ 1{Z_i > 3}; SE = √(p̂(1−p̂)/n).',
     q='Run the simulation and compare to the exact value. Edge case: n = 1000.', atype='decimal with CI', mistake='Reporting p̂ = 0 with no uncertainty for small n.',
     verify='exact p = erfc(3/√2)/2.', src='SYN',
     code='import numpy as np\nrng = np.random.default_rng(2026)\nn = 10**6; hits = int((rng.standard_normal(n) > 3).sum()); ph = hits/n; se = (ph*(1 - ph)/n)**0.5\nexact = float(sp.erfc(3/sp.sqrt(2))/2)\nprint(f"p_hat = {ph:.6f} +/- {1.96*se:.6f} (95%), exact {exact:.6f}, rel. SE {se/ph:.3f}")\nsmall = int((np.random.default_rng(2026).standard_normal(1000) > 3).sum()); print("n = 1000: hits =", small, "(far too few to estimate p)")')

prob(id=42, theme='Confidence intervals (proportions)', cat='O', pre=[41], diff=3,
     obj='Compute a Wilson score interval and compare with the normal (Wald) interval.',
     ctx='Reporting how many cells in a synthetic sample were flagged.', given='x = 37 positives out of n = 10 000 (synthetic); z = 1.96.',
     unknown='Wald and Wilson 95% intervals.', assume='independent trials.',
     model='Wilson: (p̂ + z²/2n ± z√(p̂(1−p̂)/n + z²/4n²)) / (1 + z²/n) [WIL].', q='Compute both; edge case x = 0.',
     atype='intervals', mistake='Using Wald with x = 0 (gives the impossible interval [0, 0]).', verify='Wilson interval contains p̂ and stays in [0, 1].', src='WIL',
     code='z = sp.Rational(196, 100)\ndef wilson(x, n):\n    ph = sp.Rational(x, n); den = 1 + z**2/n; cen = (ph + z**2/(2*n))/den; hw = z*sp.sqrt(ph*(1 - ph)/n + z**2/(4*n**2))/den\n    return sp.N(cen - hw, 5), sp.N(cen + hw, 5)\ndef wald(x, n):\n    ph = sp.Rational(x, n); hw = z*sp.sqrt(ph*(1 - ph)/n); return sp.N(ph - hw, 5), sp.N(ph + hw, 5)\nprint("x = 37: Wald", wald(37, 10**4), " Wilson", wilson(37, 10**4))\nprint("x = 0:  Wald", wald(0, 10**4), " Wilson", wilson(0, 10**4))')

prob(id=43, theme='Rare events (exponential intensity statistics)', cat='I', pre=[38], diff=2,
     obj='Compute exceedance probabilities for exponentially distributed intensity.',
     ctx='Fully developed speckle and thermal light have exponential intensity statistics [GOOD]; optical rogue waves are defined by departures from such light-tailed statistics [SRKJ07].',
     given='I ~ Exp with mean ⟨I⟩; thresholds k⟨I⟩, k = 2, 4, 8.', unknown='P(I > k⟨I⟩).', assume='exponential intensity statistics (no rogue-wave mechanism).',
     model='P(I > k⟨I⟩) = e^{−k}.', q='Compute for each k; how many events above 8⟨I⟩ are expected in 10⁶ shots?', atype='decimals; count',
     mistake='Using Gaussian tails for intensity (intensity is non-negative and exponential here).', verify='Lean `JW.exceed_8_mean_small` (e^{−8} ≤ 1/256).', src='GOOD',
     code='for k in (2, 4, 8):\n    say(f"P(I > {k}<I>)", sp.exp(-k), digits=5)\nsay("expected events above 8<I> in 1e6 shots", 10**6*sp.exp(-8), digits=5)')

prob(id=44, theme='Rare events (heavy vs light tails)', cat='I', pre=[43], diff=3,
     obj='Compare a heavy-tailed (Pareto) and a light-tailed (exponential) distribution with the same mean.',
     ctx='Long-tailed histograms are the statistical signature reported for optical rogue waves [SRKJ07]; here a synthetic comparison.',
     given='exponential with mean 1; Pareto with α = 3, x_m = 2/3 (mean αx_m/(α−1) = 1).', unknown='P(X > 10), P(X > 20) for both; ratios.',
     assume='synthetic models only.', model='exp: e^{−x}; Pareto: (x_m/x)^α for x ≥ x_m.', q='Compute the four tail probabilities and the two ratios.',
     atype='decimals', mistake='Judging tails by the mean or variance alone.', verify='both means equal 1 (SymPy integral).', src='SYN',
     code='x = sp.symbols("x", positive=True); a = 3; xm = sp.Rational(2, 3)\nprint("Pareto mean =", sp.integrate(x*a*xm**a/x**(a + 1), (x, xm, sp.oo)))\nfor X in (10, 20):\n    pe, pp = sp.exp(-X), (xm/X)**a\n    print(f"x = {X}: exp {sp.N(pe, 4)}, Pareto {sp.N(pp, 4)}, ratio {sp.N(pp/pe, 4)}")')

prob(id=45, theme='Rare events (threshold convention and edge cases)', cat='I', pre=[44], diff=2,
     obj='Apply a published threshold convention carefully, including ties.',
     ctx='A common rogue-wave convention flags events exceeding twice the "significant" value (mean of the highest third) [KP]. This is a convention, not the specific analysis of [SRKJ07].',
     given='synthetic intensities: 1, 2, 1, 3, 2, 1, 9, 2, 1, 4, 2, 1.', unknown='significant value I_s; threshold 2I_s; flagged events.',
     assume='strict inequality I > 2I_s; highest third = top 4 of 12.', model='I_s = mean of the top ⌈n/3⌉ values.',
     q='Compute I_s and the flagged events. Edge case: the largest value equals the threshold.', atype='exact rational; list', mistake='Using ≥ silently, or including all values in I_s.',
     verify='recompute with ≥ to see the difference.', src='KP',
     code='data = [1, 2, 1, 3, 2, 1, 9, 2, 1, 4, 2, 1]\ntop = sorted(data, reverse=True)[:len(data)//3]\nIs = sp.Rational(sum(top), len(top)); thr = 2*Is\nprint("top third:", top, " I_s =", Is, " threshold =", thr)\nprint("flagged with >  :", [v for v in data if v > thr]); print("flagged with >= :", [v for v in data if v >= thr])')

prob(id=46, theme='Group delay', cat='J', pre=[14], diff=1,
     obj='Compute pulse delay from the group index.',
     ctx='Delay through kilometer-scale dispersive fiber sets latency in time-stretch instruments.', given='L = 1 km; n_g = 1.46260 (bulk fused silica at 1550 nm, from the Malitson Sellmeier fit [MAL]).',
     unknown='group delay τ_g (µs).', assume='bulk material only (real fiber adds waveguide dispersion).', model='τ_g = L n_g / c.',
     q='Compute τ_g.', atype='decimal', mistake='Using the phase index n instead of n_g.', verify='dimension: m / (m/s) = s.', src='MAL',
     code='L = 1000; ng = sp.Rational(146260, 100000)\nsay("tau_g (us)", L*ng/c*10**6)')

prob(id=47, theme='Fiber optics (numerical aperture)', cat='J', pre=[], diff=2,
     obj='Compute critical angle and numerical aperture of a step-index fiber.',
     ctx='Light guidance in the fibers used for time stretch.', given='core n₁ = 1.450; cladding n₂ = 1.445 (synthetic step-index values).',
     unknown='θ_c (deg); NA; acceptance half-angle in air (deg).', assume='step index, ray optics.', model='sin θ_c = n₂/n₁; NA = √(n₁² − n₂²).',
     q='Compute all three. Edge case: n₂ > n₁.', atype='decimals', mistake='Measuring θ_c from the fiber axis instead of the normal.',
     verify='Lean `JW.na_pos`.', src='ST',
     code='n1, n2 = sp.Rational(1450, 1000), sp.Rational(1445, 1000)\nsay("theta_c (deg)", sp.deg(sp.asin(n2/n1))); NA = sp.sqrt(n1**2 - n2**2); say("NA", NA); say("acceptance (deg)", sp.deg(sp.asin(NA)))\nprint("edge n2 > n1: NA^2 =", (n2**2 - n1**2), "< 0 -> no guidance")')

prob(id=48, theme='Silicon photonics (index contrast)', cat='J', pre=[47], diff=2,
     obj='Quantify the high index contrast of silicon photonics.',
     ctx='Silicon photonics confines light tightly because silicon\'s index is much higher than silica\'s.',
     given='n_Si ≈ 3.48, n_SiO₂ ≈ 1.444 at 1550 nm (approximate textbook values).', unknown='normal-incidence Fresnel reflectance R at Si/air and Si/SiO₂; √(n₁² − n₂²).',
     assume='lossless, normal incidence.', model='R = ((n₁ − n₂)/(n₁ + n₂))².', q='Compute both reflectances and the index-contrast "NA".',
     atype='decimals', mistake='Interpreting a value of √(n₁²−n₂²) > 1 as an angle (it means every incidence angle from air is guided).', verify='R ∈ [0, 1).', src='ST',
     code='nSi, nOx = sp.Rational(348, 100), sp.Rational(1444, 1000)\nR = lambda a, b: ((a - b)/(a + b))**2\nsay("R Si/air", R(nSi, 1)); say("R Si/SiO2", R(nSi, nOx)); say("sqrt(n1^2 - n2^2)", sp.sqrt(nSi**2 - nOx**2))')

prob(id=49, theme='Silicon photonics (Raman Stokes wavelength)', cat='J', pre=[1], diff=2,
     obj='Convert a Raman frequency shift to a Stokes wavelength.',
     ctx='Stimulated Raman scattering in silicon (optical phonon near 15.6 THz, 520 cm⁻¹) enabled the silicon Raman laser [BJ04].',
     given='pump λ_p = 1550 nm (synthetic choice); Raman shift Δν̃ = 520 cm⁻¹.', unknown='shift Δf (THz); Stokes λ_s (nm).',
     assume='Stokes line at f_s = f_p − Δf.', model='Δf = c Δν̃; λ_s = c/(c/λ_p − Δf).', q='Compute Δf and λ_s.',
     atype='decimals', mistake='Subtracting the shift in wavelength (nm) instead of frequency.', verify='λ_s > λ_p (Stokes is red-shifted).', src='BJ04',
     code='lp = sp.Rational(1550, 10**9); dnu = 520*100            # 520 cm^-1 = 52000 m^-1\ndf = c*dnu; say("shift (THz)", df/10**12)\nls = c/(c/lp - df); say("Stokes wavelength (nm)", ls*10**9)\nassert ls > lp')

prob(id=50, theme='Research integrity (source classification)', cat='O', pre=[], diff=2,
     obj='Classify research claims as verified, historical, inferred, or unknown, using the source table in Section 2.',
     ctx='Every later batch must keep claims about the laboratory honest.', given='six claims (listed in the code cell) and the Section 2 source table.',
     unknown='a label for each claim, with the source that justifies it.', assume='only the sources in Section 2 count.',
     model='verified = primary/authoritative source found; historical = verified but not current; inferred = suggested, not stated; unknown = no source.',
     q='Label each claim.', atype='labels with justification', mistake='Treating a lab web page, Wikipedia, or a press summary as equivalent to an award database.',
     verify='compare with the answer key; check each URL yourself.', src='SYN',
     code='claims = {\n "NIH R21 GM107924 (two-photon/confocal kHz-frame-rate microscope), Jalali PI, FY2014-2016": "historical (verified in NIH RePORTER; project ended 2018-05-31)",\n "NSF 0925638 (stimulated rogue waves), 2009-2013": "historical (verified in NSF Award Search)",\n "The lab currently holds an active federal grant": "unknown / not publicly verified (no active award found in NIH RePORTER or NSF Award Search)",\n "Jalali was at Bell Labs 1988-1992": "verified (UCLA Samueli profile); note the lab page says until 2002: discrepancy",\n "DARPA funded the time-stretch ADC": "unknown / not publicly verified here (would need a paper acknowledgment or DARPA record)",\n "Jalali is professor emeritus": "verified (UCLA Samueli profile, accessed 2026-10-12)",\n}\nfor k, v in claims.items():\n    print(f"- {k}\\n    -> {v}")')
