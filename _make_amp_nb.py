import json, os

cells = []
def md(src): return {'cell_type':'markdown','metadata':{},'source':src}
def code(src): return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':src}

cells.append(md(
'# Amplitude, Power, and Complex Exponentials\n'
'### The thing Griffiths assumes you already know\n\n'
'**Keyboard nav:** `j/k` move cells, `Enter` edit, `Shift+Enter` run, `Esc` command mode  \n'
'**Run all:** Kernel menu -> Restart and Run All\n'
))

cells.append(code(
'import numpy as np\n'
'import matplotlib.pyplot as plt\n'
'from sympy import *\n'
't, omega, phi, A = symbols("t omega phi A", real=True)\n'
'print("ready")\n'
))

cells.append(md(
'---\n'
'## S1 -- Euler Identity\n\n'
'```\n'
'e^(i*phi) = cos(phi) + i*sin(phi)\n'
'```\n'
'The PHYSICAL field is always the **real part**. Complex form is bookkeeping.\n'
))

cells.append(code(
'phi_vals = np.linspace(0, 2*np.pi, 1000)\n'
'lhs = np.exp(1j * phi_vals)\n'
'rhs = np.cos(phi_vals) + 1j*np.sin(phi_vals)\n'
'print("max |e^(i*phi) - (cos+i*sin)| =", np.max(np.abs(lhs-rhs)), " (machine zero)")\n'
'\n'
'fig, ax = plt.subplots(figsize=(5,5))\n'
'ax.plot(np.cos(phi_vals), np.sin(phi_vals), "b-", lw=1)\n'
'for p, label in [(0,"0"), (np.pi/6,"pi/6"), (np.pi/4,"pi/4"),\n'
'                  (np.pi/3,"pi/3"), (np.pi/2,"pi/2"), (np.pi,"pi")]:\n'
'    ax.annotate("  "+label, (np.cos(p), np.sin(p)), fontsize=8)\n'
'    ax.plot([0,np.cos(p)],[0,np.sin(p)],"r-",alpha=0.4)\n'
'    ax.plot(np.cos(p), np.sin(p), "ro", ms=4)\n'
'ax.set_aspect("equal"); ax.grid(True, alpha=0.3)\n'
'ax.set_title("e^(i*phi) traces the unit circle -- |amplitude| = 1 always")\n'
'plt.tight_layout(); plt.show()\n'
))

cells.append(md(
'---\n'
'## S2 -- Amplitude vs Power: The Factor of 2\n\n'
'```\n'
'E(t) = A * cos(omega*t + phi)        <- real, physical\n'
'     = Re[ A * e^(i*(omega*t+phi)) ] <- complex form\n\n'
'Intensity (power per area):\n'
'  I = epsilon_0 * c * <E^2>\n'
'    = epsilon_0 * c * A^2 / 2        <- /2 from <cos^2> = 1/2\n\n'
'THE KEY RULE:\n'
'  |E_complex|^2 = A^2          no factor of 1/2\n'
'  <E_real^2>    = A^2 / 2      1/2 from time average\n'
'  Power ~ A^2, NOT A\n'
'```\n'
))

cells.append(code(
'A_val = 3.0\n'
'omega_val = 2*np.pi\n'
'phi_val = np.pi/4\n'
't_vals = np.linspace(0, 10, 10000)\n\n'
'E_real    = A_val * np.cos(omega_val*t_vals + phi_val)\n'
'E_complex = A_val * np.exp(1j*(omega_val*t_vals + phi_val))\n\n'
'time_avg_sq   = np.mean(E_real**2)\n'
'complex_mod_sq = np.mean(np.abs(E_complex)**2)\n'
'analytical    = A_val**2 / 2\n\n'
'print(f"A = {A_val}")\n'
'print(f"<E_real^2>       = {time_avg_sq:.6f}  (numerical)")\n'
'print(f"A^2/2            = {analytical:.6f}  (analytical)")\n'
'print(f"|E_complex|^2    = {complex_mod_sq:.6f}  = A^2 (no 1/2)")\n'
'print()\n'
'print("GS measures I = |E|^2 -- complex amplitude directly, no 1/2 confusion")\n'
))

cells.append(md(
'---\n'
'## S3 -- Power in 3D: Poynting Vector\n\n'
'```\n'
'S = E x H          [W/m^2]  Poynting vector, direction = propagation\n'
'I = <|S|>          [W/m^2]  intensity\n\n'
'Plane wave in z:\n'
'  E = E0 * cos(kz - omega*t) * x_hat\n'
'  H = E0/eta * cos(kz - omega*t) * y_hat    eta = 377 ohm in vacuum\n'
'  <S> = E0^2 / (2*eta) * z_hat\n'
'```\n'
))

cells.append(code(
'eta0 = 377.0\n'
'A_field = 1000.0  # V/m\n\n'
't_vals = np.linspace(0, 2, 10000)\n'
'omega_val = 2*np.pi\n'
'E_t = A_field * np.cos(omega_val * t_vals)\n'
'H_t = (A_field/eta0) * np.cos(omega_val * t_vals)\n'
'S_t = E_t * H_t\n\n'
'S_avg = np.mean(S_t)\n'
'S_anal = A_field**2 / (2 * eta0)\n\n'
'print(f"E amplitude:         {A_field:.1f} V/m")\n'
'print(f"<S> numerical:       {S_avg:.4f} W/m^2")\n'
'print(f"A^2/(2*eta) analyt:  {S_anal:.4f} W/m^2")\n'
'print(f"Sunlight intensity:  ~1361 W/m^2")\n'
'print(f"Sun E-field amp:     {np.sqrt(2*eta0*1361):.0f} V/m")\n'
))

cells.append(md(
'---\n'
'## S4 -- SymPy: Symbolic Verification\n'
))

cells.append(code(
'x, a_s = symbols("x a", real=True, positive=True)\n\n'
'# Gaussian FT pair (most important in QM and GS)\n'
'f = exp(-a_s * x**2)\n'
'print("f(x) =", f)\n'
'print("FT{f} via known result: sqrt(pi/a) * exp(-k^2/(4a))")\n'
'print()\n\n'
'# Verify Parseval: integral |f|^2 dx = integral |F|^2 dk / (2pi)\n'
'from sympy import integrate, oo, pi as spi\n'
'parseval_lhs = integrate(f**2, (x, -oo, oo))\n'
'print("Parseval LHS (x-space):", simplify(parseval_lhs))\n'
'k_s = symbols("k", real=True)\n'
'F = sqrt(spi/a_s) * exp(-k_s**2/(4*a_s))\n'
'parseval_rhs = integrate(F**2, (k_s, -oo, oo)) / (2*spi)\n'
'print("Parseval RHS (k-space):", simplify(parseval_rhs))\n'
'print("Equal:", simplify(parseval_lhs - parseval_rhs) == 0)\n'
))

cells.append(md(
'---\n'
'## S5 -- Practice Problems\n\n'
'**P1.** `E(t) = 5*cos(2*pi*3e14*t)` V/m. What is the time-averaged intensity in W/m^2?\n\n'
'**P2.** A laser has I = 1 MW/m^2. What is E0?\n\n'
'**P3.** In GS, if you double input power, does recovered phase change?\n\n'
'**P4.** Why does `|e^(i*phi)| = 1` always?\n'
))

cells.append(code(
'eta = 377.0\n\n'
'# P1\n'
'E0 = 5.0\n'
'I_P1 = E0**2 / (2*eta)\n'
'print(f"P1: I = {I_P1:.5f} W/m^2")\n\n'
'# P2\n'
'I_laser = 1e6\n'
'E_P2 = np.sqrt(2*eta*I_laser)\n'
'print(f"P2: E0 = {E_P2:.1f} V/m  = {E_P2/1e6:.3f} MV/m")\n\n'
'# P3\n'
'print("P3: phase is INVARIANT to power scale")\n'
'print("    I_new = 4*I  ->  sqrt(I_new)*exp(i*phi) = 2*sqrt(I)*exp(i*phi)")\n'
'print("    GS amplitude constraint: replaces magnitude, KEEPS phase")\n'
'print("    This is why GS works when you dont know input power")\n\n'
'# P4\n'
'phi_test = np.linspace(0, 100*np.pi, 100000)\n'
'mods = np.abs(np.exp(1j * phi_test))\n'
'print(f"P4: max deviation from 1: {np.max(np.abs(mods-1)):.2e}  (cos^2+sin^2=1)")\n'
))

cells.append(md(
'---\n'
'## S6 -- Connection to This Project\n\n'
'```\n'
'Detector measures:  I(t) = |E(t)|^2      power -- phase is GONE\n'
'GS recovers:        phi(t) = angle(E(t)) phase from two measurements\n\n'
'Heisenberg analogy:\n'
'  Measured |E|^2  <->  position (you know WHERE power is in time)\n'
'  Lost angle(E)   <->  momentum (you lost the phase)\n'
'  I1 + I2 (two dispersions) give enough constraints to recover both\n\n'
'amplitude_constraint(E, I_meas):\n'
'  return sqrt(I_meas) * exp(i * angle(E))\n'
'            ^^ power        ^^ phase kept\n'
'```\n'
))

nb = {
    'nbformat': 4, 'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'},
        'language_info': {'name':'python','version':'3.12.0'}
    },
    'cells': cells
}

os.makedirs('notebooks', exist_ok=True)
path = 'notebooks/amplitude_power_complex.ipynb'
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print(f'Written: {path}  ({len(cells)} cells)')
