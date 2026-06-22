"""Build notebooks/griffiths_ch9_dispersion.ipynb -- Ch 9/10 -> the dispersion receiver."""
import pathlib
import nbformat as nbf

md = lambda s: nbf.v4.new_markdown_cell(s)
co = lambda s: nbf.v4.new_code_cell(s)
nb = nbf.v4.new_notebook()

nb.cells = [
md("""# Griffiths E&M Ch 9 & 10 -> the dispersion-assisted optics receiver
### absorption, dispersion, and the operator the Gerchberg-Saxton receiver inverts

Chapter 9.4 (Absorption and Dispersion) is not background for this repo -- it IS the
project. A Lorentz oscillator gives a complex permittivity, so the refractive index
$n(\\omega)=\\sqrt{\\varepsilon}$ has a real part (**dispersion** -- phase speed depends on
frequency) and an imaginary part (**absorption**). Dispersion spreads a short pulse;
propagating a distance $L$ multiplies the spectrum by the **dispersion operator**
$$H(\\omega)=e^{\\,i\\beta_2\\omega^2 L/2},\\qquad \\beta_2=\\frac{d^2k}{d\\omega^2},$$
and the dispersion-assisted GS receiver **inverts exactly this** to recover the phase a
detector cannot see. Chapter 10's retarded potentials supply the causality that ties
absorption and dispersion together (Kramers-Kronig). Uses `dgs/em_dispersion.py`.
Civilian education."""),

co("""import numpy as np, matplotlib.pyplot as plt
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))
from dgs import em_dispersion as ed
w0, g, wp = 1.0, 0.05, 0.4
print("ready")"""),

md("""## 1. Ch 9.4.3 -- the Lorentz model: dispersion + absorption

The frequency-dependent permittivity gives a complex index. **Re(n)** rises with
frequency (normal dispersion) except in a narrow band around the resonance, where it
*falls* -- **anomalous dispersion** -- precisely where **Im(n)** (absorption) peaks. You
cannot have one without the other."""),
co("""w = np.linspace(0.2, 2.0, 2000)
n = ed.lorentz_index(w, w0, g, wp)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(w, n.real, lw=2); ax[0].axvline(w0, ls=":", color="gray")
ax[0].set(xlabel="omega/omega0", ylabel="Re(n)", title="dispersion: Re(n) (anomalous near resonance)")
ax[1].plot(w, n.imag, "C3", lw=2); ax[1].axvline(w0, ls=":", color="gray")
ax[1].set(xlabel="omega/omega0", ylabel="Im(n)", title="absorption: Im(n) peaks at the resonance")
for a in ax: a.grid(alpha=0.3)
plt.tight_layout(); plt.show()"""),

md("""## 2. Phase velocity vs group velocity

A single frequency's crests travel at $v_p=c/n$; a pulse envelope travels at
$v_g=d\\omega/dk=c/(n+\\omega\\,dn/d\\omega)$. In normal dispersion ($dn/d\\omega>0$) the
envelope lags the crests, $v_g<v_p$ -- the signal slows even as individual waves run
ahead."""),
co("""ws = np.linspace(0.3, 0.85, 500)
nr = ed.lorentz_index(ws, w0, g, wp).real
vp = ed.phase_velocity(ws, nr); vg = ed.group_velocity(ws, nr)
plt.figure(figsize=(6.8,4))
plt.plot(ws, vp/ed.C, lw=2, label="phase velocity v_p/c")
plt.plot(ws, vg/ed.C, lw=2, label="group velocity v_g/c")
plt.xlabel("omega/omega0"); plt.ylabel("velocity / c"); plt.legend()
plt.title("normal dispersion: v_g < v_p (both < c here)"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()"""),

md("""## 3. GVD spreads a pulse -- the thing the receiver must undo

A transform-limited pulse is built from many frequencies that travel at slightly
different speeds, so it **broadens** as it propagates -- the more $\\beta_2 L$, the wider.
That spreading is the dispersion the receiver measures and exploits."""),
co("""t = np.linspace(-50, 50, 8192); pulse = np.exp(-t**2/(2*2.0**2))
plt.figure(figsize=(7,4))
for b2L, c in [(0,"C0"), (20,"C1"), (80,"C3")]:
    out = ed.disperse_pulse(pulse, t, b2L, 1.0)
    plt.plot(t, np.abs(out)**2/np.max(np.abs(pulse)**2), c,
             label=f"beta2 L={b2L}: width {ed.pulse_width(t,out):.1f}")
plt.xlabel("time"); plt.ylabel("|pulse|^2"); plt.legend()
plt.title("group-velocity dispersion broadens a transform-limited pulse"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()"""),

md("""## 4. The dispersion operator, and the Gerchberg-Saxton move

Propagation is just multiplication by $H(\\omega)=e^{i\\beta_2\\omega^2 L/2}$ in the
frequency domain -- a **unitary** all-pass filter (it moves energy in time but never
loses it). So **un-dispersing** is exact: apply $H^{-1}=e^{-i\\beta_2\\omega^2 L/2}$ and the
pulse comes back perfectly. Two different dispersions give two intensity views of the
same field -- the diversity GS uses to solve for the hidden phase."""),
co("""there = ed.disperse_pulse(pulse, t, 80, 1.0)
back = ed.disperse_pulse(there, t, -80, 1.0)
print(f"dispersion is unitary (energy): in={np.sum(np.abs(pulse)**2):.3f}, "
      f"out={np.sum(np.abs(there)**2):.3f}")
print(f"undisperse recovers the pulse: max error = {np.max(np.abs(back-pulse)):.2e}")
plt.figure(figsize=(7,3.4))
plt.plot(t, np.abs(pulse)**2, lw=3, alpha=0.4, label="original")
plt.plot(t, np.abs(there)**2, label="dispersed (+L)")
plt.plot(t, np.abs(back)**2, "k--", label="un-dispersed (-L) = original")
plt.xlabel("time"); plt.ylabel("|pulse|^2"); plt.legend(); plt.title("disperse then undisperse = identity (the GS step)")
plt.tight_layout(); plt.show()"""),

md("""## 5. Chapter 10 -- causality ties absorption to dispersion

Chapter 10's **retarded potentials** say a field responds only to what the source did
in the past (signals travel at $c$). That causality forces the real and imaginary parts
of $n(\\omega)$ to be **Kramers-Kronig** partners -- you cannot design a dispersion curve
without the matching absorption (built in `dgs.causality`). So the receiver's dispersion
and any loss are not independent knobs; they are two faces of one causal response."""),

md("""## Takeaway

1. **Ch 9.4 = the project:** the Lorentz $n(\\omega)$ has dispersion (Re) and absorption
   (Im); the two are inseparable.
2. Dispersion spreads a pulse via $\\beta_2$; propagation is the **unitary operator**
   $H=e^{i\\beta_2\\omega^2L/2}$, which the GS receiver **inverts** to recover phase.
3. **Ch 10 causality** (retarded potentials) makes absorption and dispersion
   Kramers-Kronig partners.

Griffiths Chapters 9 and 10 are not a detour from the optics research -- they are its
foundation. Civilian education."""),
]

nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = pathlib.Path("notebooks/griffiths_ch9_dispersion.ipynb")
nbf.write(nb, out)
print("wrote", out, "with", len(nb.cells), "cells")
