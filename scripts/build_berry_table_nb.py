"""Build notebooks/berry_phase_table.ipynb -- a one-cell Berry-phase lookup (6 values).

The output is baked in (a stream output) so the table shows even without running a
kernel; re-run the cell to plug in your own angles."""
import pathlib
import numpy as np
import nbformat as nbf

CODE = '''import numpy as np

# Berry phase of a spin-1/2 whose field traces a cone of half-angle theta0 on the sphere.
# Geometric law:  gamma = -Omega/2,   Omega = 2*pi*(1 - cos theta0)   (enclosed solid angle).
def berry_phase(theta0_deg):
    th = np.radians(theta0_deg)
    Omega = 2*np.pi*(1 - np.cos(th))      # solid angle (steradians)
    gamma = -Omega/2                      # Berry (geometric) phase
    return Omega, gamma

print(f"{'theta0':>7} {'Omega(sr)':>11} {'gamma(rad)':>11} {'gamma(deg)':>11} {'gamma/pi':>9}")
for theta0 in [30, 45, 60, 90, 120, 150]:          # <-- your 6 values; change freely
    Omega, gamma = berry_phase(theta0)
    print(f"{theta0:>6}d {Omega:>11.4f} {gamma:>11.4f} {np.degrees(gamma):>11.2f} {gamma/np.pi:>9.4f}")'''


def run_table():
    lines = ["%7s %11s %11s %11s %9s" % ("theta0", "Omega(sr)", "gamma(rad)", "gamma(deg)", "gamma/pi")]
    for theta0 in [30, 45, 60, 90, 120, 150]:
        th = np.radians(theta0)
        Omega = 2 * np.pi * (1 - np.cos(th)); gamma = -Omega / 2
        lines.append("%6dd %11.4f %11.4f %11.2f %9.4f"
                     % (theta0, Omega, gamma, np.degrees(gamma), gamma / np.pi))
    return "\n".join(lines) + "\n"


md = nbf.v4.new_markdown_cell(
    "# Berry phase -- quick lookup (no re-derivation)\n\n"
    "Spin-1/2 dragged around a cone of half-angle `theta0` on the sphere picks up the "
    "geometric phase\n$$\\gamma = -\\tfrac12\\,\\Omega,\\qquad \\Omega = 2\\pi(1-\\cos\\theta_0).$$\n"
    "Anchors: **60 deg -> -pi/2**, **90 deg (equator) -> -pi**, **120 deg -> -3pi/2**. "
    "Edit the list to plug in your own angles.")
code = nbf.v4.new_code_cell(CODE)
code.outputs = [nbf.v4.new_output("stream", name="stdout", text=run_table())]
code.execution_count = 1

nb = nbf.v4.new_notebook()
nb.cells = [md, code]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = pathlib.Path("notebooks/berry_phase_table.ipynb")
nbf.write(nb, out)
print("wrote", out, "with baked-in output")
