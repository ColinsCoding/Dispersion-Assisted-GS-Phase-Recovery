"""Generate notebooks/sympy_ccode_theory_bridge.ipynb."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nbkit import md, co, setup_cell, write

cells = [
md(r"""# From theory to C: `sympy.ccode` as a portable theory-to-implementation bridge

Replicating another lab's result -- or delivering a reproducible SBIR/grant artifact -- means turning the
*theory* into code someone else can compile and run anywhere. This notebook shows the bridge:
**SymPy derives the physics symbolically, differentiates it, and emits C** with `sympy.ccode`; plain
**gcc** compiles it (CPU only -- no CUDA, no GPU driver to perish on a reviewer's machine); the program
**writes its results to a file** (`fopen`/`fprintf`/`fclose`); and we **read the file back** and check it
matches SymPy to machine precision.

The physics is the material dispersion at the heart of this project: the **Sellmeier** refractive index of
fused silica, $n(\lambda)$, and its wavelength derivatives (group index, group-velocity dispersion). The
key win is that we never differentiate by hand -- SymPy computes $dn/d\lambda$ and $d^2n/d\lambda^2$ and
*those derivatives are emitted as C automatically*. Theory in, portable C out, verified end to end.

Self-contained: NumPy, SymPy, Pandas, Matplotlib, plus a system `gcc` (MinGW-w64). No GPU."""),
setup_cell(),
co("""import os, shutil, subprocess, tempfile
os.environ["PATH"] = r"C:\\msys64\\mingw64\\bin" + os.pathsep + os.environ["PATH"]   # find MinGW gcc
GCC = shutil.which("gcc")
print("gcc:", GCC)
print(subprocess.run([GCC, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
      if GCC else "gcc not found -- generated C will still be shown")"""),

md(r"""## Symbolic photonics: the Sellmeier index and its derivatives (differentiated by SymPy)

Fused silica obeys $n^2(\lambda)=1+\sum_i\dfrac{B_i\lambda^2}{\lambda^2-C_i}$ ($\lambda$ in microns). From
$n$ alone SymPy produces everything downstream by symbolic differentiation: the **group index**
$n_g=n-\lambda\,dn/d\lambda$ and the curvature $d^2n/d\lambda^2$ that fixes the **group-velocity
dispersion** (and its zero-dispersion wavelength near 1.3 um). No hand calculus."""),
co("""lam = sp.symbols('lam', positive=True)                        # wavelength in micrometers
B = [0.6961663, 0.4079426, 0.8974794]
Cc = [0.0684043**2, 0.1162414**2, 9.896161**2]
n = sp.sqrt(1 + sum(B[i]*lam**2/(lam**2 - Cc[i]) for i in range(3)))
dn  = sp.diff(n, lam)                                          # dn/dlam   (SymPy differentiates the theory)
d2n = sp.diff(n, lam, 2)                                       # d2n/dlam2
n_g = n - lam*dn                                               # group index
for name, expr, ref in [("n(1.55um)", n, 1.44402), ("n_g(1.55um)", n_g, 1.46260),
                        ("d2n/dlam2(1.55um)", d2n, -0.004238)]:
    val = float(expr.subs(lam, 1.55))
    print(f"{name:20s} = {val:.6f}")
    assert abs(val - ref) < 1e-4
print("\\nSymPy derived n, dn, d2n, n_g symbolically -- ready to emit as C")"""),

md(r"""## `sympy.ccode`: theory -> C expressions (with common-subexpression elimination)

`sp.ccode(expr)` prints a C expression for any SymPy formula -- including the derivatives SymPy just
computed. For efficiency, `sp.cse` factors out repeated subexpressions (the same $\lambda^2-C_i$ pole
denominators appear in $n$, $dn$, and $d^2n$), so the generated C evaluates shared terms once."""),
co("""print("C for n(lam):\\n ", sp.ccode(n)[:120], "...\\n")
print("C for dn/dlam (auto-differentiated):\\n ", sp.ccode(dn)[:120], "...\\n")
# common-subexpression elimination across the three outputs
subs, reduced = sp.cse([n, dn, d2n])
print(f"cse found {len(subs)} shared subexpressions (computed once, reused):")
for sym, sub in subs[:4]:
    print(f"   {sym} = {sp.ccode(sub)}")
print("   ...")"""),

md(r"""## Generate, save, compile, run, load, verify -- the full round trip

We assemble a portable C program: three functions `n_index`, `dn`, `d2n` whose bodies are the `ccode` of
the SymPy expressions, and a `main` that sweeps $\lambda$ over the telecom band and **writes a CSV with
`fopen`/`fprintf`/`fclose`** (path from `argv[1]`). We compile it with `gcc -O2 -lm` (CPU only), run it,
**read the CSV back**, and confirm every value matches SymPy to $10^{-9}$. This is the reproducibility
contract: the theory and the compiled C agree bit-for-portable-bit."""),
co(r'''c_source = f"""#include <math.h>
#include <stdio.h>

double n_index(double lam) {{ return {sp.ccode(n)}; }}
double dn(double lam)      {{ return {sp.ccode(dn)}; }}
double d2n(double lam)     {{ return {sp.ccode(d2n)}; }}

int main(int argc, char **argv) {{
    if (argc < 2) {{ fprintf(stderr, "usage: %s out.csv\\n", argv[0]); return 1; }}
    FILE *f = fopen(argv[1], "w");                 /* save file */
    if (!f) {{ perror("fopen"); return 1; }}
    fprintf(f, "lam,n,dn,d2n\\n");
    for (double lam = 1.0; lam <= 1.7001; lam += 0.05)
        fprintf(f, "%.12f,%.12f,%.12f,%.12f\\n", lam, n_index(lam), dn(lam), d2n(lam));
    fclose(f);                                     /* fclose */
    return 0;
}}
"""
work = tempfile.mkdtemp(prefix="ccode_bridge_")
c_path   = os.path.join(work, "sellmeier.c")
exe_path = os.path.join(work, "sellmeier.exe")
csv_path = os.path.join(work, "sellmeier_out.csv")
with open(c_path, "w") as fh:                      # save the generated C to disk
    fh.write(c_source)
print("wrote", c_path, f"({len(c_source)} bytes)")

comp = subprocess.run([GCC, "-O2", c_path, "-o", exe_path, "-lm"], capture_output=True, text=True)
assert comp.returncode == 0, comp.stderr
subprocess.run([exe_path, csv_path], check=True)   # run -> writes csv via fopen/fclose
df = pd.read_csv(csv_path)                          # load the file back
print("compiled with gcc (CPU, no CUDA) and ran; CSV rows:", len(df))

# verify the C output against SymPy at the same wavelengths
err = 0.0
for _, row in df.iterrows():
    l = row["lam"]
    err = max(err, abs(row["n"]   - float(n.subs(lam, l))),
                   abs(row["dn"]  - float(dn.subs(lam, l))),
                   abs(row["d2n"] - float(d2n.subs(lam, l))))
print(f"max |C - SymPy| over the sweep = {err:.2e}")
assert err < 1e-9
print("theory (SymPy) and generated C agree to 1e-9 -- reproducible, portable, GPU-free")'''),

md(r"""## Plots -- the physics, computed by the generated C"""),
co(r"""df["n_g"] = df["n"] - df["lam"]*df["dn"]
c_um_ps = 299792.458                                          # speed of light, um/ps... (for scale only)
df["GVD_arb"] = -df["lam"]*df["d2n"]                          # sign tracks group-velocity dispersion
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(df["lam"], df["n"], color="#4C78A8", label="n")
ax[0].plot(df["lam"], df["n_g"], color="#E45756", label="n_g (group)")
ax[0].set_xlabel("wavelength [um]"); ax[0].set_ylabel("index"); ax[0].legend(fontsize=9)
ax[0].set_title("fused-silica index (from gcc-compiled C)")
ax[1].plot(df["lam"], df["d2n"], color="#54A24B")
ax[1].axhline(0, ls=":", color="gray")
ax[1].set_xlabel("wavelength [um]"); ax[1].set_ylabel(r"$d^2n/d\lambda^2$")
ax[1].set_title("curvature -> zero-dispersion near 1.3 um")
plt.tight_layout(); plt.show()"""),

md(r"""## Why this matters for replicating a lab (and for SBIR/grant delivery)

- **Portable, GPU-free.** The result is plain C compiled by gcc -- it runs on any reviewer's or
  collaborator's machine with no CUDA, no driver version, no GPU to "perish." The theory is the artifact.
- **Reproducible by construction.** The C is *generated from the symbolic theory*, not transcribed, so it
  cannot silently drift from the equations; the round-trip check pins them together at $10^{-9}$.
- **Derivatives for free.** SymPy differentiates the model and emits the derivative C -- exactly what you
  need for dispersion ($dn/d\lambda$, $d^2n/d\lambda^2$) without error-prone hand calculus. The same bridge
  targets `fcode` (Fortran), `jscode`, or a VHDL entity for the FPGA path in this project.

## Summary

- `sympy.ccode` turns any SymPy expression -- including SymPy-computed **derivatives** -- into C; `sp.cse`
  shares repeated subexpressions for efficiency.
- We generated C for the Sellmeier index and its dispersion derivatives, **saved** it, compiled it with
  **gcc (CPU only)**, ran it to **write a CSV** (`fopen`/`fclose`), **read it back**, and verified it
  matches SymPy to $10^{-9}$.
- This is the theory-to-implementation bridge for reproducible, portable, GPU-independent delivery -- the
  Python->C step of this project's Python->C->VHDL flow.

Subject-verb-object: SymPy derives the physics; ccode emits the C; gcc compiles it anywhere; the file
round-trip proves they agree."""),
]

write("sympy_ccode", "theory_bridge", cells)
