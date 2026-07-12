"""Generate notebooks/00_scientific_python.ipynb."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nbkit import md, co, frontmatter, setup_cell, section, write

cells = [
frontmatter("00", "Scientific Python and reproducible numerics", {
    "physics": "A measurement or simulation must give the same number twice.",
    "mathematics": "Floating-point arithmetic, rounding error, and error growth.",
    "algorithm": "Vectorized evaluation and seeded pseudo-random number generation.",
    "software": "NumPy, SymPy, Pandas, Matplotlib; optional PyTorch.",
    "experiment": "Any experiment whose analysis pipeline must be reproducible.",
    "engineering": "Deterministic firmware and DSP produce bit-identical results.",
}),
setup_cell(),

section("English explanation"),
md("""Scientific computing represents real numbers with a finite number of bits, so most values are
stored approximately. A reproducible pipeline controls three things: the numeric type (how much
precision), the order of operations (rounding is not associative), and every source of randomness
(seed the generator). This chapter fixes those habits; every later chapter depends on them.

The subject-verb-object frame: the *computer stores* an approximation, the *operation accumulates*
error, and the *seed fixes* the random stream."""),

section("Mathematical derivation"),
md("""A binary floating-point number has the form $x=\\pm(1+f)\\,2^{e}$ with a fixed-width fraction
$f$. The gap between consecutive representable numbers near 1 is the machine epsilon
$\\varepsilon$; for IEEE-754 double precision $\\varepsilon=2^{-52}\\approx2.22\\times10^{-16}$. A single
rounding introduces relative error at most $\\varepsilon/2$. Subtracting two nearly equal numbers
$a-b$ **cancels** their leading digits and amplifies that relative error by $\\frac{|a|+|b|}{|a-b|}$ --
catastrophic cancellation, the main enemy of naive formulas."""),

section("Dimensions and SI units"),
md("""Floating-point values are pure numbers; physical meaning comes from the unit attached to them.
The repository keeps that meaning explicit with `physkit.units`. Below, Planck's constant times a
frequency is checked to carry the dimension of energy before any number is trusted."""),
co("""h = U.Quantity(C.H, U.ACTION)          # J s
f = U.Quantity(4.84e14, U.FREQUENCY)   # Hz (red light)
E = h * f
E.to_dimension(U.ENERGY)               # raises if the dimensions were wrong
print("photon energy dimension:", E.dim, "| value:", E.value, "J =", E.value / C.E, "eV")"""),

section("SymPy derivation"),
md("""SymPy computes with exact symbols, so it is the reference against which floating point is
judged. Here the associativity failure of floating-point addition is exhibited: exactly, the sum is
order-independent; in `float`, it is not."""),
co("""a, b, c = sp.Rational(1), sp.Rational(1, 10**16), sp.Rational(-1)
exact = (a + b) + c
print("SymPy exact ((1 + 1e-16) - 1) =", exact, "=", float(exact))
fa, fb, fc = 1.0, 1e-16, -1.0
print("float  ((1 + 1e-16) - 1) =", (fa + fb) + fc, " vs (1e-16 + (1 - 1)) =", fb + (fa + fc))
print("=> floating-point addition is not associative")"""),

section("NumPy implementation"),
md("""NumPy evaluates array expressions in compiled loops (vectorization): the same result as a
Python loop, produced far faster and with a fixed evaluation order. We also show a numerically stable
rewrite that avoids cancellation."""),
co("""eps = np.finfo(np.float64).eps
print("machine epsilon (float64):", eps)

# catastrophic cancellation: sqrt(1+x) - 1 for small x
x = 1e-12
naive = np.sqrt(1 + x) - 1.0
stable = x / (np.sqrt(1 + x) + 1.0)         # algebraically identical, no cancellation
print(f"naive  = {naive:.3e}")
print(f"stable = {stable:.3e}  (reference x/2 = {x/2:.3e})")

# vectorization vs a Python loop (same numbers)
v = np.linspace(0, 1, 1_000_00)
loop = sum(vi**2 for vi in v.tolist())
vect = float(np.sum(v**2))
print("loop and vectorized agree:", np.isclose(loop, vect))"""),

section("Pandas tables"),
md("""Pandas organizes numeric facts into tables for reporting. Here: the precision of each common
NumPy float type."""),
co("""rows = []
for dt in (np.float16, np.float32, np.float64):
    fi = np.finfo(dt)
    rows.append({"dtype": np.dtype(dt).name, "bits": fi.bits, "eps": fi.eps,
                 "max": fi.max, "decimal_digits": fi.precision})
df = pd.DataFrame(rows)
print(df.to_string(index=False))"""),

section("Matplotlib plots"),
md("""The stable and naive formulas diverge as $x\\to0$. Plotting relative error against $x$ shows the
naive form losing all significant digits once cancellation dominates."""),
co("""xs = np.logspace(-16, -1, 200)
naive = np.sqrt(1 + xs) - 1.0
stable = xs / (np.sqrt(1 + xs) + 1.0)
ref = xs / 2
plt.figure()
plt.loglog(xs, np.abs(naive - ref) / ref, label="naive  sqrt(1+x)-1")
plt.loglog(xs, np.abs(stable - ref) / ref + 1e-18, label="stable  x/(sqrt(1+x)+1)")
plt.xlabel("x"); plt.ylabel("relative error vs x/2")
plt.title("catastrophic cancellation ruins the naive formula")
plt.legend(); plt.tight_layout(); plt.show()"""),

section("PyTorch (optional)"),
md("""When PyTorch is installed it provides the same arithmetic on tensors (and on a GPU). The NumPy
result stays authoritative; PyTorch is a cross-check. The cell is guarded so the notebook runs
without it."""),
co("""if torch is not None:
    xt = torch.linspace(0, 1, 1000, dtype=torch.float64)
    print("torch sum(x^2) =", float((xt**2).sum()),
          "| numpy =", float(np.sum(np.linspace(0,1,1000)**2)))
else:
    print("PyTorch absent -- NumPy path already computed the result above.")"""),

section("Exercises"),
md("""1. Show numerically that `0.1 + 0.2 == 0.3` is `False` in float64, and explain why using the
   binary expansion of `0.1`.
2. The quadratic formula loses precision for one root when $b^2\\gg4ac$. Implement the stable variant
   using $x_1 x_2=c/a$ and compare.
3. Seed `np.random.default_rng` with a fixed value and confirm two runs produce identical arrays;
   remove the seed and confirm they differ. State why reproducibility needs the seed."""),

section("Engineering applications"),
md("""A **firmware** routine on a microcontroller must accumulate ADC samples in an order that avoids
cancellation, or a slowly drifting baseline will corrupt the sum. A **DSP** filter implemented in
fixed point chooses word length from the same $\\varepsilon$ analysis. A **GPU** kernel that reduces
millions of values uses pairwise summation to bound error growth. Reproducibility -- fixed dtype,
fixed order, fixed seed -- is what lets an instrument's result be audited.

Summary (subject-verb-object): the computer stores approximations; the algorithm controls error; the
seed fixes randomness; the engineer guarantees reproducibility."""),
]

write("00", "scientific_python", cells)
