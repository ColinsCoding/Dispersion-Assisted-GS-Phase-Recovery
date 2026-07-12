"""Generate notebooks/05_calculus_review.ipynb."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nbkit import md, co, frontmatter, setup_cell, section, write

cells = [
frontmatter("05", "Calculus review: derivative, integral, Taylor, gradient", {
    "physics": "Velocity is the rate of change of position; work is the integral of force.",
    "mathematics": "Limits, the derivative, the definite integral, the fundamental theorem, Taylor series, the gradient.",
    "algorithm": "Finite-difference differentiation and trapezoid/Simpson integration; automatic differentiation.",
    "software": "SymPy for exact calculus; NumPy/SciPy for numerics; optional PyTorch autograd.",
    "experiment": "A position sensor is differentiated to velocity; a power meter is integrated to energy.",
    "engineering": "Firmware differences an ADC stream; a running sum integrates it; autograd is backpropagation.",
}),
setup_cell(),

section("English explanation"),
md("""Calculus turns a signal into its rate of change (the **derivative**) and accumulates a rate into
a total (the **integral**); the **fundamental theorem** says these undo each other. The **Taylor
series** approximates a smooth function by a polynomial near a point -- the basis of every numerical
method that follows. In three dimensions the **gradient** points uphill on a scalar field. This
chapter fixes the exact results (SymPy) and their numerical counterparts (NumPy), building on the
floating-point care of chapter 00.

Subject-verb-object: the derivative measures the rate; the integral accumulates the total; the
gradient points uphill."""),

section("Mathematical derivation"),
md("""The derivative is the limit $f'(x)=\\lim_{h\\to0}\\frac{f(x+h)-f(x)}{h}$; the central difference
$\\frac{f(x+h)-f(x-h)}{2h}$ approximates it with error $O(h^2)$. The definite integral is the limit of
Riemann sums, and the **fundamental theorem** gives $\\int_a^b f'(x)\\,dx=f(b)-f(a)$. Taylor's theorem
writes $f(x)=\\sum_{n\\ge0}\\frac{f^{(n)}(a)}{n!}(x-a)^n$; truncating gives a polynomial approximant. The
gradient of $f(x,y)$ is $\\nabla f=(\\partial_x f,\\partial_y f)$."""),

section("Dimensions and SI units"),
md("""Differentiating with respect to time divides the dimension by time, integrating over time
multiplies by it. So $\\frac{d}{dt}[\\text{position}]$ is a velocity and $\\frac{d}{dt}[\\text{velocity}]$
an acceleration, while $\\int[\\text{power}]\\,dt$ is an energy. `physkit.units` confirms the bookkeeping."""),
co("""x = U.LENGTH; t = U.TIME
velocity = x / t; acceleration = velocity / t
print("d(position)/dt ->", velocity, "(VELOCITY:", velocity == U.VELOCITY, ")")
print("d(velocity)/dt ->", acceleration, "(ACCELERATION:", acceleration == U.ACCELERATION, ")")
energy = U.POWER * U.TIME                         # integral of power over time
print("int(power) dt ->", energy, "(ENERGY:", energy == U.ENERGY, ")")
assert velocity == U.VELOCITY and energy == U.ENERGY"""),

section("SymPy derivation"),
md("""SymPy differentiates and integrates exactly, and verifies the fundamental theorem and a Taylor
expansion."""),
co("""xs = sp.symbols('x', real=True)
f = sp.sin(xs) * sp.exp(xs)
fp = sp.diff(f, xs)
print("d/dx [sin(x) e^x] =", sp.simplify(fp))
# fundamental theorem: integral of f' from 0 to 1 equals f(1) - f(0)
ftc = sp.integrate(fp, (xs, 0, 1)) - (f.subs(xs, 1) - f.subs(xs, 0))
assert sp.simplify(ftc) == 0
print("fundamental theorem verified: int_0^1 f' dx = f(1) - f(0)")
# Taylor series of sin about 0
print("sin(x) =", sp.series(sp.sin(xs), xs, 0, 8))"""),

section("NumPy implementation"),
md("""The central-difference derivative and the trapezoid integral, checked against the exact SymPy
results. As in chapter 00, too small an $h$ reintroduces rounding error."""),
co("""fnum = lambda x: np.sin(x)*np.exp(x)
fp_exact = lambda x: np.exp(x)*(np.sin(x)+np.cos(x))
x0, h = 1.3, 1e-5
central = (fnum(x0+h) - fnum(x0-h)) / (2*h)
print(f"central diff f'(1.3) = {central:.8f} | exact = {fp_exact(x0):.8f}")

# trapezoid integral of f' from 0 to 1 should recover f(1)-f(0)
xg = np.linspace(0, 1, 2001)
I = np.trapezoid(fp_exact(xg), xg)
print(f"trapezoid int f' [0,1] = {I:.6f} | f(1)-f(0) = {fnum(1)-fnum(0):.6f}")"""),

section("Pandas tables"),
md("""Integration error falls with the number of panels: trapezoid as $O(N^{-2})$, Simpson as
$O(N^{-4})$."""),
co("""def simpson(y, x):
    n = len(x) - 1                       # number of panels (even)
    h = (x[-1] - x[0]) / n
    return h/3 * (y[0] + y[-1] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2]))

exact = fnum(1) - fnum(0)
rows = []
for N in (10, 20, 40, 80, 160):
    xg = np.linspace(0, 1, N+1)
    y = fp_exact(xg)
    rows.append({"N": N, "trapezoid_err": abs(np.trapezoid(y, xg) - exact),
                 "simpson_err": abs(simpson(y, xg) - exact)})
df = pd.DataFrame(rows)
print(df.to_string(index=False))"""),

section("Matplotlib plots"),
co("""import math
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
xg = np.linspace(-4, 4, 400)
ax[0].plot(xg, np.sin(xg), "k", lw=2, label="sin x")
for terms in (2, 3, 4):                          # partial sums of the sine Taylor series
    approx = sum(((-1)**k)/math.factorial(2*k+1) * xg**(2*k+1) for k in range(terms))
    ax[0].plot(xg, approx, "--", label=f"Taylor, {terms} terms")
ax[0].set_ylim(-2, 2); ax[0].set_title("Taylor series converges to sin x"); ax[0].legend(fontsize=8)
ax[0].set_xlabel("x")
ax[1].loglog(df["N"], df["trapezoid_err"], "o-", label="trapezoid ~ N^-2")
ax[1].loglog(df["N"], df["simpson_err"], "s-", label="Simpson ~ N^-4")
ax[1].set_xlabel("panels N"); ax[1].set_ylabel("integration error"); ax[1].legend()
ax[1].set_title("higher-order rules converge faster")
plt.tight_layout(); plt.show()"""),

section("PyTorch (optional)"),
md("""Automatic differentiation computes exact derivatives by the chain rule -- the same mechanism as
backpropagation. When present, PyTorch autograd reproduces the SymPy derivative."""),
co("""if torch is not None:
    xv = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    y = torch.sin(xv) * torch.exp(xv)
    y.backward()
    print(f"autograd f'(1.3) = {float(xv.grad):.8f} | exact = {fp_exact(1.3):.8f}")
else:
    print("PyTorch absent -- the central-difference and SymPy derivatives above are authoritative.")"""),

section("Exercises"),
md("""1. Plot the central-difference error of $f'(1.3)$ versus $h$ from $10^{-1}$ to $10^{-12}$ and
   find the optimal $h$ where truncation error meets rounding error (chapter 00).
2. Use SymPy to obtain the Taylor series of $e^x$, $\\cos x$, and $\\ln(1+x)$; identify the radius of
   convergence of the last.
3. Compute the gradient of $f(x,y)=x^2+y^2$ symbolically and numerically, and confirm it points
   radially outward."""),

section("Engineering applications"),
md("""A motion controller **differences** successive **ADC** position samples to estimate velocity --
the discrete central difference -- and low-pass filters the result because differentiation amplifies
high-frequency noise. A power meter **accumulates** samples with a running sum, the discrete integral,
to report energy. In machine learning and adaptive **DSP**, **autograd** propagates derivatives
through a computation graph; it is the calculus of this chapter applied at scale on a **GPU**.

Summary (subject-verb-object): the sensor supplies samples; the difference estimates the rate; the
running sum accumulates the total; autograd differentiates the graph."""),
]

write("05", "calculus_review", cells)
