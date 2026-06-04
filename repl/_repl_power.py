import numpy as np, torch
from scipy.signal import welch

np.random.seed(42)
N, dt, fs = 1024, 1e-9, 1e9
t = np.arange(N)*dt
sig = 3.0*np.sin(2*np.pi*100e6*t) + 1.5*np.sin(2*np.pi*350e6*t) + 0.5*np.random.randn(N)
freqs, Pxx = welch(sig, fs=fs, nperseg=256)
K = N//256
sigma_Pxx = Pxx * np.sqrt(2.0/K)

pk1 = np.argmin(np.abs(freqs - 100e6))
pk2 = np.argmin(np.abs(freqs - 350e6))
noise_med = np.median(Pxx)

print("=== Power Spectrum + Propagation of Errors ===")
print("  Tone 100 MHz:  P={:.4f} +/- {:.4f} W/Hz".format(Pxx[pk1], sigma_Pxx[pk1]))
print("  Tone 350 MHz:  P={:.4f} +/- {:.4f} W/Hz".format(Pxx[pk2], sigma_Pxx[pk2]))
print("  Noise floor:   P={:.6f}".format(noise_med))
print("  SNR:           {:.1f} dB".format(10*np.log10(Pxx[pk1]/noise_med)))
print("  Fractional err on each bin: sqrt(2/K) = {:.1f}%  (K={} averages)".format(np.sqrt(2/K)*100, K))
print()

print("=== General error propagation rules ===")
rules = [
    ("z = a + b",       "sigma_z = sqrt(sigma_a^2 + sigma_b^2)"),
    ("z = a * b",       "sigma_z/z = sqrt((sa/a)^2 + (sb/b)^2)"),
    ("z = a^n",         "sigma_z/z = n * sigma_a/a"),
    ("z = log(a)",      "sigma_z = sigma_a / a"),
    ("z = exp(a)",      "sigma_z = z * sigma_a"),
    ("z = PSD bin",     "sigma_z = z * sqrt(2/K)  chi-sq, K averages"),
]
for formula, rule in rules:
    print("  {:22s}  ->  {}".format(formula, rule))
print()

print("=== Ratio test: torch autograd on common physics functions ===")
x_vals = torch.linspace(0.01, 5.0, 1000)

funcs = [
    ("x^2",        lambda x: x**2),
    ("exp(-x)",    lambda x: torch.exp(-x)),
    ("sin(x)/x",   lambda x: torch.sin(x)/x),
    ("1/x",        lambda x: 1.0/x),
    ("x*log(x)",   lambda x: x*torch.log(x)),
]

print("  {:16s}  {:>16s}  {:>14s}".format("Function", "integral[0.01,5]", "d/dx at x=2.5"))
for name, fn in funcs:
    try:
        y = fn(x_vals)
        integral = float(torch.trapezoid(y, x_vals))
        x_pt = torch.tensor(2.5, requires_grad=True)
        y_pt = fn(x_pt)
        y_pt.backward()
        grad = float(x_pt.grad)
        print("  {:16s}  {:>16.4f}  {:>14.4f}  autograd OK".format(name, integral, grad))
    except Exception as e:
        print("  {:16s}  ERROR: {}".format(name, e))

print()
print("=== Ratio test for series convergence ===")
print("  lim |a_{n+1}/a_n| < 1  ->  converges  (torch can integrate it)")
print("  lim |a_{n+1}/a_n| > 1  ->  diverges")
series = [
    ("1/n!  (exp)",     lambda n: 1.0/__import__('math').factorial(min(n,20))),
    ("1/n^2",           lambda n: 1.0/n**2),
    ("1/n",             lambda n: 1.0/n),
    ("r^n r=0.9",       lambda n: 0.9**n),
    ("r^n r=1.1",       lambda n: 1.1**n),
]
for name, a in series:
    ns = np.arange(2, 12)
    ratios = [a(n+1)/a(n) for n in ns]
    lim = np.mean(ratios[-3:])
    verdict = "CONVERGES" if lim < 1.0 else "DIVERGES"
    print("  {:20s}  ratio -> {:.3f}  {}".format(name, lim, verdict))
