import json

cells = []

def md(id_, text):
    return {"cell_type": "markdown", "id": id_, "metadata": {}, "source": text.splitlines(keepends=True)}

def code(id_, text):
    return {"cell_type": "code", "execution_count": None, "id": id_, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}

# ======================================================================
cells.append(md("title", r"""
# Chain Rule and Differentiable PyTorch — From First Principles to This Session's Own Code

**A Feynman-style deep dive**: explain the physical idea in plain language first, then derive
it, then encode it, then test it, then ask what could be wrong. The chain rule is the single
idea underneath every `.backward()` call used elsewhere in this repository — this notebook
opens that up explicitly, using worked examples pulled from the SEALS/dispersion work already
built and validated in this session.
"""))

cells.append(code("imports", r"""
import sys, pathlib
sys.path.insert(0, '.')
sys.path.insert(0, str(pathlib.Path('.').resolve().parents[1]))

import numpy as np
import sympy as sp
import torch
from scipy.special import spherical_jn
import matplotlib.pyplot as plt
from IPython.display import display

from inverse import dispersion

torch.set_default_dtype(torch.float64)
sp.init_printing(use_latex='mathjax')
plt.rcParams.update({'figure.dpi': 110, 'font.size': 11})

def show(label, expr):
    print(label)
    display(expr)

print('ready')
"""))

# ======================================================================
cells.append(md("p1-intro", r"""
---
# Part 1 — What the Chain Rule Actually Says

**The physical picture, no symbols yet.** Turn a small crank on gear $A$. Gear $A$ turns gear
$B$. Gear $B$ turns gear $C$. If turning $A$ by one small notch turns $B$ by 3 notches, and
turning $B$ by one notch turns $C$ by 2 notches, then turning $A$ by one notch turns $C$ by
$3\times2=6$ notches — **the rates multiply along the chain.** That is the entire chain rule.
Nothing about calculus changes this idea; calculus just lets the "notches" shrink to
infinitesimally small steps, and the gear ratios become derivatives.

**The chain in this repository that matters most**, which Part 4 opens up in full: a particle's
*diameter* affects the Mie *size parameter*, which affects a *Bessel function's* value, which
affects the *Mie scattering coefficients*, which affect the *scattered field*, which affects the
*measured intensity*. Five gears. `PyTorch.backward()` turns the crank at the diameter end and
tells you how fast the intensity gear spins — by multiplying five local ratios together,
automatically, without you ever writing that product out by hand (though Part 3 makes it write
itself out, to prove that's really all that's happening).
"""))

cells.append(md("p1-questions", r"""
**Feynman check:** if gear $B$ momentarily stops turning $C$ at all (local ratio $=0$) even
though $A$ is still turning $B$ vigorously, what happens to the overall $A\to C$ rate? What does
that correspond to in a neural network or physics-model gradient?
"""))

# ======================================================================
cells.append(md("p2-intro", r"""
---
# Part 2 — A Concrete Chain, by Hand (SymPy)

A four-link chain, deliberately mixing operation types so each link is a genuinely different
kind of "gear ratio":
$$
u_1 = x^2 \qquad u_2=\sin(u_1) \qquad u_3=e^{u_2} \qquad y=\sqrt{u_3}
$$
"""))

cells.append(code("p2-sympy-chain", r"""
x = sp.Symbol('x', positive=True)
u1 = x**2
u2 = sp.sin(u1)
u3 = sp.exp(u2)
y = sp.sqrt(u3)

# each LOCAL derivative -- one gear ratio each
du1_dx = sp.diff(u1, x)
du2_du1 = sp.diff(sp.sin(sp.Symbol('u1')), sp.Symbol('u1')).subs(sp.Symbol('u1'), u1)
du3_du2 = sp.diff(sp.exp(sp.Symbol('u2')), sp.Symbol('u2')).subs(sp.Symbol('u2'), u2)
dy_du3 = sp.diff(sp.sqrt(sp.Symbol('u3')), sp.Symbol('u3')).subs(sp.Symbol('u3'), u3)

chain_product = sp.simplify(dy_du3 * du3_du2 * du2_du1 * du1_dx)
direct = sp.simplify(sp.diff(y, x))

show('dy/dx via the chain-rule PRODUCT of four local derivatives:', chain_product)
show('dy/dx via differentiating the composed expression directly:', direct)
print('identical:', sp.simplify(chain_product - direct) == 0)
"""))

cells.append(code("p2-numeric", r"""
x0 = 0.8
vals = {x: x0}
u1_val = float(u1.subs(vals))
u2_val = float(u2.subs(vals))
u3_val = float(u3.subs(vals))
y_val = float(y.subs(vals))

local_grads = [float(du1_dx.subs(vals)), float(du2_du1.subs(vals)),
               float(du3_du2.subs(vals)), float(dy_du3.subs(vals))]
names = ['du1/dx', 'du2/du1', 'du3/du2', 'dy/du3']

print(f'x0={x0}  ->  u1={u1_val:.5f}  u2={u2_val:.5f}  u3={u3_val:.5f}  y={y_val:.5f}')
print()
product = 1.0
for name, g in zip(names, local_grads):
    product *= g
    print(f'{name:10} = {g:.6f}   (running product: {product:.6f})')

exact = float(direct.subs(vals))
print(f'\nfinal chain-rule product = {product:.8f}')
print(f'exact dy/dx (SymPy)      = {exact:.8f}')
assert abs(product - exact) < 1e-10
"""))

# ======================================================================
cells.append(md("p3-intro", r"""
---
# Part 3 — The Same Chain as a PyTorch Autograd Graph

Build the identical four gears as PyTorch tensors, call `.backward()` once, then check that
`x.grad` really is the same four-number product computed by hand in Part 2 — and, going one
level deeper, that the *intermediate* gradients PyTorch records at each gear
(`.retain_grad()`) are the *partial* products, not the individual local ratios themselves.
"""))

cells.append(code("p3-torch-chain", r"""
x_t = torch.tensor(0.8, requires_grad=True)
u1_t = x_t**2;          u1_t.retain_grad()
u2_t = torch.sin(u1_t); u2_t.retain_grad()
u3_t = torch.exp(u2_t); u3_t.retain_grad()
y_t = torch.sqrt(u3_t)

y_t.backward()

print(f'x.grad  (dy/dx)   = {x_t.grad.item():.8f}   (Part 2 hand product: {product:.8f})')
assert abs(x_t.grad.item() - product) < 1e-10
print('MATCH: autograd\'s answer is exactly the hand-multiplied chain-rule product.')
"""))

cells.append(code("p3-intermediate-grads", r"""
# what autograd records AT EACH GEAR is dy/d(that gear) -- a PARTIAL product from y backward
# to that point, not the local ratio alone. Verify each against the Part 2 numbers directly.
partial_from_y = {
    'u3.grad (= dy/du3)':               (u3_t.grad.item(), local_grads[3]),
    'u2.grad (= dy/du3 * du3/du2)':      (u2_t.grad.item(), local_grads[3]*local_grads[2]),
    'u1.grad (= dy/du3*du3/du2*du2/du1)':(u1_t.grad.item(), local_grads[3]*local_grads[2]*local_grads[1]),
}
for name, (autograd_val, hand_val) in partial_from_y.items():
    print(f'{name:38} autograd={autograd_val:.6f}   hand={hand_val:.6f}   match={abs(autograd_val-hand_val)<1e-10}')
    assert abs(autograd_val - hand_val) < 1e-10
"""))

cells.append(md("p3-note", r"""
This is literally what `backward()` does for any graph, not just this toy one: walk the graph
from output to input, and at every node multiply the incoming partial product by that node's
own local derivative. A custom `autograd.Function` (Part 4a) is exactly a place where *you*
supply that one node's local derivative by hand instead of letting PyTorch differentiate the
forward formula automatically.
"""))

# ======================================================================
cells.append(md("p4-intro", r"""
---
# Part 4 — Worked Examples From This Session's Own Code

Two different ways a chain-rule link shows up in real code already built this session.
"""))

cells.append(md("p4a-intro", r"""
## 4a. A custom-supplied link: differentiating through a Bessel function

PyTorch has no spherical Bessel functions. The SEALS Mie-scattering work (built earlier this
session, in the standalone differentiable-Mie notebook) needed $j_n(x)$ and its gradient with
respect to a physical parameter buried inside $x$ — solved with a custom `autograd.Function`
that calls `scipy.special.spherical_jn` for the *value* and `scipy.special.spherical_jn(...,
derivative=True)` for that one gear's *local ratio*, and lets PyTorch chain everything else
automatically. Reproduced in miniature here: $d\to x=\pi d/(\lambda/n)\to j_2(x)$.
"""))

cells.append(code("p4a-bessel", r"""
class SphBesselJ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n):
        x_val = float(x.detach())
        val = spherical_jn(n, x_val)
        deriv = spherical_jn(n, x_val, derivative=True)   # the ONE local ratio SciPy supplies
        ctx.save_for_backward(torch.tensor(deriv, dtype=x.dtype))
        return torch.tensor(val, dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (deriv,) = ctx.saved_tensors
        return grad_output * deriv, None   # chain rule: incoming product * this link's ratio

lam, n_med, n_order = 1.59e-6, 1.0, 2
d_t = torch.tensor(9.94e-6, dtype=torch.float64, requires_grad=True)
x_t2 = np.pi * d_t / (lam / n_med)          # native torch op -- PyTorch differentiates this link itself
j2 = SphBesselJ.apply(x_t2, n_order)         # custom link -- WE differentiate this one

j2.backward()
dj2_dd_autograd = d_t.grad.item()

h = d_t.item() * 1e-6
dj2_dd_fd = (spherical_jn(n_order, np.pi*(d_t.item()+h)/(lam/n_med))
             - spherical_jn(n_order, np.pi*(d_t.item()-h)/(lam/n_med))) / (2*h)

print(f'dj2/dd, autograd (custom link + native chain): {dj2_dd_autograd:.6e}')
print(f'dj2/dd, finite difference (independent check):  {dj2_dd_fd:.6e}')
rel_err = abs(dj2_dd_autograd - dj2_dd_fd) / abs(dj2_dd_fd)
print(f'relative error: {rel_err:.3e}')
assert rel_err < 1e-6
"""))

cells.append(md("p4b-intro", r"""
## 4b. A fully-native chain: the dispersion operator's loss gradient

`inverse.dispersion.dispersive_operator` (built and validated earlier this session, matching
`dgs.gs_core.disperse`'s convention) uses only `torch.fft.fft/ifft`, complex multiplication,
and `.abs()**2` — every one of those is a native, already-differentiable PyTorch op, so no
custom `autograd.Function` is needed here at all. The chain is: phase estimate $\to$ complex
field $\to$ dispersed field $\to$ intensity $\to$ loss against a target measurement.
"""))

cells.append(code("p4b-dispersion", r"""
torch.manual_seed(0)
N = 32
amp = torch.rand(N, dtype=torch.float64) + 0.5
phase_target = torch.randn(N, dtype=torch.float64)
E_target = amp * torch.exp(1j * phase_target)
I_target = dispersion.dispersive_operator(E_target, D=3.7).abs()**2

phase_est = torch.zeros(N, dtype=torch.float64, requires_grad=True)
E_est = amp * torch.exp(1j * phase_est)
E_dispersed = dispersion.dispersive_operator(E_est, D=3.7)
I_est = E_dispersed.abs()**2
loss = torch.mean((I_est - I_target)**2)
loss.backward()

grad_autograd = phase_est.grad.clone()

# independent finite-difference check on a few components (this chain has no custom links,
# so this validates PyTorch's OWN native chain rule through fft/ifft/complex ops, not our code)
def loss_at(phase_vec):
    E = amp * torch.exp(1j * phase_vec)
    I = dispersion.dispersive_operator(E, D=3.7).abs()**2
    return torch.mean((I - I_target)**2).item()

h = 1e-6
idx_check = [0, 5, 17, 31]
print(f'{"index":>6} {"autograd":>14} {"finite-diff":>14} {"rel err":>10}')
for i in idx_check:
    p_plus = phase_est.detach().clone(); p_plus[i] += h
    p_minus = phase_est.detach().clone(); p_minus[i] -= h
    fd = (loss_at(p_plus) - loss_at(p_minus)) / (2*h)
    rel = abs(grad_autograd[i].item() - fd) / (abs(fd) + 1e-30)
    print(f'{i:>6} {grad_autograd[i].item():>14.6e} {fd:>14.6e} {rel:>10.3e}')
    assert rel < 1e-4
"""))

# ======================================================================
cells.append(md("p5-intro", r"""
---
# Part 5 — Validation Summary
"""))

cells.append(code("p5-summary", r"""
print('Part 2 vs Part 3 (toy chain):        exact match, hand product == autograd, to 1e-10')
print('Part 4a (custom Bessel link):        autograd vs finite-difference, relative error < 1e-6')
print('Part 4b (fully-native FFT chain):    autograd vs finite-difference, relative error < 1e-4')
print()
print('All three used a DIFFERENT method to get the "ground truth" to check against')
print('(exact symbolic derivative, independent scipy-based finite difference, and a direct')
print('finite-difference perturbation of the loss itself) -- deliberately, so no single')
print('assumption is shared between what is being tested and what is doing the checking.')
"""))

# ======================================================================
cells.append(md("p6-questions", r"""
---
# Part 6 — Feynman Questions

1. In Part 1's gear analogy, what physically corresponds to a *vanishing gradient* — a real
   phenomenon in deep networks with many chained layers?
2. Part 3 showed `u1.grad` is a **partial product** counted from $y$ backward to $u_1$, not the
   local ratio $du_1/dx$ alone. Why does PyTorch record it this way, rather than storing each
   local ratio separately and multiplying at the end?
3. Part 4a's custom `backward()` returns `grad_output * deriv` — why is `grad_output` there at
   all, rather than just returning `deriv`? What would break if you returned `deriv` alone?
4. Part 4b needed no custom `autograd.Function`. What property of `torch.fft.fft`/`ifft` and
   complex-tensor arithmetic makes that possible, that plain `scipy.special.spherical_jn`
   (Part 4a) does not have?
5. Both Part 4 examples were checked against finite differences, not against each other. Why is
   that the more convincing test?
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python (.venv Spring2026)", "language": "python", "name": "spring2026"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

path = "chain_rule_and_autograd.ipynb"
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
json.load(open(path, encoding="utf-8"))
print(f"{path}: valid JSON, {len(cells)} cells")
