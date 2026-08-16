"""photonics_regime_decisions_torch.py -- two real true/false decision
points already implicit in this repo's photonics physics, each formalized
BOTH symbolically (SymPy Piecewise) and differentiably (torch), following
the same two-paradigm pattern as dgs/conditional_formalism_sympy_torch.py
(which formalized the sinc(x) x=0 special case). These two are chosen
because they are physically DIFFERENT kinds of decision:

  * FRESNEL / TOTAL INTERNAL REFLECTION (a HARD branch): past the critical
    angle theta_c, the Fresnel transmission formula's sqrt(1 - sin^2) term
    goes imaginary -- there is genuinely no real transmitted wave, so
    torch.where must SELECT between two different formulas, the same
    "evaluate both branches, pick one" pattern as the sinc(x) example.
    Checked here to be CONTINUOUS at theta_c (R -> 1 smoothly from below).

  * FAR-FIELD DISPERSIVE-FOURIER VALIDITY (a diagnostic threshold, not a
    formula switch): dgs/dispersive_fourier.py's gvd_propagate already
    computes `far_field_ok = L_m > 10 * L_D` as a plain Python bool -- a
    genuinely DISCONTINUOUS step with zero gradient almost everywhere, so
    it carries no design signal. far_field_soft_gate replaces it with a
    smooth sigmoid so a gradient-based fiber-length optimizer (like
    dgs/dispersive_fourier_torch.py's design_fiber_length_for_stretch_factor)
    could ALSO be penalized toward satisfying the far-field regime, not
    just toward a target stretch factor.
"""

from __future__ import annotations
import sympy as sp

# ── 1. Fresnel / total internal reflection: a HARD branch ───────────────────

_theta_i, _n1, _n2 = sp.symbols("theta_i n_1 n_2", positive=True)


def critical_angle(n1: float, n2: float) -> float:
    """theta_c = asin(n2/n1), valid only going from denser (n1) to rarer
    (n2) medium. Raises ValueError if n1 <= n2 (no TIR is possible in that
    direction -- there is no angle where the formula's sqrt goes imaginary)."""
    if n1 <= 0 or n2 <= 0:
        raise ValueError(f"n1, n2 must be > 0, got n1={n1}, n2={n2}")
    if n1 <= n2:
        raise ValueError(f"TIR requires n1 > n2 (denser to rarer medium), got n1={n1}, n2={n2}")
    import math
    return math.asin(n2 / n1)


def fresnel_reflectance_TE_symbolic():
    """R_TE(theta_i) as a SymPy Piecewise: the ordinary Fresnel TE
    reflectance below the critical angle, exactly 1 (total reflection)
    at/above it -- built with sp.Eq/comparisons on the SAME sin(theta_i)
    that appears in the ordinary-branch formula, so the Piecewise condition
    and the formula it gates are expressed in the same terms."""
    sin_t_transmitted = _n1 * sp.sin(_theta_i) / _n2
    cos_t_transmitted = sp.sqrt(1 - sin_t_transmitted**2)
    R_TE_ordinary = ((_n1 * sp.cos(_theta_i) - _n2 * cos_t_transmitted)
                      / (_n1 * sp.cos(_theta_i) + _n2 * cos_t_transmitted))**2
    is_tir = sin_t_transmitted >= 1
    expr = sp.Piecewise((sp.Integer(1), is_tir), (R_TE_ordinary, True))
    return expr, R_TE_ordinary


def verify_fresnel_continuous_at_critical_angle(n1: float, n2: float) -> bool:
    """CHECKED: unlike the far-field threshold below, R_TE is CONTINUOUS at
    theta_c -- the ordinary-branch formula's LIMIT as theta_i -> theta_c
    from below must equal 1 (the TIR branch's value), even though the
    formula's sqrt term becomes imaginary just past that point. This is
    what makes the Piecewise a legitimate continuous function, not just a
    convenient bookkeeping split."""
    _, R_ordinary = fresnel_reflectance_TE_symbolic()
    theta_c = critical_angle(n1, n2)
    n1_s, n2_s = sp.nsimplify(n1), sp.nsimplify(n2)
    R_at_theta_c = R_ordinary.subs({_n1: n1_s, _n2: n2_s})
    limit_from_below = sp.limit(R_at_theta_c, _theta_i, theta_c, dir="-")
    diff = sp.simplify(sp.N(limit_from_below) - 1)
    if abs(complex(diff)) > 1e-9:
        raise AssertionError(f"R_TE limit as theta_i -> theta_c^- should be 1, got {sp.N(limit_from_below)}")
    return True


def fresnel_reflectance_TE_torch(theta_i, n1: float, n2: float):
    """torch.where evaluates BOTH branches for every element (the ordinary
    Fresnel formula AND the constant 1), then selects per-element -- exactly
    dgs/conditional_formalism_sympy_torch.py's sinc(x) pattern. `safe_sin_t`
    clamps the transmitted sine to <=1 BEFORE it reaches the sqrt, so the
    discarded branch never produces a NaN gradient contribution even for
    angles past the critical angle. Imports torch lazily (py-3.12-only)."""
    import torch
    if n1 <= 0 or n2 <= 0:
        raise ValueError(f"n1, n2 must be > 0, got n1={n1}, n2={n2}")
    if n1 <= n2:
        raise ValueError(f"TIR requires n1 > n2 (denser to rarer medium), got n1={n1}, n2={n2}")

    sin_t = n1 / n2 * torch.sin(theta_i)
    is_tir = sin_t >= 1.0
    safe_sin_t = torch.where(is_tir, torch.zeros_like(sin_t), sin_t)
    cos_t = torch.sqrt(1 - safe_sin_t**2)

    R_ordinary = ((n1 * torch.cos(theta_i) - n2 * cos_t)
                  / (n1 * torch.cos(theta_i) + n2 * cos_t))**2
    return torch.where(is_tir, torch.ones_like(R_ordinary), R_ordinary)


def verify_torch_fresnel(n1: float = 1.5, n2: float = 1.0) -> dict:
    """Actually run the torch formalization and check it against three
    independently-known facts: (1) normal incidence gives the textbook
    R0=((n1-n2)/(n1+n2))^2, (2) R==1 exactly at and beyond theta_c, (3) R
    is continuous approaching theta_c from below (matches
    verify_fresnel_continuous_at_critical_angle's symbolic result)."""
    import torch
    import math

    # R has a SQRT-type singularity in its slope right at theta_c (R ~ 1 -
    # const*sqrt(theta_c - theta)) -- it approaches 1 continuously, but only
    # very close in; a 0.05 rad offset still leaves R well below 1 (~0.31
    # for n1=1.5, n2=1.0), so the "continuous approach" probe below needs a
    # genuinely tiny offset, not a generic small-looking one.
    theta_c = critical_angle(n1, n2)
    thetas = torch.tensor([0.0, theta_c - 1e-6, theta_c, theta_c + 0.05, 1.5],
                           dtype=torch.float64, requires_grad=True)
    R = fresnel_reflectance_TE_torch(thetas, n1, n2)

    checks = {}
    R0_expected = ((n1 - n2) / (n1 + n2))**2
    checks["normal incidence matches R0=((n1-n2)/(n1+n2))^2"] = \
        abs(R[0].item() - R0_expected) < 1e-10
    checks["R==1 exactly AT theta_c"] = abs(R[2].item() - 1.0) < 1e-10
    checks["R==1 exactly PAST theta_c"] = abs(R[3].item() - 1.0) < 1e-10
    checks["R continuous approaching theta_c from below"] = abs(R[1].item() - 1.0) < 0.01

    R.sum().backward()
    grads = thetas.grad
    checks["gradient finite everywhere (no NaN from the discarded TIR branch)"] = \
        bool(torch.isfinite(grads).all())

    return {"thetas": thetas.detach().numpy(), "R": R.detach().numpy(),
            "grads": grads.numpy(), "checks": checks}


# ── 2. Far-field dispersive-Fourier validity: a smooth DESIGN gate ──────────

_L, _L_D = sp.symbols("L L_D", positive=True)


def far_field_hard_piecewise_symbolic():
    """The existing plain-Python decision in dgs/dispersive_fourier.py
    (`far_field_ok = L_m > 10 * L_D`), formalized as a Piecewise penalty:
    0 if the far-field condition holds, 1 (maximally penalized) if not.
    UNLIKE the Fresnel case above, this is checked to be DISCONTINUOUS at
    L=10*L_D -- it's a bookkeeping threshold on an approximation's
    validity, not a physical field quantity that has to vary continuously."""
    penalty = sp.Piecewise((sp.Integer(0), _L > 10 * _L_D), (sp.Integer(1), True))
    return penalty


def verify_far_field_penalty_discontinuous(L_D_val: float = 1.0, eps: float = 1e-6) -> bool:
    """CHECKED: the values immediately left and right of L=10*L_D genuinely
    disagree (1 vs 0) -- confirming this threshold is NOT continuous, the
    opposite conclusion from verify_fresnel_continuous_at_critical_angle,
    and why a smooth surrogate (far_field_soft_gate below) is needed for
    gradient-based design rather than the hard threshold itself. Uses
    direct substitution at boundary +/- eps rather than SymPy's .limit(),
    which was found to return 0 on BOTH sides for this strict-inequality
    Piecewise -- a real SymPy limitation with Piecewise + strict
    inequalities, not a property of the function itself (direct
    substitution at nearby concrete points gives the expected 1 and 0)."""
    penalty = far_field_hard_piecewise_symbolic()
    penalty_fixed_L_D = penalty.subs(_L_D, L_D_val)
    boundary = 10 * L_D_val
    left = penalty_fixed_L_D.subs(_L, boundary - eps)
    right = penalty_fixed_L_D.subs(_L, boundary + eps)
    if left == right:
        raise AssertionError(f"expected a genuine discontinuity at L=10*L_D, got left={left} right={right}")
    return True


def far_field_soft_gate_torch(L, L_D, steepness: float = 5.0):
    """Smooth sigmoid surrogate for far_field_ok: sigmoid(k*(L/L_D - 10)).
    ->1 deep in the far-field regime, ->0 deep in the near-field regime,
    =0.5 exactly at the hard threshold L=10*L_D -- and, unlike the hard
    boolean, has a nonzero gradient everywhere, so it can sit inside a
    gradient-based fiber-length design loss (e.g. augmenting
    dgs/dispersive_fourier_torch.py's design_fiber_length_for_stretch_factor
    with a term that also pushes the design toward the regime where the
    far-field approximation is actually trustworthy)."""
    import torch
    if steepness <= 0:
        raise ValueError(f"steepness must be > 0, got {steepness}")
    return torch.sigmoid(steepness * (L / L_D - 10.0))


def verify_far_field_soft_gate() -> dict:
    """Actually run the soft gate and check it: ~0.5 exactly at the
    threshold, ->1 far into the far-field regime, ->0 far into the
    near-field regime, and a NONZERO gradient at the threshold itself
    (the entire point -- the hard boolean's gradient there is either 0 or
    undefined, giving a design optimizer no signal to work with)."""
    import torch
    L_D = torch.tensor(1.0, dtype=torch.float64)
    L_vals = torch.tensor([1.0, 5.0, 10.0, 15.0, 100.0], dtype=torch.float64, requires_grad=True)
    gate = far_field_soft_gate_torch(L_vals, L_D)

    checks = {}
    checks["gate(L=10*L_D) == 0.5 (exactly at hard threshold)"] = abs(gate[2].item() - 0.5) < 1e-9
    checks["gate(L>>10*L_D) close to 1"] = gate[4].item() > 0.999
    checks["gate(L<<10*L_D) close to 0"] = gate[0].item() < 0.001
    checks["gate is monotonically increasing in L"] = bool(torch.all(gate[1:] >= gate[:-1] - 1e-12))

    gate.sum().backward()
    checks["gradient at threshold is nonzero (informative for design)"] = abs(L_vals.grad[2].item()) > 1e-6

    return {"L_vals": L_vals.detach().numpy(), "gate": gate.detach().numpy(),
            "grads": L_vals.grad.numpy(), "checks": checks}


if __name__ == "__main__":
    print("=== 1. Fresnel / TIR: a HARD branch (SymPy Piecewise) ===")
    expr, _ = fresnel_reflectance_TE_symbolic()
    print(f"R_TE(theta_i) = {expr}")

    n1, n2 = 1.5, 1.0
    theta_c = critical_angle(n1, n2)
    import math
    print(f"\nn1={n1}, n2={n2}  ->  critical angle = {math.degrees(theta_c):.2f} deg")
    continuous = verify_fresnel_continuous_at_critical_angle(n1, n2)
    print(f"R_TE continuous at theta_c (limit from below == 1): {continuous}")

    print("\n=== torch.where: the differentiable Fresnel/TIR branch ===")
    try:
        result = verify_torch_fresnel(n1, n2)
        for name, ok in result["checks"].items():
            print(f"  {name}: {ok}")
        assert all(result["checks"].values())
    except ImportError as e:
        print(f"torch unavailable ({e}) -- run via `py -3.12 -m dgs.photonics_regime_decisions_torch`")

    print("\n=== 2. Far-field validity: a DISCONTINUOUS threshold ===")
    penalty = far_field_hard_piecewise_symbolic()
    print(f"penalty(L, L_D) = {penalty}")
    discontinuous = verify_far_field_penalty_discontinuous()
    print(f"Confirmed genuinely discontinuous at L=10*L_D (unlike the Fresnel case above): {discontinuous}")

    print("\n=== torch.sigmoid: a smooth, gradient-informative surrogate ===")
    try:
        result = verify_far_field_soft_gate()
        for name, ok in result["checks"].items():
            print(f"  {name}: {ok}")
        assert all(result["checks"].values())
        print("\nConfirmed: the soft gate gives a gradient-based fiber-length optimizer")
        print("(dgs/dispersive_fourier_torch.py's design_fiber_length_for_stretch_factor)")
        print("real design signal toward the far-field regime -- something the hard")
        print("boolean far_field_ok, with zero gradient almost everywhere, cannot.")
    except ImportError as e:
        print(f"torch unavailable ({e}) -- run via `py -3.12 -m dgs.photonics_regime_decisions_torch`")
