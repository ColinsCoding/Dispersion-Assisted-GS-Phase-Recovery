"""General chemical-kinetics rate-law problems: rate = k[A]^n, its integrated
form, and half-life -- the bracket [A] always means molar concentration of A.

For a reaction A -> products with rate = -d[A]/dt = k[A]^n, integrating gives
a different closed form for each order n, and a DIFFERENT half-life
dependence on [A]_0 -- which is exactly how you tell the order of a reaction
from data: plot the right transform of [A] vs t and see which one is linear.

This is the general single-species case; dgs.biochem_kinetics covers the
enzyme-specific saturating case (Michaelis-Menten), which is NOT a simple
power-law in [S] and needs its own module.
"""

import numpy as np


def rate_law(concentrations, k, orders):
    """rate = k * prod_i [X_i]^order_i -- the general rate law for any number
    of species, e.g. rate_law({'A': 0.5, 'B': 0.2}, k=3.0, orders={'A':1,'B':2})."""
    rate = k
    for species, order in orders.items():
        rate *= concentrations[species] ** order
    return rate


# -- Single-species integrated rate laws: A -> products, rate = k[A]^n ---------

def integrated_concentration(A0, k, t, order):
    """[A](t) for a single-species reaction of the given order, by direct
    integration of d[A]/dt = -k[A]^order:

      order 0:  [A] = [A]0 - k t                    (linear decay, can hit 0 and go negative -- clipped)
      order 1:  [A] = [A]0 * exp(-k t)               (exponential decay)
      order 2:  [A] = [A]0 / (1 + k [A]0 t)          (hyperbolic decay)
    """
    t = np.asarray(t, dtype=float)
    if order == 0:
        return np.clip(A0 - k * t, 0, None)
    if order == 1:
        return A0 * np.exp(-k * t)
    if order == 2:
        return A0 / (1 + k * A0 * t)
    raise ValueError("order must be 0, 1, or 2")


def half_life(A0, k, order):
    """Time for [A] to fall to [A]0/2, by order -- each order has a
    DIFFERENT dependence on [A]0, which is the diagnostic signature used to
    determine reaction order from a half-life-vs-concentration experiment:

      order 0:  t_half = [A]0 / (2k)         (depends on [A]0)
      order 1:  t_half = ln(2) / k           (independent of [A]0 -- the
                                               textbook hallmark of 1st order)
      order 2:  t_half = 1 / (k [A]0)        (depends on [A]0, oppositely to order 0)
    """
    if order == 0:
        return A0 / (2 * k)
    if order == 1:
        return np.log(2) / k
    if order == 2:
        return 1.0 / (k * A0)
    raise ValueError("order must be 0, 1, or 2")


def linearized_transform(A, order):
    """The transform of [A] that is LINEAR in time for the given order --
    plotting this against t and checking for a straight line is the standard
    method/of-initial-rates-free way to determine reaction order from data:

      order 0: [A] itself is linear in t
      order 1: ln([A]) is linear in t
      order 2: 1/[A] is linear in t
    """
    A = np.asarray(A, dtype=float)
    if order == 0:
        return A
    if order == 1:
        return np.log(A)
    if order == 2:
        return 1.0 / A
    raise ValueError("order must be 0, 1, or 2")


def fit_reaction_order(t, A, candidate_orders=(0, 1, 2)):
    """Determine which candidate order best linearizes the data: transform
    [A] by each order's linearized_transform, fit a line by least squares,
    and return the order with the best linear fit (smallest residual) --
    the actual computational version of the "which plot is straightest"
    method used in a kinetics lab."""
    t = np.asarray(t, dtype=float)
    A = np.asarray(A, dtype=float)
    best_order, best_resid, best_fit = None, np.inf, None
    for order in candidate_orders:
        y = linearized_transform(A, order)
        X = np.vstack([t, np.ones_like(t)]).T
        coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ coeffs
        resid = float(np.sum((y - y_pred) ** 2))
        if resid < best_resid:
            best_order, best_resid, best_fit = order, resid, coeffs
    slope, intercept = best_fit
    return {"order": best_order, "residual": best_resid, "slope": slope, "intercept": intercept}


def arrhenius_rate_constant(A_prefactor, Ea_eV, T, k_B_eV=8.617333262e-5):
    """k = A * exp(-Ea / (k_B T)) -- the Arrhenius temperature dependence of
    a rate constant, with activation energy Ea in eV (the natural unit at
    the molecular-bond scale, ~1-5 eV for most chemical reactions) and k_B in
    eV/K so the exponent stays dimensionless without unit juggling."""
    return A_prefactor * np.exp(-Ea_eV / (k_B_eV * T))
