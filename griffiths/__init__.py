"""griffiths -- symbolic engine for Griffiths *Introduction to Electrodynamics* Ch. 1.

Vector calculus (separation vectors, product/quotient rules, (A.del)B),
parity/handedness (vectors vs pseudovectors), and Dirac-delta machinery.
Built on SymPy; designed to be driven from a Jupyter notebook with
``sympy.init_printing()`` active so every result renders as typeset math.
"""

from .vectors import (
    CARTESIAN,
    x, y, z, xp, yp, zp,
    separation_vector,
    separation_length,
    grad,
    div,
    curl,
    a_dot_del,
    grad_r_power,
    check_product_rule,
    check_quotient_rule,
    PRODUCT_RULES,
    QUOTIENT_RULES,
)
from .parity import (
    parity_matrix,
    handedness,
    invert,
    INVERSION_SIGNS,
)
from .deltas import (
    delta_integral,
    delta_rescale,
    step,
    d_step_dx,
    x_ddx_delta,
)
from .fields import (
    is_conservative,
    is_solenoidal,
    scalar_potential,
    rr2_field,
    rr2_paradox,
    radial_div_theorem,
    trig_substitution,
    sifting_property,
)
from . import relativity
from . import potentials
from . import quantum
from . import electrostatics
from . import potential_theory
from . import bessel
from . import dielectrics
from . import magnetostatics
from . import magnetic_matter
from . import hilbert

__version__ = "0.1.0"

__all__ = [
    "CARTESIAN", "x", "y", "z", "xp", "yp", "zp",
    "separation_vector", "separation_length",
    "grad", "div", "curl", "a_dot_del", "grad_r_power",
    "check_product_rule", "check_quotient_rule",
    "PRODUCT_RULES", "QUOTIENT_RULES",
    "parity_matrix", "handedness", "invert", "INVERSION_SIGNS",
    "delta_integral", "delta_rescale", "step", "d_step_dx", "x_ddx_delta",
    "is_conservative", "is_solenoidal", "scalar_potential",
    "rr2_field", "rr2_paradox", "radial_div_theorem",
    "trig_substitution", "sifting_property",
    "relativity", "potentials", "quantum", "electrostatics", "potential_theory",
    "bessel", "dielectrics", "magnetostatics", "magnetic_matter", "hilbert",
]
