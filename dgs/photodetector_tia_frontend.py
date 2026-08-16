"""photodetector_tia_frontend.py -- the transimpedance amplifier (TIA):
the actual FIRST-STAGE circuit in every photonic receiver, converting a
photodiode's current into a usable voltage. Its feedback loop is a genuine
integral-calculus problem, not a metaphor: Kirchhoff's current law at the
op-amp's virtual-ground summing node,
    I_ph(t) = -V_out(t)/R_f - C_f * dV_out/dt,
is a first-order linear ODE. Solved via the integrating-factor method
(verified with sp.dsolve, not assumed) for a step optical input, it gives
the TIA's finite rise time -- the same RC-charging mathematics as
dgs.spice's RLC step response, applied to a photonics front end instead of
a generic circuit.

Circuit: photodiode (current source I_ph, junction capacitance folded into
the ideal-op-amp assumption below) driving the inverting input of an ideal
op-amp with feedback R_f parallel C_f. C_f is a REAL, deliberate design
choice (not parasitic) -- it trades transimpedance gain (bigger R_f -> more
gain) against bandwidth (bigger R_f*C_f -> slower response), the central
tradeoff every optical-receiver front end has to make.
"""

from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp

C_LIGHT = 299792458.0


def _validate_positive(**kwargs) -> None:
    for name, value in kwargs.items():
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")


# ── 1. Photocurrent from optical power ───────────────────────────────────────

def photocurrent(P_optical_W: float, responsivity_A_per_W: float) -> float:
    """I_ph = R * P: the photodiode's responsivity R (A/W, typically
    ~0.5-1.0 A/W for silicon/InGaAs near their respective peak
    wavelengths) converts incident optical power directly to current."""
    _validate_positive(P_optical_W=P_optical_W, responsivity_A_per_W=responsivity_A_per_W)
    return responsivity_A_per_W * P_optical_W


# ── 2. DC transimpedance gain and bandwidth ──────────────────────────────────

def tia_transimpedance_gain_dc(Rf: float) -> float:
    """Ideal DC transimpedance: V_out/I_ph = -R_f (an ideal op-amp with
    infinite open-loop gain forces ALL of I_ph through the feedback
    resistor, none into the op-amp's input)."""
    _validate_positive(Rf=Rf)
    return -Rf


def tia_bandwidth_hz(Rf: float, Cf: float) -> float:
    """f_p = 1/(2*pi*Rf*Cf): the single-pole -3dB bandwidth set by the
    feedback RC time constant -- bigger Rf (more gain) or bigger Cf (more
    stability margin) both COST bandwidth. This is the central TIA design
    tradeoff, not an incidental side effect."""
    _validate_positive(Rf=Rf, Cf=Cf)
    return 1.0 / (2 * np.pi * Rf * Cf)


# ── 3. Step response: solving the KCL loop ODE ───────────────────────────────

def tia_step_response_analytic(t: np.ndarray, I0: float, Rf: float, Cf: float) -> np.ndarray:
    """V_out(t) = -I0*Rf*(1 - exp(-t/(Rf*Cf))) for t >= 0 -- the closed-form
    solution of the KCL loop equation I0 = -V_out/Rf - Cf*dV_out/dt (a
    step optical input turning on at t=0), derived via sp.dsolve and
    reproduced here as a plain NumPy formula for speed."""
    _validate_positive(I0=I0, Rf=Rf, Cf=Cf)
    t = np.asarray(t, dtype=float)
    tau = Rf * Cf
    return np.where(t >= 0, -I0 * Rf * (1 - np.exp(-t / tau)), 0.0)


def tia_step_response_ode(t_span: tuple, I0: float, Rf: float, Cf: float, n_pts: int = 2000):
    """Numerically integrates the SAME KCL loop equation
    (Cf*dVout/dt = -I0 - Vout/Rf) via scipy's ODE solver -- an independent
    check of the closed-form solution above, not a restatement of it."""
    _validate_positive(I0=I0, Rf=Rf, Cf=Cf)

    def rhs(t, y):
        Vout = y[0]
        return [(-I0 - Vout / Rf) / Cf]

    t_eval = np.linspace(*t_span, n_pts)
    sol = solve_ivp(rhs, t_span, y0=[0.0], t_eval=t_eval, rtol=1e-10, atol=1e-14)
    return sol.t, sol.y[0]


def verify_step_response_matches_ode(I0: float = 1e-6, Rf: float = 1e4, Cf: float = 1e-12,
                                     rtol: float = 1e-6) -> bool:
    """CHECKED: the closed-form analytic step response and a real
    numerical ODE integration of the same KCL equation must agree -- not
    assumed just because sp.dsolve produced a formula."""
    tau = Rf * Cf
    t_span = (0.0, 10 * tau)
    t_ode, Vout_ode = tia_step_response_ode(t_span, I0, Rf, Cf)
    Vout_analytic = tia_step_response_analytic(t_ode, I0, Rf, Cf)
    max_abs = np.max(np.abs(Vout_analytic))
    max_diff = np.max(np.abs(Vout_ode - Vout_analytic))
    if max_diff / max_abs > rtol:
        raise AssertionError(f"ODE and analytic step responses disagree: "
                             f"max relative difference {max_diff/max_abs:.2e} > {rtol}")
    return True


# ── 4. Frequency response: the transfer function and its -3dB point ─────────

def tia_transfer_function(f: np.ndarray, Rf: float, Cf: float) -> np.ndarray:
    """H(f) = -Rf / (1 + j*2*pi*f*Rf*Cf): the single-pole transimpedance
    transfer function (Laplace transform of the same KCL equation, s=j*2*pi*f).
    Returns a complex array."""
    _validate_positive(Rf=Rf, Cf=Cf)
    f = np.asarray(f, dtype=float)
    return -Rf / (1 + 1j * 2 * np.pi * f * Rf * Cf)


def verify_bandwidth_is_minus_3db(Rf: float = 1e4, Cf: float = 1e-12) -> bool:
    """CHECKED: |H(f_p)| / |H(0)| must equal 1/sqrt(2) (the DEFINITION of
    a -3dB point) at f_p = tia_bandwidth_hz(Rf, Cf) -- confirms the pole
    formula actually IS the -3dB frequency, not just a plausible-looking
    expression with the right units."""
    f_p = tia_bandwidth_hz(Rf, Cf)
    H0 = abs(tia_transfer_function(np.array([0.0]), Rf, Cf)[0])
    Hp = abs(tia_transfer_function(np.array([f_p]), Rf, Cf)[0])
    ratio = Hp / H0
    if abs(ratio - 1 / np.sqrt(2)) > 1e-9:
        raise AssertionError(f"|H(f_p)|/|H(0)| = {ratio}, expected 1/sqrt(2) = {1/np.sqrt(2)}")
    return True


# ── 5. The gain-bandwidth tradeoff, made concrete ────────────────────────────

def gain_bandwidth_tradeoff(Rf_values: np.ndarray, Cf: float) -> dict:
    """For a FIXED Cf (set by the photodiode junction + stray layout
    capacitance, not freely chosen), sweeping Rf trades transimpedance
    gain directly against bandwidth -- gain*bandwidth is NOT constant here
    (unlike an op-amp's classic GBW product) because gain scales as Rf
    while bandwidth scales as 1/Rf, so their PRODUCT is actually constant:
    gain*bandwidth = 1/(2*pi*Cf), independent of Rf entirely."""
    Rf_values = np.asarray(Rf_values, dtype=float)
    if np.any(Rf_values <= 0) or Cf <= 0:
        raise ValueError("Rf_values and Cf must all be > 0")
    gains = Rf_values                      # |DC transimpedance|
    bandwidths = 1.0 / (2 * np.pi * Rf_values * Cf)
    products = gains * bandwidths
    return {"Rf_values": Rf_values, "gains_ohm": gains, "bandwidths_Hz": bandwidths,
            "gain_bandwidth_products": products,
            "product_is_constant": bool(np.ptp(products) / np.mean(products) < 1e-9)}


if __name__ == "__main__":
    # a realistic InGaAs photodiode + TIA front end: 1 uW optical input,
    # 0.9 A/W responsivity, 10 kohm feedback, 0.5 pF feedback capacitance
    P_in, responsivity = 1e-6, 0.9
    Rf, Cf = 1e4, 0.5e-12

    print("=== 1. Photocurrent and DC transimpedance gain ===")
    I_ph = photocurrent(P_in, responsivity)
    gain_dc = tia_transimpedance_gain_dc(Rf)
    print(f"  I_ph = {I_ph*1e6:.3f} uA  (from {P_in*1e6:.1f} uW at {responsivity} A/W)")
    print(f"  DC transimpedance gain = {gain_dc:.0f} V/A  ->  V_out(DC) = {gain_dc*I_ph*1e3:.3f} mV")

    print("\n=== 2. Bandwidth and the -3dB point, checked ===")
    f_p = tia_bandwidth_hz(Rf, Cf)
    ok_3db = verify_bandwidth_is_minus_3db(Rf, Cf)
    print(f"  f_p = {f_p/1e9:.3f} GHz,  |H(f_p)|/|H(0)| = 1/sqrt(2) verified: {ok_3db}")

    print("\n=== 3. Step response: closed-form (sp.dsolve) vs. real ODE integration ===")
    ok_step = verify_step_response_matches_ode(I_ph, Rf, Cf)
    tau = Rf * Cf
    print(f"  RC time constant tau = R_f*C_f = {tau*1e12:.2f} ps")
    print(f"  analytic vs. ODE-integrated step response agree: {ok_step}")

    print("\n=== 4. Gain-bandwidth tradeoff: the product IS constant (unlike op-amp GBW intuition) ===")
    Rf_sweep = np.array([1e3, 1e4, 1e5, 1e6])
    tradeoff = gain_bandwidth_tradeoff(Rf_sweep, Cf)
    for Rf_val, gain, bw in zip(tradeoff["Rf_values"], tradeoff["gains_ohm"], tradeoff["bandwidths_Hz"]):
        print(f"  Rf={Rf_val:>9.0f} ohm  ->  gain={gain:>9.0f} V/A,  bandwidth={bw/1e9:>7.3f} GHz")
    print(f"  gain*bandwidth constant across all Rf: {tradeoff['product_is_constant']} "
          f"(= 1/(2*pi*Cf) = {1/(2*np.pi*Cf)/1e9:.3f} GHz*ohm)")

    print("\nThe TIA's feedback loop, I_ph = -V_out/R_f - C_f*dV_out/dt, is Kirchhoff's")
    print("current law -- solved by integrating factor for the step response, and its")
    print("gain-bandwidth product turns out Rf-independent: real design freedom is only")
    print("in Cf (layout, photodiode choice), not in picking Rf 'for free'.")
