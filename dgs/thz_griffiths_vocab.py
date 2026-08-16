"""thz_griffiths_vocab.py -- a plain-English glossary of every symbol and
term used in dgs/thz_waveguide_dispersion_relation.py and
dgs/cylindrical_waveguide_resonance.py, as a pandas DataFrame.

WHY THIS EXISTS: the THz waveguide module's docstrings and code use math
notation the way code has to write it (omega_c, beta_2, k_c) instead of the
way a textbook typesets it (omega_c, beta_2, k_c). This module is the
lookup table connecting the two, for reading the code and notebook without
re-deriving what each symbol means every time.

Run: py -3.13 -c "from dgs.thz_griffiths_vocab import vocab_table; print(vocab_table())"
"""
import pandas as pd

VOCAB = [
    {
        "symbol": "omega  (ω)",
        "name": "angular frequency",
        "meaning": "How fast the wave oscillates in time, in radians/second (omega = 2*pi*f).",
        "formula": "omega = 2πf",
        "source": "dgs/thz_waveguide_dispersion_relation.py",
    },
    {
        "symbol": "omega_c  (ωᴄ)",
        "name": "cutoff frequency",
        "meaning": "The lowest frequency that can actually propagate down the waveguide. "
                   "Below it, the wave dies out exponentially instead of traveling (evanescent).",
        "formula": "omega_c = c·k_c",
        "source": "dgs/cylindrical_waveguide_resonance.py waveguide_cutoff_frequency()",
    },
    {
        "symbol": "k",
        "name": "wavenumber / propagation constant",
        "meaning": "How many radians of phase the wave accumulates per meter traveled down the guide.",
        "formula": "k(omega) = sqrt(omega² - omega_c²) / c",
        "source": "dgs/thz_waveguide_dispersion_relation.py waveguide_wavenumber()",
    },
    {
        "symbol": "k_c",
        "name": "transverse (cutoff) wavenumber",
        "meaning": "The wavenumber of the field's cross-sectional pattern, set purely by the "
                   "waveguide's shape and size (radius for circular, width/height for rectangular) "
                   "-- not by the signal frequency at all.",
        "formula": "k_c = j'_(m,n) / a   (circular, TE)   or   k_c = pi*sqrt((m/a)²+(n/b)²)   (rectangular)",
        "source": "dgs/cylindrical_waveguide_resonance.py radial_wavenumber(); "
                  "dgs/thz_waveguide_dispersion_relation.py rectangular_waveguide_cutoff_frequency()",
    },
    {
        "symbol": "m_eff",
        "name": "effective photon mass",
        "meaning": "A real, nonzero mass a photon behaves as if it has, purely because it's "
                   "confined in the waveguide -- the photon itself has no actual rest mass.",
        "formula": "m_eff = ħ·omega_c / c²",
        "source": "dgs/thz_waveguide_dispersion_relation.py effective_photon_mass()",
    },
    {
        "symbol": "v_phase (v_p)",
        "name": "phase velocity",
        "meaning": "The speed at which a single crest of the wave appears to move. Exceeds c here, "
                   "but carries no signal/energy, so this does not violate relativity.",
        "formula": "v_p = omega / k",
        "source": "dgs/thz_waveguide_dispersion_relation.py phase_velocity()",
    },
    {
        "symbol": "v_group (v_g)",
        "name": "group velocity",
        "meaning": "The speed at which energy/information actually travels -- always below c.",
        "formula": "v_g = d(omega)/d(k)",
        "source": "dgs/thz_waveguide_dispersion_relation.py group_velocity()",
    },
    {
        "symbol": "beta_2  (β₂)",
        "name": "group velocity dispersion (GVD)",
        "meaning": "How much the group velocity itself changes with frequency -- the reason a "
                   "pulse (a bundle of frequencies) spreads out in time as it travels. This module's "
                   "central quantity: proven always negative here (Problems 1 & 2).",
        "formula": "β₂ = d²k/d(omega)² = -omega_c² / (c·(omega²-omega_c²)^1.5)",
        "source": "dgs/thz_waveguide_dispersion_relation.py group_velocity_dispersion()",
    },
    {
        "symbol": "Delta_t",
        "name": "pulse broadening",
        "meaning": "How much a pulse's duration grows after traveling distance L, due to GVD.",
        "formula": "Δt ≈ |β₂| · L · (bandwidth)",
        "source": "dgs/thz_waveguide_dispersion_relation.py thz_pulse_broadening()",
    },
    {
        "symbol": "TE / TM",
        "name": "transverse electric / transverse magnetic mode",
        "meaning": "TE: the electric field has no component along the guide's axis. "
                   "TM: the magnetic field has no component along the guide's axis. Each has its "
                   "own boundary condition, hence its own allowed k_c values.",
        "formula": "TE: J'_m(k_c·a)=0    TM: J_m(k_c·a)=0   (circular guide)",
        "source": "dgs/cylindrical_waveguide_resonance.py",
    },
    {
        "symbol": "J_m, J'_m",
        "name": "Bessel function (and its derivative)",
        "meaning": "The natural radial shape a wave takes when confined inside a circle -- the "
                   "circular-waveguide analog of sine/cosine for a rectangular guide.",
        "formula": "R(r) = J_m(k_c·r)",
        "source": "dgs/cylindrical_waveguide_resonance.py radial_mode_profile()",
    },
    {
        "symbol": "E²=(pc)²+(mc²)²",
        "name": "relativistic dispersion relation",
        "meaning": "The energy-momentum relation for a particle of rest mass m. This module's core "
                   "result: substituting E=ħω, p=ħk, m=ħω_c/c² turns this into the waveguide's own "
                   "dispersion relation, exactly.",
        "formula": "E² = (pc)² + (mc²)²",
        "source": "dgs/thz_waveguide_dispersion_relation.py verify_waveguide_matches_relativistic_dispersion()",
    },
    {
        "symbol": "α (alpha)",
        "name": "linear thermal expansion coefficient",
        "meaning": "How much a material's length grows per degree of temperature rise. Used in "
                   "Problem 4 to check whether a hot waveguide disperses meaningfully differently.",
        "formula": "a(T) = a₀·(1 + α·ΔT)",
        "source": "dgs/thz_waveguide_dispersion_relation.py thermal_broadening_shift()",
    },
    {
        "symbol": "∇² (nabla squared)",
        "name": "Laplacian (Helmholtz equation)",
        "meaning": "The equation any confined wave must satisfy: nabla^2*psi + k^2*psi = 0. "
                   "Separating it into transverse (x,y) and axial (z) parts is where the whole "
                   "omega²=c²(k²+k_c²) relation comes from (Problem 5), for ANY cross-section shape.",
        "formula": "∇²ψ + k²ψ = 0",
        "source": "dgs/cylindrical_waveguide_resonance.py module docstring; "
                  "dgs/thz_waveguide_dispersion_relation.py verify_dispersion_relation_is_geometry_independent()",
    },
]


def vocab_table() -> pd.DataFrame:
    """Return the glossary as a pandas DataFrame, one row per term."""
    return pd.DataFrame(VOCAB, columns=["symbol", "name", "meaning", "formula", "source"])


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # Windows console defaults to cp1252
    df = vocab_table()
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 200)
    print(df[["symbol", "name", "formula"]].to_string(index=False))
    out = "physics_repo/docs/thz_griffiths_vocab.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved full glossary (with meanings + source files) to {out}")
