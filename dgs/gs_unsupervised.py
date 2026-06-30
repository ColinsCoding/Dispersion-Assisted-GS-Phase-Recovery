"""Project 5: unsupervised / self-supervised phase retrieval
(Deep Dispersion Prior — no ground-truth phase required).

The key innovation over gs_core: we never have the true phase to compare
against. Instead we minimise a self-consistency loss across multiple
dispersion realisations of the SAME unknown field.

Ghost tracking (Mario Kart model)
----------------------------------
Like the ghost car in Mario Kart, we keep a frozen copy of the *best*
field seen so far (lowest loss). At any point we can "rewind to the ghost"
rather than continuing from the current (possibly worse) iterate. This is
critical in non-convex optimisation: GS can wander away from a good basin.

Architecture
------------
  - gs_prior_step()    : one iteration of self-supervised GS
  - GhostTracker       : maintains best-ever iterate + loss history
  - unsupervised_gs()  : full loop with ghost tracking + convergence guard
  - DeepDispersionPrior: thin PINN wrapper (torch, py-3.12 only)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Ghost tracker — best-iterate memory across GS iterations
# ---------------------------------------------------------------------------

@dataclass
class GhostTracker:
    """Tracks the lowest-loss field seen during optimisation.

    Analogy: Mario Kart ghost car = transparent replay of your best lap.
    Here the 'ghost' is the complex field E with the lowest self-consistency
    loss seen so far. At any point you can call .rewind() to recover it.

    Attributes
    ----------
    ghost_field : complex array of the best field found, or None before first update
    ghost_loss  : corresponding loss value
    history     : (iteration, loss) list for the full run
    n_rewinds   : how many times the ghost was rewound to
    """
    ghost_field: Optional[np.ndarray] = None
    ghost_loss: float = float('inf')
    history: List[Tuple[int, float]] = field(default_factory=list)
    n_rewinds: int = 0

    def update(self, iteration: int, E: np.ndarray, loss: float) -> bool:
        """Offer a new iterate. Returns True if this became the new ghost."""
        self.history.append((iteration, loss))
        if loss < self.ghost_loss:
            self.ghost_field = E.copy()
            self.ghost_loss  = loss
            return True
        return False

    def rewind(self) -> np.ndarray:
        """Return the ghost field (best seen so far). Increments rewind counter."""
        if self.ghost_field is None:
            raise RuntimeError("no ghost yet — call update() at least once")
        self.n_rewinds += 1
        return self.ghost_field.copy()

    def since_last_improvement(self) -> int:
        """Number of iterations since the ghost was last updated (0 = just now)."""
        if not self.history:
            return 0
        # scan backwards for the most recent time ghost_loss was achieved
        for k in range(len(self.history) - 1, -1, -1):
            if self.history[k][1] <= self.ghost_loss * 1.0001:
                return len(self.history) - 1 - k
        return len(self.history)

    def summary(self) -> str:
        n = len(self.history)
        if n == 0:
            return "GhostTracker: no data"
        best_iter = self.history[[l for _, l in self.history].index(self.ghost_loss)][0]
        return (f"GhostTracker: {n} iters | "
                f"best loss={self.ghost_loss:.4e} at iter {best_iter} | "
                f"rewinds={self.n_rewinds}")


# ---------------------------------------------------------------------------
# Self-consistency loss for unsupervised GS
# ---------------------------------------------------------------------------

def _fft(x: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x)))


def _ifft(x: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x)))


def _disperse(E: np.ndarray, D: float) -> np.ndarray:
    """Apply GVD dispersion exp(i D omega^2 / 2) in the frequency domain."""
    N  = len(E)
    omega = np.fft.fftshift(np.fft.fftfreq(N)) * 2 * np.pi
    return _ifft(_fft(E) * np.exp(1j * D * omega**2 / 2))


def self_consistency_loss(E: np.ndarray, I1: np.ndarray, I2: np.ndarray,
                          D: float) -> float:
    """RMS mismatch between model intensities and measured intensities.

    Loss = sqrt(mean( (|E|² - I1)² )) + sqrt(mean( (|D(E)|² - I2)² ))

    This is zero only when E simultaneously fits both intensity measurements,
    with no ground-truth phase required.
    """
    err1 = np.sqrt(np.mean((np.abs(E)**2 - I1)**2))
    E_d  = _disperse(E, D)
    err2 = np.sqrt(np.mean((np.abs(E_d)**2 - I2)**2))
    return float(err1 + err2)


# ---------------------------------------------------------------------------
# GS prior step (one iteration, no ground truth)
# ---------------------------------------------------------------------------

def gs_prior_step(E: np.ndarray, I1: np.ndarray, I2: np.ndarray,
                  D: float) -> np.ndarray:
    """One Gerchberg-Saxton iteration for unsupervised phase retrieval.

    Enforces amplitude constraints from I1 (at D=0) and I2 (at dispersion D)
    alternately — same logic as gs_core.gs_iteration but with no reference phase.
    """
    # enforce I1 amplitude in direct domain
    amp = np.sqrt(np.maximum(I1, 0))
    mask = amp > 0
    E_out = E.copy()
    E_out[mask] = amp[mask] * np.exp(1j * np.angle(E[mask]))

    # disperse, enforce I2, undisperse
    E_d   = _disperse(E_out, D)
    amp2  = np.sqrt(np.maximum(I2, 0))
    mask2 = amp2 > 0
    E_d[mask2] = amp2[mask2] * np.exp(1j * np.angle(E_d[mask2]))
    E_out = _disperse(E_d, -D)   # undisperse

    return E_out


# ---------------------------------------------------------------------------
# Full unsupervised GS loop with ghost tracking
# ---------------------------------------------------------------------------

def unsupervised_gs(
    I1: np.ndarray,
    I2: np.ndarray,
    D: float,
    n_iter: int = 200,
    rewind_patience: int = 30,
    verbose: bool = True,
) -> Tuple[np.ndarray, GhostTracker]:
    """Unsupervised GS phase retrieval with ghost tracking.

    Parameters
    ----------
    I1          : measured intensity before dispersion (direct domain)
    I2          : measured intensity after dispersion D
    D           : dispersion value (|D| >= 100 for meaningful diversity)
    n_iter      : maximum number of GS iterations
    rewind_patience : rewind to ghost after this many non-improving iterations
    verbose     : print progress

    Returns
    -------
    E_best : complex field (the ghost — lowest self-consistency loss)
    ghost  : GhostTracker with full loss history
    """
    if abs(D) < 100:
        raise ValueError(f"|D| must be >= 100 for GS convergence; got D={D}")
    if n_iter <= 0:
        raise ValueError("n_iter must be positive")
    if np.any(I1 < 0) or np.any(I2 < 0):
        raise ValueError("intensities must be non-negative")

    N = len(I1)
    rng = np.random.default_rng(42)

    # initialise with random phase, measured amplitude
    phase0 = rng.uniform(-np.pi, np.pi, N)
    E = np.sqrt(np.maximum(I1, 0)) * np.exp(1j * phase0)

    ghost = GhostTracker()
    loss0 = self_consistency_loss(E, I1, I2, D)
    ghost.update(0, E, loss0)

    last_rewind_iter = -rewind_patience  # cooldown: don't rewind again too soon

    for i in range(1, n_iter + 1):
        E = gs_prior_step(E, I1, I2, D)
        loss = self_consistency_loss(E, I1, I2, D)
        is_new_best = ghost.update(i, E, loss)

        if verbose and (i % 25 == 0 or i == 1):
            marker = " <-- ghost" if is_new_best else ""
            print(f"  iter {i:4d}  loss={loss:.4e}{marker}")

        # rewind to ghost if stagnating, with cooldown to avoid infinite loops
        stale = ghost.since_last_improvement()
        cooldown_ok = (i - last_rewind_iter) >= rewind_patience
        if stale >= rewind_patience and cooldown_ok:
            if verbose:
                print(f"  [ghost rewind at iter {i}]  stale for {stale} iters")
            E = ghost.rewind()
            # perturb to escape the basin
            E += rng.normal(0, ghost.ghost_loss * 0.05, N) * np.exp(
                1j * rng.uniform(-np.pi, np.pi, N))
            last_rewind_iter = i

    if verbose:
        print(ghost.summary())

    return ghost.rewind(), ghost


# ---------------------------------------------------------------------------
# Demo with synthetic data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N   = 512
    D   = -800.0
    t   = np.linspace(-10, 10, N)
    # True field: two Gaussian pulses (unknown phase)
    E_true = (np.exp(-t**2 / 0.5) + 0.6 * np.exp(-(t - 3)**2 / 0.3)
              ) * np.exp(1j * np.pi * t / 5)
    I1 = np.abs(E_true)**2
    I2 = np.abs(_disperse(E_true, D))**2

    print(f"Unsupervised GS demo  |  N={N}  D={D}")
    print(f"True field: two-pulse, unknown phase")
    print()
    E_rec, ghost = unsupervised_gs(I1, I2, D, n_iter=200, verbose=True)

    # quality: correlation of recovered |E|² with true |E|²
    corr = float(np.corrcoef(np.abs(E_rec)**2, I1)[0, 1])
    print(f"\nAmplitude correlation with ground truth: {corr:.4f}")
    print("(1.0 = perfect; phase is determined up to a global constant)")
