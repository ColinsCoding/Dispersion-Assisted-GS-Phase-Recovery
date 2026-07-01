"""Neural network spectral phase regression -- Paper [3] concept.

Based on: "Time-stretch accelerated convolutional neural network"
          (Jalali lab, neural network for time-stretch spectral recovery).

WHAT PAPER [3] DID:
  - Input:  two intensity waveforms I1(t), I2(t) from two fiber dispersions
  - Output: recovered field E(t) -- amplitude AND phase
  - How:    convolutional neural network trained on labeled (I1, I2) -> E pairs
  - Key advantage over iterative GS: breaks the conjugate phase ambiguity
    because the NN learns signal priors (smooth phase, bounded bandwidth)
    that vanilla GS ignores.

WHAT THIS MODULE DOES (numpy, no torch):
  - Implements a minimal 2-layer MLP in pure numpy with backprop
  - Trains on synthetic (I1, I2) -> phase pairs from dgs.gs_unsupervised
  - Shows that learned priors improve phase correlation vs iterative GS
  - Architecture: [2N] -> [hidden] -> [N]  (input: concat I1+I2, output: phase)
  - Loss: MSE on phase + self-consistency penalty

Run: py -3.13 scripts/run_nn_spectral_regression.py
     py -3.13 -m pytest tests/test_nn_spectral_regression.py -v
"""
import numpy as np


# ── Activation functions ──────────────────────────────────────────────────────

def relu(x):
    return np.maximum(0.0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def tanh_act(x):
    return np.tanh(x)

def tanh_grad(x):
    return 1.0 - np.tanh(x) ** 2


# ── MLP (2-layer, numpy backprop) ─────────────────────────────────────────────

class PhaseRegressionMLP:
    """Two-layer MLP: [2N] -> ReLU -> [hidden] -> tanh -> [N] -> phase in [-pi, pi].

    Input:  concatenated [I1, I2] (2N floats, normalized)
    Output: phase phi(t) in [-pi, pi]  (N floats)

    WHY THIS BREAKS THE CONJUGATE AMBIGUITY:
      Iterative GS satisfies |E|^2 = I1 AND |D(E)|^2 = I2.
      Both E and E* satisfy these. GS picks one arbitrarily.
      The MLP, trained on many signals, learns that real signals have
      smooth, slowly-varying phase -- E* has the OPPOSITE chirp.
      The learned prior selects the physically correct branch.
    """

    def __init__(self, N, hidden=64, lr=1e-3, rng=None):
        if N < 4:
            raise ValueError("N must be >= 4")
        if hidden < 4:
            raise ValueError("hidden must be >= 4")
        if lr <= 0:
            raise ValueError("lr must be positive")
        self.N = N
        self.hidden = hidden
        self.lr = float(lr)
        rng = rng or np.random.default_rng(0)
        # Xavier init
        scale1 = np.sqrt(2.0 / (2 * N))
        scale2 = np.sqrt(2.0 / hidden)
        self.W1 = rng.standard_normal((hidden, 2 * N)) * scale1
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((N, hidden)) * scale2
        self.b2 = np.zeros(N)
        self.train_losses = []

    def forward(self, x):
        """x: (2N,) -> phi: (N,)"""
        z1 = self.W1 @ x + self.b1
        a1 = relu(z1)
        z2 = self.W2 @ a1 + self.b2
        phi = np.pi * tanh_act(z2)   # output in (-pi, pi)
        return phi, (x, z1, a1, z2)

    def loss_and_grad(self, x, phi_true, I1, I2, D, lam_physics=0.1):
        """MSE loss on phase + optional self-consistency penalty."""
        phi_pred, cache = self.forward(x)
        inp, z1, a1, z2 = cache

        # MSE phase loss
        diff = phi_pred - phi_true
        mse = float(np.mean(diff ** 2))

        # self-consistency: |IFFT[H*FFT[sqrt(I1)*exp(i*phi)]]|^2 should match I2
        from dgs.gs_unsupervised import _disperse
        E_est = np.sqrt(np.maximum(I1, 0)) * np.exp(1j * phi_pred)
        I2_est = np.abs(_disperse(E_est, D)) ** 2
        phys_loss = float(np.mean((I2_est - I2) ** 2)) / (I2.max() ** 2 + 1e-12)

        total_loss = mse + lam_physics * phys_loss

        # backprop through MSE only (physics grad is expensive in numpy)
        dL_dphi = 2.0 * diff / len(diff)

        # through tanh output: phi = pi*tanh(z2)
        dL_dz2 = dL_dphi * np.pi * tanh_grad(z2)

        dL_dW2 = np.outer(dL_dz2, a1)
        dL_db2 = dL_dz2.copy()
        dL_da1 = self.W2.T @ dL_dz2

        # through ReLU hidden
        dL_dz1 = dL_da1 * relu_grad(z1)
        dL_dW1 = np.outer(dL_dz1, inp)
        dL_db1 = dL_dz1.copy()

        grads = (dL_dW1, dL_db1, dL_dW2, dL_db2)
        return total_loss, mse, phys_loss, grads

    def step(self, grads):
        dW1, db1, dW2, db2 = grads
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def predict(self, x):
        phi, _ = self.forward(x)
        return phi


# ── Dataset generation ────────────────────────────────────────────────────────

def make_training_sample(N, D1, D2, rng):
    """Generate one (I1, I2, phi_true) training sample.

    Signal: sum of 1-3 Gaussian pulses with random chirp.
    This is the signal PRIOR the NN learns implicitly.
    """
    from dgs.gs_unsupervised import _disperse
    t = np.linspace(-10, 10, N)
    n_pulses = rng.integers(1, 4)
    E = np.zeros(N, dtype=complex)
    for _ in range(n_pulses):
        center = rng.uniform(-5, 5)
        width = rng.uniform(0.5, 2.5)
        amp = rng.uniform(0.5, 1.0)
        chirp = rng.uniform(-0.3, 0.3)
        E += amp * np.exp(-(t - center)**2 / width**2) * np.exp(1j * chirp * np.pi * t**2)
    # normalize
    E = E / (np.abs(E).max() + 1e-12)
    I1 = np.abs(E) ** 2
    I2 = np.abs(_disperse(E, D2 - D1)) ** 2
    phi_true = np.angle(E)
    # input: concat normalized I1, I2
    x = np.concatenate([I1 / (I1.max() + 1e-12), I2 / (I2.max() + 1e-12)])
    return x, phi_true, I1, I2, E


def make_dataset(n_samples, N, D1=-5000.0, D2=-15000.0, seed=42):
    """Generate n_samples training examples."""
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if abs(D1) < 100 or abs(D2) < 100:
        raise ValueError("|D1|, |D2| must be >= 100")
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_samples):
        x, phi, I1, I2, E = make_training_sample(N, D1, D2, rng)
        samples.append({"x": x, "phi": phi, "I1": I1, "I2": I2, "E": E})
    return samples, D2 - D1


# ── Training loop ─────────────────────────────────────────────────────────────

def train(mlp, dataset, D, n_epochs=50, lam_physics=0.1, verbose=True):
    """Train the MLP on the dataset for n_epochs passes.

    Returns list of (epoch, mean_loss) tuples.
    """
    if n_epochs < 1:
        raise ValueError("n_epochs must be >= 1")
    history = []
    n = len(dataset)
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        idx = np.random.permutation(n)
        for i in idx:
            s = dataset[i]
            loss, mse, phys, grads = mlp.loss_and_grad(
                s["x"], s["phi"], s["I1"], s["I2"], D, lam_physics=lam_physics
            )
            mlp.step(grads)
            epoch_loss += loss
        mean_loss = epoch_loss / n
        history.append((epoch, mean_loss))
        if verbose and (epoch % max(1, n_epochs // 5) == 0 or epoch == n_epochs - 1):
            print(f"  epoch {epoch:4d}/{n_epochs}  loss={mean_loss:.4e}")
    return history


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(mlp, dataset):
    """Return mean phase correlation over the dataset."""
    corrs = []
    for s in dataset:
        phi_pred = mlp.predict(s["x"])
        # remove global phase offset
        offset = float(np.mean(phi_pred - s["phi"]))
        phi_adj = phi_pred - offset
        corr = float(np.corrcoef(s["phi"], phi_adj)[0, 1])
        corrs.append(corr)
    return float(np.mean(corrs)), float(np.std(corrs))


def compare_gs_vs_nn(N=128, D1=-5000.0, D2=-15000.0,
                     n_train=100, n_test=20, n_epochs=30, seed=0):
    """Full comparison: iterative GS vs trained MLP on phase correlation.

    Shows the key result: NN breaks conjugate ambiguity, GS does not.
    """
    from dgs.gs_unsupervised import unsupervised_gs, _disperse

    train_data, D = make_dataset(n_train, N, D1, D2, seed=seed)
    test_data, _  = make_dataset(n_test,  N, D1, D2, seed=seed + 999)

    print(f"\n  Training MLP: N={N}, D1={D1}, D2={D2}, "
          f"n_train={n_train}, n_epochs={n_epochs}")
    mlp = PhaseRegressionMLP(N, hidden=64, lr=5e-4, rng=np.random.default_rng(seed))
    history = train(mlp, train_data, D, n_epochs=n_epochs, lam_physics=0.05, verbose=True)

    nn_mean, nn_std = evaluate(mlp, test_data)

    # iterative GS baseline on same test set
    gs_corrs = []
    for s in test_data:
        I1, I2 = s["I1"], s["I2"]
        E_rec, _ = unsupervised_gs(I1, I2, D=D, n_iter=50, verbose=False)
        phi_rec = np.angle(E_rec)
        offset = float(np.mean(phi_rec - s["phi"]))
        corr = float(np.corrcoef(s["phi"], phi_rec - offset)[0, 1])
        gs_corrs.append(corr)
    gs_mean = float(np.mean(gs_corrs))
    gs_std  = float(np.std(gs_corrs))

    print(f"\n  {'Method':<20} {'Phase corr (mean)':<22} {'Std'}")
    print(f"  {'-'*55}")
    print(f"  {'Iterative GS':<20} {gs_mean:<22.4f} {gs_std:.4f}")
    print(f"  {'Trained MLP':<20} {nn_mean:<22.4f} {nn_std:.4f}")
    print(f"\n  NN improvement: {nn_mean - gs_mean:+.4f}  "
          f"({'better' if nn_mean > gs_mean else 'worse -- train longer'})")
    print(f"\n  WHY NN CAN WIN:")
    print(f"  GS picks E or E* arbitrarily (50/50 coin flip on each sample).")
    print(f"  NN sees patterns in I1+I2 that distinguish E from E*")
    print(f"  because real signals have smooth, slowly-varying phase (chirp),")
    print(f"  while E* has the conjugate chirp -- different spectral shape.")

    return {
        "gs_phase_corr_mean": gs_mean,
        "gs_phase_corr_std":  gs_std,
        "nn_phase_corr_mean": nn_mean,
        "nn_phase_corr_std":  nn_std,
        "nn_improvement": nn_mean - gs_mean,
        "train_history": history,
        "mlp": mlp,
    }
