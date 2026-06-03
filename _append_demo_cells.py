"""Append §12 (3D pipe) and §13 (CNN + upload) to dispersion_gs_demo.ipynb."""
import json, textwrap

NB_PATH = 'notebooks/dispersion_gs_demo.ipynb'

with open(NB_PATH) as f:
    nb = json.load(f)


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}


# ─── §12 · 3D pipe ────────────────────────────────────────────────────────────
MD12 = textwrap.dedent("""\
    ## §12 · 3D pipe phase retrieval

    `retrieve_phase_pipe` processes a **cylindrical signal stack** $\\phi(\\theta, z, t)$.

    Physical picture: fibers, waveguide arrays, or distributed sensors on a cylinder.
    - **Axial continuity**: removes z-drift between adjacent cross-sections.
    - **Angular continuity**: closes the wrap-around seam at $\\theta=0 \\leftrightarrow 2\\pi$.

    Stack shape: `(N_theta, N_z, N_t)` — angular × axial × temporal.
""")

CODE12_BUILD = textwrap.dedent("""\
    from gs_core import retrieve_phase_pipe, pipe_surface_plot

    N_THETA, N_Z, N_T = 8, 6, 512
    D1_p, D2_p = -5000.0, -5750.0

    I1_pipe       = np.zeros((N_THETA, N_Z, N_T))
    I2_pipe       = np.zeros((N_THETA, N_Z, N_T))
    phi_true_pipe = np.zeros((N_THETA, N_Z, N_T))

    seed = 0
    for i in range(N_THETA):
        for j in range(N_Z):
            d = make_measurements('QPSK', n_symbols=64, sps=8,
                                  D1=D1_p, D2=D2_p, snr_db=25.0, rng_seed=seed)
            seed += 1
            n = min(N_T, len(d['I1']))
            I1_pipe[i, j, :n]       = d['I1'][:n]
            I2_pipe[i, j, :n]       = d['I2'][:n]
            phi_true_pipe[i, j, :n] = d['phi_true'][:n]

    print(f'Pipe stack: {I1_pipe.shape}  ({N_THETA}θ x {N_Z}z x {N_T}t)')
    phi_pipe, errors_pipe = retrieve_phase_pipe(
        I1_pipe, I2_pipe, D1_p, D2_p,
        n_iter=50, unit_amplitude=True,
        angular_continuity=True, axial_continuity=True,
    )

    rms_pipe = np.zeros((N_THETA, N_Z))
    for i in range(N_THETA):
        for j in range(N_Z):
            off = np.angle(np.mean(np.exp(1j*(phi_true_pipe[i,j]-phi_pipe[i,j]))))
            dlt = np.angle(np.exp(1j*(phi_pipe[i,j]+off-phi_true_pipe[i,j])))
            rms_pipe[i,j] = np.degrees(np.sqrt(np.mean(dlt**2)))

    print(f'Mean RMS: {rms_pipe.mean():.2f} deg +/- {rms_pipe.std():.2f} deg')
""")

CODE12_PLOT = textwrap.dedent("""\
    # Pipe surface visualization
    fig, ax3 = pipe_surface_plot(phi_pipe, t_slice=0)
    plt.savefig('notebooks/pipe_surface.png', bbox_inches='tight')
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    im = axes[0].imshow(rms_pipe, cmap='hot_r', aspect='auto')
    axes[0].set_xlabel('Axial index z'); axes[0].set_ylabel('Angular index theta')
    axes[0].set_title('RMS phase error (deg) on pipe surface')
    plt.colorbar(im, ax=axes[0], label='deg')

    mean_errs = errors_pipe.mean(axis=(0, 1))
    axes[1].semilogy(mean_errs, 'o-', color='crimson', markersize=4)
    axes[1].set_title('Mean GS convergence across pipe nodes')
    axes[1].set_xlabel('Iteration'); axes[1].set_ylabel('Mean amplitude error')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'3D Pipe phase retrieval  ({N_THETA}theta x {N_Z}z)', fontweight='bold')
    plt.tight_layout()
    plt.savefig('notebooks/demo_pipe_rms.png', bbox_inches='tight')
    plt.show()
""")

# ─── §13 · CNN + upload widget ────────────────────────────────────────────────
MD13 = textwrap.dedent("""\
    ## §13 · CNN phase refinement + real-data upload widget

    Two sub-sections:
    - **13a** — 1D CNN (PyTorch): input $(I_1, I_2, \\hat{\\phi}_{GS})$ → output $\\hat{\\phi}_{refined}$.
    - **13b** — File upload: load real `.npy` measurements from Yiming/Callen.

    Reference: [3] Neural network enabled time-stretch spectral regression.
""")

CODE13_CNN = textwrap.dedent("""\
    # §13a — 1D CNN phase refinement
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        TORCH_OK = True
    except ImportError:
        print('PyTorch not installed.  Run: pip install torch')
        TORCH_OK = False

    if TORCH_OK:
        class PhaseRefineNet(nn.Module):
            \"\"\"
            1D CNN residual network: (B, 3, N) -> (B, 1, N).
            Channels: [I1, I2, phi_gs].  Preserves temporal length N.
            \"\"\"
            def __init__(self, ch=32):
                super().__init__()
                self.enc = nn.Sequential(
                    nn.Conv1d(3,  ch, 7, padding=3), nn.ReLU(),
                    nn.Conv1d(ch, ch, 5, padding=2), nn.ReLU(),
                    nn.Conv1d(ch, ch, 5, padding=2), nn.ReLU(),
                    nn.Conv1d(ch, ch, 3, padding=1), nn.ReLU(),
                )
                self.head = nn.Conv1d(ch, 1, 1)
                self.skip = nn.Conv1d(3,  1, 1)   # residual: passes phi_gs through

            def forward(self, x):
                return self.head(self.enc(x)) + self.skip(x)

        net = PhaseRefineNet()
        print(f'PhaseRefineNet: {sum(p.numel() for p in net.parameters()):,} params')

        # Build training batch (32 QPSK signals at SNR=20 dB)
        def make_batch(n=32, N_t=512, snr=20.0):
            X_list, Y_list = [], []
            for i in range(n):
                d = make_measurements('QPSK', n_symbols=64, sps=8,
                                      D1=-5000., D2=-5750., snr_db=snr, rng_seed=i)
                phi_gs, _ = retrieve_phase(d['I1'], d['I2'], d['D1'], d['D2'],
                                           n_iter=50, unit_amplitude=True)
                nt = min(N_t, len(d['I1']))
                X_list.append([d['I1'][:nt], d['I2'][:nt], phi_gs[:nt]])
                Y_list.append(d['phi_true'][:nt])
            X = torch.tensor(np.array(X_list), dtype=torch.float32)
            Y = torch.tensor(np.array(Y_list)[:, None, :], dtype=torch.float32)
            return X, Y

        print('Building training batch...')
        X_tr, Y_tr = make_batch()

        opt = optim.Adam(net.parameters(), lr=1e-3)
        losses = []
        for ep in range(40):
            net.train(); opt.zero_grad()
            loss = nn.MSELoss()(net(X_tr), Y_tr)
            loss.backward(); opt.step()
            losses.append(float(loss))
            if ep % 10 == 0:
                print(f'  epoch {ep:3d}  loss={float(loss):.5f}')

        # Evaluate on fresh test sample
        net.eval()
        with torch.no_grad():
            d_te = make_measurements('QPSK', n_symbols=64, sps=8,
                                     D1=-5000., D2=-5750., snr_db=20.0, rng_seed=999)
            phi_gs_te, _ = retrieve_phase(d_te['I1'], d_te['I2'],
                                          d_te['D1'], d_te['D2'],
                                          n_iter=50, unit_amplitude=True)
            nt = min(512, len(d_te['I1']))
            X_te = torch.tensor(
                np.stack([d_te['I1'][:nt], d_te['I2'][:nt], phi_gs_te[:nt]])[None],
                dtype=torch.float32)
            phi_cnn_te = net(X_te).squeeze().numpy()

        def rms_deg(phi_hat, phi_ref):
            off = np.angle(np.mean(np.exp(1j*(phi_ref - phi_hat))))
            dlt = np.angle(np.exp(1j*(phi_hat + off - phi_ref)))
            return np.degrees(np.sqrt(np.mean(dlt**2)))

        rgs  = rms_deg(phi_gs_te[:nt],  d_te['phi_true'][:nt])
        rcnn = rms_deg(phi_cnn_te,       d_te['phi_true'][:nt])
        print(f'GS  RMS: {rgs:.2f} deg')
        print(f'CNN RMS: {rcnn:.2f} deg  (improvement: {rgs-rcnn:+.2f} deg)')

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        t_show = np.linspace(0,1,nt)
        axes[0].plot(t_show, d_te['phi_true'][:nt], 'k', lw=1.5, label='True')
        axes[0].plot(t_show, phi_gs_te[:nt],  '--', color='steelblue', lw=1.5, label=f'GS  {rgs:.1f}°')
        axes[0].plot(t_show, phi_cnn_te,       '-.',  color='crimson',   lw=1.5, label=f'CNN {rcnn:.1f}°')
        axes[0].set_title('GS vs CNN phase recovery'); axes[0].legend(fontsize=9)
        axes[0].set_xlabel('Normalized time'); axes[0].set_ylabel('Phase (rad)')

        axes[1].semilogy(losses, 'o-', color='steelblue', markersize=4)
        axes[1].set_title('CNN training loss'); axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MSE loss'); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('notebooks/demo_cnn_compare.png', bbox_inches='tight')
        plt.show()
""")

CODE13_UPLOAD = textwrap.dedent("""\
    # §13b — Real data upload widget
    # Provide .npy files from Yiming (yimingz0416@g.ucla.edu) or Callen (cmacphee@g.ucla.edu)

    import os

    def load_lab_data(I1_path=None, I2_path=None):
        \"\"\"
        Load real lab measurements.

        Format: .npy float arrays, shape (N,) — intensity vs. time sample.

        Parameters
        ----------
        I1_path, I2_path : str or None
            Local file paths.  If None and in Colab, opens upload dialog.
            If None and not in Colab, falls back to synthetic QPSK data.

        Returns
        -------
        dict: I1, I2, N, [phi_true if synthetic]
        \"\"\"
        IN_COLAB = 'google.colab' in __import__('sys').modules

        if I1_path is not None and I2_path is not None:
            I1 = np.load(I1_path).ravel().astype(float)
            I2 = np.load(I2_path).ravel().astype(float)
            print(f'Loaded: I1={I1.shape}, I2={I2.shape}')
            return {'I1': I1, 'I2': I2, 'N': min(len(I1), len(I2))}

        if IN_COLAB:
            from google.colab import files
            import io
            print('Upload I1.npy  (arm 1, dispersion D1):')
            up1 = files.upload()
            print('Upload I2.npy  (arm 2, dispersion D2):')
            up2 = files.upload()
            I1 = np.load(io.BytesIO(list(up1.values())[0])).ravel().astype(float)
            I2 = np.load(io.BytesIO(list(up2.values())[0])).ravel().astype(float)
            return {'I1': I1, 'I2': I2, 'N': min(len(I1), len(I2))}

        # Fallback: synthetic
        print('[DEMO MODE] No file paths provided.  Using synthetic QPSK data.')
        print('To use real data: load_lab_data(\"I1.npy\", \"I2.npy\")')
        d = make_measurements('QPSK', n_symbols=128, snr_db=25.0, rng_seed=77)
        return {'I1': d['I1'], 'I2': d['I2'], 'N': len(d['I1']),
                'phi_true': d['phi_true'], '_synthetic': True}


    # ─── Run on lab data (swap in real file paths once data is provided) ──────
    # lab = load_lab_data('data/I1_arm1.npy', 'data/I2_arm2.npy')
    lab = load_lab_data()

    D1_lab, D2_lab = -5000.0, -5750.0   # physical: -695/-800 ps/nm  (paper values)

    phi_lab, errs_lab = retrieve_phase(
        lab['I1'], lab['I2'], D1_lab, D2_lab,
        n_iter=50, unit_amplitude=True,
    )

    N_show = min(300, lab['N'])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(lab['I1'][:N_show], lw=1.2, label='$I_1$ (D1)')
    axes[0].plot(lab['I2'][:N_show], lw=1.2, label='$I_2$ (D2)', alpha=0.8)
    title_tag = ' [SYNTHETIC DEMO]' if lab.get('_synthetic') else ' [LAB DATA]'
    axes[0].set_title('Measurement intensities' + title_tag)
    axes[0].set_xlabel('Sample index'); axes[0].legend()

    axes[1].plot(phi_lab[:N_show], lw=1.5, color='steelblue', label='GS recovered')
    if 'phi_true' in lab:
        off = np.angle(np.mean(np.exp(1j*(lab['phi_true'][:N_show]-phi_lab[:N_show]))))
        axes[1].plot(lab['phi_true'][:N_show], '--', color='crimson',
                     alpha=0.7, label='True (synthetic)')
        axes[1].legend()
    axes[1].set_title('Recovered phase'); axes[1].set_xlabel('Sample'); axes[1].set_ylabel('rad')

    plt.suptitle('Lab data pipeline (GS phase retrieval)', fontweight='bold')
    plt.tight_layout()
    plt.savefig('notebooks/demo_lab_pipeline.png', bbox_inches='tight')
    plt.show()
    print('Next: email yimingz0416@g.ucla.edu / cmacphee@g.ucla.edu for real I1.npy, I2.npy')
""")

new_cells = [
    md(MD12), code(CODE12_BUILD), code(CODE12_PLOT),
    md(MD13), code(CODE13_CNN),   code(CODE13_UPLOAD),
]

nb['cells'].extend(new_cells)

with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print(f'Done. Notebook now has {len(nb["cells"])} cells.')
print('New sections: §12 (3D pipe) and §13 (CNN + upload widget).')
