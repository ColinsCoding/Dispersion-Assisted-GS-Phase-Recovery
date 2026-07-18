"""Generate the research notebook notebooks/01_gaussian_beam_pipeline.ipynb.

The notebook follows the required structure: theory -> equations -> SymPy derivation ->
numerical verification -> plots -> feature extraction -> Torch integration ->
generated C -> conclusions.

Usage:
    python scripts/build_notebook.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md = nbf.v4.new_markdown_cell
co = nbf.v4.new_code_cell


def build() -> Path:
    """Write the notebook and return its path."""
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# Gaussian-beam pipeline: physics -> SymPy -> features -> ML -> C\n\n"
            "A research notebook exercising the full `photonics_ml_pipeline` stack: derive the beam\n"
            "width symbolically, verify it numerically, extract features, train a PyTorch classifier,\n"
            "and generate portable C from the finalized equation."
        ),
        co(
            "import sys, pathlib\n"
            "sys.path.insert(0, str(pathlib.Path.cwd().parent))\n"
            "import numpy as np, torch, sympy as sp\n"
            "import matplotlib.pyplot as plt\n"
            "from physics.symbolic import gaussian_beam_width\n"
            "from physics.gaussian_beam import GaussianBeam\n"
            "from feature_extraction.features import extract_features\n"
            "from ml.dataset import BeamFeatureDataset\n"
            "from ml.train import train_model\n"
            "from ml.inference import predict, confusion_matrix\n"
            "from c_codegen.generator import generate_c\n"
            "sp.init_printing()\n"
            "print('ready')"
        ),
        md("## 1-3. Theory, equations, SymPy derivation\n"
           "$w(z)=w_0\\sqrt{1+(z/z_R)^2}$, $z_R=\\pi w_0^2/\\lambda$. SymPy differentiates it for us."),
        co(
            "bw = gaussian_beam_width()\n"
            "grad = bw.gradient()\n"
            "print('w(z) =', bw.expr)\n"
            "print('dw/dz =', sp.simplify(grad[0]))"
        ),
        md("## 4-5. Numerical verification and plot"),
        co(
            "beam = GaussianBeam(1.55, 10.0)\n"
            "z = np.linspace(-2000, 2000, 256)\n"
            "w = beam.width_um(z)\n"
            "f = bw.lambdify('numpy')\n"
            "assert np.allclose(w, f(z, beam.waist_um, beam.rayleigh_range_um))\n"
            "plt.figure(figsize=(6,3)); plt.plot(z, w); plt.plot(z, -w)\n"
            "plt.xlabel('z (um)'); plt.ylabel('w(z) (um)'); plt.title('beam width'); plt.tight_layout(); plt.show()"
        ),
        md("## 6-7. Feature extraction and Torch classifier"),
        co(
            "ds = BeamFeatureDataset(n_samples=600, seed=0)\n"
            "res = train_model(ds, input_dim=ds.feature_dim, n_classes=ds.n_classes, epochs=40, seed=0)\n"
            "print('validation accuracy =', round(res.val_accuracy, 3))\n"
            "X = torch.stack([ds[i][0] for i in range(len(ds))])\n"
            "y = torch.stack([ds[i][1] for i in range(len(ds))])\n"
            "cm = confusion_matrix(y, predict(res.model, X), ds.n_classes)\n"
            "print('confusion matrix:\\n', cm)"
        ),
        md("## 8. Generated C from the finalized equation"),
        co("print(generate_c(bw).source)"),
        md("## 9. Conclusions\n"
           "One symbolic model flows end to end: SymPy derives and verifies it, physics generates\n"
           "features, PyTorch classifies beams with high accuracy, and portable C is emitted for\n"
           "embedded deployment -- no GPU required."),
    ]
    nb.metadata.kernelspec = {"name": "py312", "display_name": "Python 3.12 (torch)", "language": "python"}
    out = ROOT / "notebooks" / "01_gaussian_beam_pipeline.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(out))
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    build()
