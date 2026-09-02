"""Generate the reference image + real phycv.PST output that
hardware/phycv_pst_cuda.cu cross-checks its own CUDA/cuFFT implementation
against. Run this once before building/running that file:

    py -3.13 scripts/generate_pst_cuda_reference.py
    (cd hardware, then build+run phycv_pst_cuda.cu as documented in its header)
"""
import pathlib
import numpy as np
from phycv import PST

N = 128
img = np.full((N, N), 50.0)
img[30:98, 30:98] = 200.0   # a bright square on a dim background -- sharp edges, the natural PST target

S, W = 0.5, 15.0
sigma_LPF = 0.1

p = PST(h=N, w=N)
p.load_img(img_array=img)
p.init_kernel(S, W)
p.apply_kernel(sigma_LPF, 0.1, 0.9, morph_flag=False)
feature = p.pst_feature

out_dir = pathlib.Path(__file__).resolve().parents[1] / "hardware"
np.savetxt(out_dir / "pst_ref_image.txt", img, fmt="%.10f")
np.savetxt(out_dir / "pst_ref_feature.txt", feature, fmt="%.10f")
print(f"N={N} S={S} W={W} sigma_LPF={sigma_LPF}")
print(f"feature range: [{feature.min():.4f}, {feature.max():.4f}]")
print(f"wrote {out_dir / 'pst_ref_image.txt'} and {out_dir / 'pst_ref_feature.txt'}")
