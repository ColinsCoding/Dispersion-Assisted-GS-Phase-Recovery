"""Morphology feature extraction: the stage between phase reconstruction and
classification in the QPI pipeline

  illumination -> microscope -> camera -> phase reconstruction ->
  [AI segmentation] -> [THIS MODULE: morphology] -> cell count / classifier

dgs/gs_core.py, dgs/steam_imaging.py, and dgs/fourier_ptychography.py already
cover phase reconstruction (labeled-free, from intensity-only measurements --
the QPI principle: measure optical phase, not fluorescence). "AI segmentation"
(going from a continuous phase image to a binary cell mask) is its OWN stage,
not built here -- this module takes a mask as already-given input, plus one
classical (non-learned) Otsu threshold helper so the module is testable
end-to-end without depending on the unbuilt learned-segmentation stage.

Feature set matches dgs/sbir_portfolio.py's P5 (Bayesian CTC detection)
proposal, which already names {I_max, phi_mean, phi_std, morphology_entropy}
as its classifier features -- this module is where those get computed for
real from a recovered phase image and a cell mask, plus standard geometric
shape descriptors (area, eccentricity, circularity) used throughout
quantitative cell biology (CellProfiler, scikit-image regionprops).
"""
import numpy as np


# ── Classical (non-learned) thresholding, to produce a mask for testing ───────

def threshold_otsu(image, n_bins=256):
    """Otsu's method (Otsu, 1979): the threshold that maximizes between-
    class variance of a 2-class (foreground/background) split of the
    image's intensity histogram. Standard, well-established, NOT the
    learned "AI segmentation" stage referenced above -- a classical
    baseline used here only so this module can be tested end-to-end."""
    img = np.asarray(image, float)
    if img.size == 0:
        raise ValueError("image must be non-empty")
    hist, bin_edges = np.histogram(img, bins=n_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    weight_bg = np.cumsum(hist)
    weight_fg = weight_bg[-1] - weight_bg
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_bg = np.cumsum(hist * bin_centers) / weight_bg
        mean_fg = (np.cumsum((hist * bin_centers)[::-1])[::-1]) / weight_fg
    variance_between = weight_bg[:-1] * weight_fg[:-1] * (mean_bg[:-1] - mean_fg[:-1]) ** 2
    variance_between = np.nan_to_num(variance_between, nan=0.0)
    if not np.any(variance_between > 0):
        return float(bin_centers[len(bin_centers) // 2])
    idx = int(np.argmax(variance_between))
    return float(bin_centers[idx])


def segment_mask(image, threshold=None, above=True):
    """Binary mask from a threshold -- Otsu's if none given. above=True
    keeps pixels ABOVE threshold as foreground (e.g. a phase bump); set
    above=False for a phase dip."""
    img = np.asarray(image, float)
    if threshold is None:
        threshold = threshold_otsu(img)
    return (img > threshold) if above else (img < threshold)


# ── Image moments and ellipse-fit shape descriptors ───────────────────────────

def image_moments(mask):
    """Raw and central 2nd-order image moments of a binary mask (standard
    image-moments formalism). Returns centroid and the central-moment
    covariance-matrix entries used by ellipse_axes_from_moments."""
    m = np.asarray(mask, bool)
    if not np.any(m):
        raise ValueError("mask must contain at least one foreground pixel")
    yy, xx = np.mgrid[0:m.shape[0], 0:m.shape[1]]
    M00 = float(np.sum(m))
    xbar = float(np.sum(xx[m])) / M00
    ybar = float(np.sum(yy[m])) / M00
    dx = xx[m] - xbar
    dy = yy[m] - ybar
    mu20 = float(np.sum(dx * dx)) / M00
    mu02 = float(np.sum(dy * dy)) / M00
    mu11 = float(np.sum(dx * dy)) / M00
    return {"M00": M00, "centroid": (ybar, xbar), "mu20": mu20, "mu02": mu02, "mu11": mu11}


def ellipse_axes_from_moments(mask):
    """Major/minor axis lengths of the ellipse with the same 2nd-order
    moments as the mask (the standard "equivalent ellipse" -- same
    convention as skimage.measure.regionprops: axis_length = 4*sqrt(eigenvalue)
    of the central-moment covariance matrix)."""
    mom = image_moments(mask)
    cov = np.array([[mom["mu20"], mom["mu11"]], [mom["mu11"], mom["mu02"]]])
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    lam_major, lam_minor = float(eigvals[1]), float(eigvals[0])
    return 4.0 * np.sqrt(lam_major), 4.0 * np.sqrt(lam_minor)


def eccentricity(mask):
    """Ellipse eccentricity e = sqrt(1 - (minor/major)^2): 0 for a
    perfect circle, approaching 1 for an elongated shape. Standard formula
    (same convention as skimage regionprops.eccentricity)."""
    major, minor = ellipse_axes_from_moments(mask)
    if major <= 0:
        return 0.0
    return float(np.sqrt(max(0.0, 1.0 - (minor / major) ** 2)))


# ── Area, perimeter, circularity ───────────────────────────────────────────────

def area_px(mask):
    """Pixel count of the foreground region."""
    return int(np.sum(np.asarray(mask, bool)))


def perimeter_px(mask):
    """Discrete boundary length: counts every foreground/background edge
    (4-connectivity) around the mask. KNOWN SIMPLIFICATION stated honestly:
    this "crack length" estimator does not apply the sqrt(2) diagonal-step
    correction some libraries (e.g. skimage) use, so it systematically
    OVER-estimates perimeter (and thus UNDER-estimates circularity below)
    for diagonally-oriented boundaries relative to a smoothed contour --
    acceptable for a relative shape-comparison feature, not a calibrated
    absolute perimeter measurement."""
    m = np.asarray(mask, bool)
    if not np.any(m):
        raise ValueError("mask must contain at least one foreground pixel")
    padded = np.pad(m, 1, mode='constant', constant_values=False)
    edges = 0
    edges += np.sum(padded[1:-1, 1:-1] & ~padded[:-2, 1:-1])   # up
    edges += np.sum(padded[1:-1, 1:-1] & ~padded[2:, 1:-1])    # down
    edges += np.sum(padded[1:-1, 1:-1] & ~padded[1:-1, :-2])   # left
    edges += np.sum(padded[1:-1, 1:-1] & ~padded[1:-1, 2:])    # right
    return int(edges)


def circularity(mask):
    """4*pi*Area/Perimeter^2 -- 1.0 for a perfect circle, smaller for
    irregular/elongated shapes. Standard shape descriptor; inherits
    perimeter_px's discretization bias (see its docstring)."""
    area = area_px(mask)
    perim = perimeter_px(mask)
    if perim == 0:
        raise ValueError("perimeter is zero -- degenerate mask")
    return float(4.0 * np.pi * area / perim ** 2)


# ── Intensity / phase statistics + entropy ────────────────────────────────────

def shannon_entropy(values, n_bins=32):
    """Shannon entropy H = -sum(p*log(p)) of the histogram of `values`
    (standard information-theoretic texture/heterogeneity measure -- higher
    entropy = more heterogeneous phase distribution within the region).
    This is dgs/sbir_portfolio.py P5's 'morphology_entropy' feature."""
    v = np.asarray(values, float).ravel()
    if v.size == 0:
        raise ValueError("values must be non-empty")
    hist, _ = np.histogram(v, bins=n_bins)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def extract_cell_features(phase_image, mask):
    """Full feature vector for one segmented cell, matching
    dgs/sbir_portfolio.py P5's named feature set {I_max, phi_mean, phi_std,
    morphology_entropy} plus standard geometric shape descriptors.

    Parameters
    ----------
    phase_image : float array -- recovered phase phi(x,y), e.g. from
                  dgs.gs_core.retrieve_phase or dgs.fourier_ptychography's
                  reconstructed object's np.angle()
    mask         : bool array, same shape -- the segmented cell region

    Returns
    -------
    dict of feature name -> float
    """
    phi = np.asarray(phase_image, float)
    m = np.asarray(mask, bool)
    if phi.shape != m.shape:
        raise ValueError("phase_image and mask must have the same shape")
    if not np.any(m):
        raise ValueError("mask must contain at least one foreground pixel")

    phi_in = phi[m]
    return {
        "area_px": area_px(m),
        "perimeter_px": perimeter_px(m),
        "eccentricity": eccentricity(m),
        "circularity": circularity(m),
        "I_max": float(np.max(np.abs(phi_in))),
        "phi_mean": float(np.mean(phi_in)),
        "phi_std": float(np.std(phi_in)),
        "morphology_entropy": shannon_entropy(phi_in),
    }


if __name__ == "__main__":
    print("=== Cell morphology features from a recovered phase image ===\n")

    N = 96
    yy, xx = np.mgrid[0:N, 0:N]

    # Synthetic elongated "cell": Gaussian phase bump, stretched along x
    cy, cx = N // 2, N // 2
    a, b = 22.0, 12.0   # semi-major, semi-minor (pixels)
    phi = 0.6 * np.exp(-(((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2))
    phi += 0.02 * np.random.default_rng(0).standard_normal((N, N))   # measurement noise

    mask = segment_mask(phi, threshold=0.2, above=True)
    feats = extract_cell_features(phi, mask)

    print(f"Segmented area: {feats['area_px']} px  (analytic ellipse ~ {np.pi*a*b:.0f} px for full extent)")
    e_analytic = np.sqrt(1.0 - (b / a) ** 2)
    print(f"Eccentricity: {feats['eccentricity']:.3f}  (analytic ellipse b/a=12/22 -> e={e_analytic:.3f})")
    print(f"Circularity:  {feats['circularity']:.3f}  (1.0 = perfect circle; elongated shape -> <1)")
    print(f"I_max={feats['I_max']:.3f}  phi_mean={feats['phi_mean']:.3f}  "
          f"phi_std={feats['phi_std']:.3f}  morphology_entropy={feats['morphology_entropy']:.3f}")
    print("\nThese four (I_max, phi_mean, phi_std, morphology_entropy) are exactly")
    print("dgs/sbir_portfolio.py P5's named Bayesian CTC classifier features.")
