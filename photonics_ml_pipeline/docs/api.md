# API reference

Auto-generated from module docstrings.

### `physics.gaussian_beam`

Numeric Gaussian-beam model.

- **`GaussianBeam(wavelength_um: 'float', waist_um: 'float') -> None`** -- A paraxial Gaussian beam defined by wavelength and waist radius (in um).

### `physics.symbolic`

Symbolic engine (SymPy).

- **`SymbolicExpression(expr: 'sp.Expr', symbols: 'tuple[sp.Symbol, ...]', name: 'str' = 'f') -> None`** -- A named SymPy scalar expression together with its ordered free symbols.
- **`rayleigh_range() -> 'SymbolicExpression'`** -- z_R = pi * w0**2 / lambda.
- **`gaussian_beam_width() -> 'SymbolicExpression'`** -- w(z) = w0 * sqrt(1 + (z / z_R)**2).
- **`gouy_phase() -> 'SymbolicExpression'`** -- psi(z) = atan(z / z_R).

### `optics.abcd`

ABCD (ray-transfer) matrix optics for Gaussian beams.

- **`free_space(distance_um: 'float') -> 'np.ndarray'`** -- ABCD matrix for propagation over `distance_um`.
- **`thin_lens(focal_length_um: 'float') -> 'np.ndarray'`** -- ABCD matrix for a thin lens of focal length `focal_length_um`.
- **`propagate_q(q: 'complex', matrix: 'np.ndarray') -> 'complex'`** -- Apply an ABCD matrix to the complex beam parameter q.
- **`q_at_waist(rayleigh_range_um: 'float') -> 'complex'`** -- Complex beam parameter at the waist: q = i z_R.
- **`width_from_q(q: 'complex', wavelength_um: 'float') -> 'float'`** -- Recover the beam radius w from the complex beam parameter q.

### `photonics.dispersion`

Group-velocity dispersion as an all-pass spectral filter.

- **`transfer_function(freq: 'np.ndarray', dispersion: 'float') -> 'np.ndarray'`** -- All-pass dispersion transfer function H_D(f) = exp(i pi D f^2).
- **`apply_dispersion(pulse: 'np.ndarray', dispersion: 'float') -> 'np.ndarray'`** -- Apply second-order dispersion to a complex time-domain pulse.
- **`group_delay(freq: 'np.ndarray', dispersion: 'float') -> 'np.ndarray'`** -- Group delay tau(f) = D f imposed by the dispersion.

### `feature_extraction.features`

Deterministic feature extraction from 1-D optical intensity profiles / images.

- **`FeatureVector(names: 'tuple[str, ...]', values: 'np.ndarray') -> None`** -- A named, ordered feature vector.
- **`extract_features(field: 'np.ndarray') -> 'FeatureVector'`** -- Compute the 8-element feature vector for an intensity profile / image.
- **`feature_names() -> 'tuple[str, ...]'`** -- Ordered names of the extracted features.

### `ml.dataset`

PyTorch dataset built from physics-derived features.

- **`BeamFeatureDataset(n_samples: 'int' = 900, wavelength_um: 'float' = 1.55, z_um: 'float' = 300.0, half_width_um: 'float' = 60.0, n_transverse: 'int' = 256, seed: 'int' = 0) -> 'None'`** -- Standardized (feature, label) pairs from simulated Gaussian beams.

### `ml.inference`

Inference and evaluation helpers.

- **`predict(model: 'nn.Module', features: 'torch.Tensor') -> 'torch.Tensor'`** -- Return predicted class indices for a batch of feature vectors.
- **`confusion_matrix(y_true: 'torch.Tensor', y_pred: 'torch.Tensor', n_classes: 'int') -> 'np.ndarray'`** -- Row = true class, column = predicted class.

### `ml.model`

Feed-forward classifier consuming physics-derived features.

- **`FeatureMLP(input_dim: 'int', hidden_dims: 'Sequence[int]', n_classes: 'int') -> 'None'`** -- Multilayer perceptron for feature-vector classification.

### `ml.train`

Training loop for the feature classifier.

- **`TrainResult(model: 'FeatureMLP', history: 'dict[str, list[float]]' = <factory>, val_accuracy: 'float' = 0.0) -> None`** -- Outcome of a training run.
- **`train_model(dataset: 'Dataset', input_dim: 'int', n_classes: 'int', hidden_dims: 'tuple[int, ...]' = (32, 16), epochs: 'int' = 60, learning_rate: 'float' = 0.001, batch_size: 'int' = 32, val_fraction: 'float' = 0.25, seed: 'int' = 0) -> 'TrainResult'`** -- Train a `FeatureMLP` and return the model with its metric history.

### `c_codegen.generator`

Theory-to-code generator: SymPy expression -> optimized C / Fortran / JS.

- **`CodegenResult(name: 'str', language: 'str', source: 'str', header: 'str', example: 'str') -> None`** -- A generated translation unit: source, header, and a runnable example.
- **`generate_c(sym: 'SymbolicExpression') -> 'CodegenResult'`** -- Generate an optimized C function (with CSE) plus header and example `main`.
- **`generate_fortran(sym: 'SymbolicExpression') -> 'str'`** -- Generate a Fortran function body via SymPy `fcode`.
- **`generate_js(sym: 'SymbolicExpression') -> 'str'`** -- Generate a JavaScript expression via SymPy `jscode`.
- **`write_c(result: 'CodegenResult', src_dir: 'str | Path', include_dir: 'str | Path') -> 'dict[str, Path]'`** -- Write the C source/header/example to disk; return the created paths.
- **`compile_and_run_c(sym: 'SymbolicExpression', args: 'Sequence[float]', cc: 'str | None' = None) -> 'float'`** -- Compile the generated C for `sym`, run it with `args`, and return the result.

### `visualization.plots`

Matplotlib figure builders for every pipeline stage.

- **`plot_optical_field(z_um: 'np.ndarray', width_um: 'np.ndarray', title: 'str' = 'Gaussian beam width') -> 'plt.Figure'`** -- Plot beam width w(z) versus axial position.
- **`plot_training_history(history: 'dict[str, list[float]]') -> 'plt.Figure'`** -- Plot training loss and validation accuracy over epochs.
- **`plot_confusion_matrix(matrix: 'np.ndarray', class_names: 'list[str] | None' = None) -> 'plt.Figure'`** -- Render a confusion matrix as an annotated heatmap.
- **`plot_feature_importance(names: 'list[str]', importances: 'np.ndarray') -> 'plt.Figure'`** -- Horizontal bar chart of feature importances.
- **`plot_timing(labels: 'list[str]', times_s: 'list[float]') -> 'plt.Figure'`** -- Bar chart of per-backend timings on a log scale.
