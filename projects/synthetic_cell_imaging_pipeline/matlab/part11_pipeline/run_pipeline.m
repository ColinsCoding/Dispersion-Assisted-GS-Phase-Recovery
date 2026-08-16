function results = run_pipeline(N, sigma_blur, noise_std, seed)
%RUN_PIPELINE  Part 11: a clean MATLAB data pipeline -- acquire -> preprocess -> calibrate ->
%   reconstruct -> classify -> store checkpoint, using FUNCTIONS (this file, and each stage's
%   own file), not one giant script.
%
%   TEXTBOOK SOFTWARE ENGINEERING (public domain: a standard multi-stage data pipeline
%   pattern, not tied to any specific product). Each stage is independently testable and
%   independently callable; this function is thin orchestration only -- it contains no
%   physics or reconstruction math of its own, all of that lives in acquire.m/preprocess.m/
%   calibrate.m/reconstruct.m/classify.m/store_checkpoint.m.
%
%   results = RUN_PIPELINE(N, sigma_blur, noise_std, seed) with defaults N=24, sigma_blur=1.5,
%   noise_std=0.02, seed=0.

    if nargin < 1 || isempty(N), N = 24; end
    if nargin < 2 || isempty(sigma_blur), sigma_blur = 1.5; end
    if nargin < 3 || isempty(noise_std), noise_std = 0.02; end
    if nargin < 4 || isempty(seed), seed = 0; end

    here = fileparts(mfilename('fullpath'));
    if isempty(which('acquire')), addpath(here); end

    fprintf('=== run_pipeline: acquire -> preprocess -> calibrate -> reconstruct -> classify -> store ===\n');

    [y_raw, gain_map, ground_truth] = acquire(N, sigma_blur, noise_std, seed);
    y_processed = preprocess(y_raw);
    y_calibrated = calibrate(y_processed, gain_map);

    % CHECK: calibration should recover something close to the uncalibrated-but-clean measurement
    calib_recovery_error = norm(y_calibrated(:) - ground_truth.y_uncalibrated(:)) / norm(ground_truth.y_uncalibrated(:));
    fprintf('calibration recovery relative error: %.4f\n', calib_recovery_error);
    if calib_recovery_error > 0.05
        error('run_pipeline:calibration', 'calibrated measurement should closely match the uncalibrated-but-clean signal');
    end

    x_hat = reconstruct(y_calibrated, ground_truth.H);
    reconstruction_error = norm(x_hat(:) - ground_truth.x_true(:)) / norm(ground_truth.x_true(:));
    fprintf('reconstruction relative error: %.4f\n', reconstruction_error);

    decision = classify(x_hat, y_calibrated, noise_std);
    if ~decision.accept
        error('run_pipeline:expected_pass', ...
            'the regularized reconstruction should PASS classify() (contrast with Part 8''s raw-blur FAIL case)');
    end

    checkpoint_data.x_hat = x_hat;
    checkpoint_data.decision = decision;
    checkpoint_data.reconstruction_error = reconstruction_error;
    checkpoint_data.calib_recovery_error = calib_recovery_error;
    checkpoint_data.params = struct('N', N, 'sigma_blur', sigma_blur, 'noise_std', noise_std, 'seed', seed);

    checkpoint_path = fullfile(here, 'checkpoint.mat');
    store_checkpoint(checkpoint_path, checkpoint_data);

    % CHECK: the checkpoint actually round-trips
    loaded = load(checkpoint_path);
    if ~isequal(loaded.data.decision.label, decision.label)
        error('run_pipeline:checkpoint_roundtrip', 'loaded checkpoint does not match what was saved');
    end

    results = checkpoint_data;
    results.y_raw = y_raw;
    results.y_calibrated = y_calibrated;
    results.x_true = ground_truth.x_true;

    fprintf('\nrun_pipeline: all checks passed. decision=%s, reconstruction error=%.4f\n', ...
        decision.label, reconstruction_error);
end
