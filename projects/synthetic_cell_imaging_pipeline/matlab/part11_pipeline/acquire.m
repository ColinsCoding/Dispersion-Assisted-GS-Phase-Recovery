function [y_raw, gain_map, ground_truth] = acquire(N, sigma_blur, noise_std, seed)
%ACQUIRE  Stage 1 of the data pipeline: simulate one raw sensor acquisition.
%
%   Calls the Part-4 forward model (y = H*x + n) and additionally applies a synthetic,
%   FIXED per-pixel sensitivity variation (a "gain map", +/-5%, the kind every real sensor
%   has and every real pipeline calibrates out in the next stage) -- so CALIBRATE has
%   something real to correct, not a no-op step.
%
%   [y_raw, gain_map, ground_truth] = ACQUIRE(N, sigma_blur, noise_std, seed).

    parent_dir = fileparts(fileparts(mfilename('fullpath')));
    if isempty(which('part4_forward_model'))
        addpath(parent_dir);
    end

    if nargin < 1 || isempty(N), N = 24; end
    if nargin < 2 || isempty(sigma_blur), sigma_blur = 1.5; end
    if nargin < 3 || isempty(noise_std), noise_std = 0.02; end
    if nargin < 4 || isempty(seed), seed = 0; end

    fwd = part4_forward_model(N, sigma_blur, noise_std, seed);

    rng(seed + 100);
    gain_map = 1 + 0.05 * randn(N, N);   % +/-5% fixed pixel-to-pixel sensitivity variation
    y_raw = fwd.y .* gain_map;

    ground_truth.x_true = fwd.x_true;
    ground_truth.H = fwd.H;
    ground_truth.y_uncalibrated = fwd.y;
    ground_truth.noise_std = noise_std;

    fprintf('[acquire] N=%d, sigma_blur=%.2f, noise_std=%.3f -> y_raw %dx%d acquired\n', ...
        N, sigma_blur, noise_std, size(y_raw, 1), size(y_raw, 2));
end
