function results = part4_forward_model(N, sigma_blur, noise_std, seed)
%PART4_FORWARD_MODEL  A synthetic-cell computational-imaging forward model: y = H*x + n.
%
%   OUR EDUCATIONAL SIMULATION (original, not real medical data, not a reproduction of any
%   specific instrument or patented implementation). x is a synthetic "cell" image (a
%   membrane ring + off-center nucleus + a few organelle dots -- shapes only, no biological
%   measurement behind them). H models the optical system's blur response (finite-aperture/
%   diffraction-limited optics blur every real image by SOME point-spread function; a
%   Gaussian PSF is the standard first-order textbook model). n is detector noise.
%
%   TEXTBOOK PHYSICS: any linear, shift-invariant optical system's action on an object is a
%   CONVOLUTION with its point-spread function; writing the object and image as vectors
%   turns that convolution into a matrix-vector product y = H*x -- this is just linear
%   algebra applied to the convolution theorem, not a new physical assumption.
%
%   results = PART4_FORWARD_MODEL(N, sigma_blur, noise_std, seed) with defaults N=24,
%   sigma_blur=1.5 (px), noise_std=0.02, seed=0. Returns x_true, H, n_noise, y, and a
%   quick pinv-based reconstruction (the deeper rank/SVD/conditioning story is Part 5).

    if nargin < 1 || isempty(N), N = 24; end
    if nargin < 2 || isempty(sigma_blur), sigma_blur = 1.5; end
    if nargin < 3 || isempty(noise_std), noise_std = 0.02; end
    if nargin < 4 || isempty(seed), seed = 0; end
    rng(seed);

    x_true = build_cell_object(N);
    H = build_blur_matrix(N, sigma_blur);

    x_vec = x_true(:);
    y_clean_vec = H * x_vec;
    y_clean = reshape(y_clean_vec, N, N);

    % cross-check the matrix-vector blur against an independent conv2-based circular
    % convolution -- CHECKED to agree, not assumed just because both "look like a blur"
    kernel1d = gaussian_kernel_1d(N, sigma_blur);
    kernel2d = kernel1d(:) * kernel1d(:)';
    y_conv = local_circular_conv2(x_true, kernel2d);
    conv_vs_matrix_error = max(abs(y_clean(:) - y_conv(:)));
    fprintf('matrix-blur vs. independent conv2-blur agreement, max abs diff: %.3e\n', conv_vs_matrix_error);
    if conv_vs_matrix_error > 1e-8
        error('part4_forward_model:blur_mismatch', 'matrix-vector blur does not match direct convolution');
    end

    n_noise = noise_std * randn(N, N);
    y = y_clean + n_noise;

    % quick (unregularized) reconstruction just to complete the visualization triptych --
    % Part 5 explains WHY this looks the way it does and how to do better
    x_hat_naive = reshape(pinv(H) * y(:), N, N);

    fig = figure('Visible', 'off');
    subplot(1, 3, 1); imagesc(x_true); axis image off; colormap(gca, 'gray'); title('true object x');
    subplot(1, 3, 2); imagesc(y); axis image off; colormap(gca, 'gray'); title('measured y = Hx + n');
    subplot(1, 3, 3); imagesc(x_hat_naive); axis image off; colormap(gca, 'gray'); title('naive reconstruction (pinv)');
    out_dir = fileparts(mfilename('fullpath'));
    saveas(fig, fullfile(out_dir, 'part4_forward_model.png'));
    close(fig);

    results.x_true = x_true;
    results.H = H;
    results.n_noise = n_noise;
    results.y = y;
    results.x_hat_naive = x_hat_naive;
    results.conv_vs_matrix_error = conv_vs_matrix_error;
    results.N = N;

    fprintf('part4_forward_model: N=%d (H is %dx%d), all checks passed. Figure saved to part4_forward_model.png\n', ...
        N, N^2, N^2);
end

function img = build_cell_object(N)
%BUILD_CELL_OBJECT  A synthetic "cell": a membrane ring, an off-center nucleus, and organelle
%   dots. Purely geometric shapes on an NxN grid, values in [0,1] -- not real imaging data.
    [X, Y] = meshgrid(linspace(-1, 1, N), linspace(-1, 1, N));
    cx = 0; cy = 0;
    R = sqrt((X - cx).^2 + (Y - cy).^2);

    membrane = double(R > 0.75 & R < 0.9);                       % thin ring near the boundary
    nucleus = 0.9 * exp(-((X - 0.15).^2 + (Y + 0.1).^2) / (2 * 0.18^2));  % off-center bright blob
    organelle1 = 0.5 * exp(-((X + 0.35).^2 + (Y - 0.3).^2) / (2 * 0.05^2));
    organelle2 = 0.5 * exp(-((X + 0.2).^2 + (Y + 0.45).^2) / (2 * 0.04^2));

    img = membrane + nucleus + organelle1 + organelle2;
    img = img / max(img(:));
end

function H = build_blur_matrix(N, sigma_blur)
%BUILD_BLUR_MATRIX  Exact NxN separable circular-Gaussian blur matrix (N^2 x N^2) acting on
%   a column-major-vectorized NxN image: y(:) = H * x(:). Built as kron(H1D, H1D) from a 1D
%   circulant Gaussian blur matrix, the standard exact construction for a separable 2D
%   convolution's dense matrix form.
    h1 = gaussian_kernel_1d(N, sigma_blur);
    H1 = zeros(N, N);
    for row = 1:N
        H1(row, :) = circshift(h1, row - 1 - floor(N/2));
    end
    H = kron(H1, H1);
end

function h = gaussian_kernel_1d(N, sigma_blur)
%GAUSSIAN_KERNEL_1D  A length-N, zero-centered (via circshift elsewhere), unit-sum Gaussian.
    n = (0:N-1) - floor(N/2);
    h = exp(-(n.^2) / (2 * sigma_blur^2));
    h = h / sum(h);
end

function y = local_circular_conv2(x, kernel2d)
%LOCAL_CIRCULAR_CONV2  2D circular convolution via FFT -- an independent implementation path
%   (not matrix multiplication) used only to cross-check build_blur_matrix's dense H.
    N = size(x, 1);
    kernel_centered = circshift(kernel2d, [ceil(N/2), ceil(N/2)]);   % move kernel center to (1,1) index convention
    y = real(ifft2(fft2(x) .* fft2(kernel_centered)));
end
