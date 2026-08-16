function results = part5_matrix_analysis(N, sigma_blur, noise_std, seed)
%PART5_MATRIX_ANALYSIS  rank(H), svd(H), cond(H): what they mean for image recoverability.
%
%   TEXTBOOK MATH (public domain: every linear-algebra textbook's treatment of ill-posed
%   inverse problems). H maps every possible object x to a measurement Hx. Its SINGULAR
%   VALUES sigma_i measure how much each of H's orthogonal input directions (right singular
%   vectors) gets amplified on the way to the output: a direction with a TINY sigma_i is
%   almost invisible in the measurement -- any noise added AFTER measurement gets divided by
%   that same tiny sigma_i when you try to invert, so recovering that direction of the object
%   requires amplifying noise by 1/sigma_i. rank(H) counts how many directions have any
%   nonzero response at all; cond(H) = sigma_max/sigma_min is the worst-case noise
%   amplification factor for a full inversion. A physically blurring H (like this project's
%   Gaussian PSF) always has a WIDE spread of singular values -- low spatial frequencies pass
%   through nearly unchanged (large sigma), high spatial frequencies are almost killed (tiny
%   sigma) -- which is exactly why naive inversion (Part 4's pinv reconstruction) explodes
%   into high-frequency checkerboard noise: it is dividing noise by those near-zero sigma_i.
%
%   Tikhonov regularization x_hat = (H'H + lambda*I)^-1 H'y trades a little bias (blur left
%   in the answer) for a lot less noise amplification, by not dividing by near-zero
%   singular values at full strength -- CHECKED below to beat the naive pinv reconstruction
%   by a real, measured error reduction, not just asserted to be better.
%
%   results = PART5_MATRIX_ANALYSIS(N, sigma_blur, noise_std, seed) reuses part4_forward_model
%   for the same H, x_true, y (defaults N=24, sigma_blur=1.5, noise_std=0.02, seed=0).

    if nargin < 1 || isempty(N), N = 24; end
    if nargin < 2 || isempty(sigma_blur), sigma_blur = 1.5; end
    if nargin < 3 || isempty(noise_std), noise_std = 0.02; end
    if nargin < 4 || isempty(seed), seed = 0; end

    fwd = part4_forward_model(N, sigma_blur, noise_std, seed);
    H = fwd.H; x_true = fwd.x_true; y = fwd.y;

    % ---- rank, singular values, condition number ----
    r = rank(H);
    s = svd(H);
    condH = cond(H);

    fprintf('\n=== matrix analysis of H (%dx%d) ===\n', size(H, 1), size(H, 2));
    fprintf('rank(H) = %d  (out of %d possible)\n', r, size(H, 1));
    fprintf('largest singular value  sigma_max = %.4e\n', s(1));
    fprintf('smallest singular value sigma_min = %.4e\n', s(end));
    fprintf('cond(H) = sigma_max/sigma_min = %.4e\n', condH);
    fprintf('  (a condition number this large means some object directions need noise\n');
    fprintf('   amplified by a factor of ~%.1e to invert exactly -- exactly what pinv did.)\n', condH);

    if abs(condH - s(1)/s(end)) / condH > 1e-6
        error('part5_matrix_analysis:cond_check', 'cond(H) does not match sigma_max/sigma_min');
    end

    % ---- naive (pinv) vs Tikhonov-regularized reconstruction ----
    x_hat_naive = fwd.x_hat_naive;
    naive_error = norm(x_hat_naive(:) - x_true(:));

    lambdas = logspace(-6, 1, 25);
    errs = zeros(size(lambdas));
    HtH = H' * H;
    Hty = H' * y(:);
    I = eye(size(H, 2));
    for i = 1:numel(lambdas)
        x_hat_lambda = (HtH + lambdas(i) * I) \ Hty;
        errs(i) = norm(x_hat_lambda - x_true(:));
    end
    [best_err, best_idx] = min(errs);
    best_lambda = lambdas(best_idx);
    x_hat_reg = reshape((HtH + best_lambda * I) \ Hty, N, N);

    fprintf('\n=== naive (pinv) vs Tikhonov-regularized reconstruction ===\n');
    fprintf('naive (pinv) reconstruction error  ||x_hat - x_true|| = %.4f\n', naive_error);
    fprintf('best Tikhonov lambda = %.4e, error = %.4f\n', best_lambda, best_err);
    fprintf('regularization improves error by a factor of %.2fx\n', naive_error / best_err);

    if best_err >= naive_error
        error('part5_matrix_analysis:regularization', ...
            'Tikhonov regularization at its best lambda should beat the unregularized pinv reconstruction');
    end

    fig = figure('Visible', 'off');
    subplot(1, 3, 1); semilogy(s, 'LineWidth', 1.5);
    xlabel('index i'); ylabel('\sigma_i'); title(sprintf('singular values of H (cond=%.2e)', condH)); grid on;
    subplot(1, 3, 2); loglog(lambdas, errs, 'LineWidth', 1.5); hold on;
    plot(best_lambda, best_err, 'ro', 'MarkerFaceColor', 'r');
    xlabel('\lambda'); ylabel('||x_{hat} - x_{true}||'); title('Tikhonov error vs. \lambda'); grid on;
    subplot(1, 3, 3); imagesc(x_hat_reg); axis image off; colormap(gca, 'gray');
    title(sprintf('regularized reconstruction (\\lambda=%.2e)', best_lambda));
    out_dir = fileparts(mfilename('fullpath'));
    saveas(fig, fullfile(out_dir, 'part5_matrix_analysis.png'));
    close(fig);

    results.rank_H = r;
    results.singular_values = s;
    results.cond_H = condH;
    results.naive_error = naive_error;
    results.best_lambda = best_lambda;
    results.best_reg_error = best_err;
    results.x_hat_reg = x_hat_reg;
    results.lambdas = lambdas;
    results.errs = errs;

    fprintf('\npart5_matrix_analysis: all checks passed. Figure saved to part5_matrix_analysis.png\n');
end
