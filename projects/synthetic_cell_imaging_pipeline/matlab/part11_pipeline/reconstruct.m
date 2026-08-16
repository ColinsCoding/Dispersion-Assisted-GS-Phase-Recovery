function x_hat = reconstruct(y_calibrated, H, lambda)
%RECONSTRUCT  Stage 4: Tikhonov-regularized inversion, x_hat = (H'H + lambda*I)^-1 H'y.
%
%   Uses a FIXED, pre-calibrated lambda (a production pipeline does not re-run Part 5's
%   full lambda sweep on every acquisition -- it uses a value already validated offline,
%   here taken from Part 5's own best_lambda for this exact H).
%
%   x_hat = RECONSTRUCT(y_calibrated, H, lambda) with default lambda = 3.16e-3.

    if nargin < 3 || isempty(lambda), lambda = 3.16e-3; end
    N = round(sqrt(size(H, 1)));
    if N^2 ~= size(H, 1)
        error('reconstruct:shape', 'H must be square with a perfect-square dimension (N^2 x N^2)');
    end

    HtH = H' * H;
    Hty = H' * y_calibrated(:);
    x_hat = reshape((HtH + lambda * eye(size(H, 2))) \ Hty, N, N);
    fprintf('[reconstruct] lambda=%.3e -> x_hat %dx%d\n', lambda, N, N);
end
