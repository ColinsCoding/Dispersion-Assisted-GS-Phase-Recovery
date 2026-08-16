function results = part6_kinetics_fit(k_true, C0, noise_std, seed)
%PART6_KINETICS_FIT  First-order decay kinetics C(t)=C0*exp(-k*t), fit from noisy optical data.
%
%   TEXTBOOK PHYSICS/CHEMISTRY (public domain: first-order reaction kinetics, in every
%   introductory chemistry/physics textbook). C(t) = C0*exp(-k*t) is the concentration of a
%   species undergoing a first-order process (radioactive decay, dye photobleaching,
%   dilution -- the equation doesn't care which). OUR EDUCATIONAL SIMULATION: a hypothetical
%   laser-based optical measurement whose detected intensity is proportional to
%   concentration, I(t) = alpha*C(t) + noise (a generic Beer-Lambert-style assumption, not
%   any specific instrument's calibration).
%
%   THREE WAYS TO GET k, in order (per the assignment): (1) analytic -- linearize
%   ln(I) = ln(alpha*C0) - k*t and read k off the slope directly (no optimizer needed, exact
%   for noiseless data); (2) MATLAB numerical optimization (lsqcurvefit) on the noisy data
%   directly, no linearization; (3) PyTorch autograd equivalent lives in the companion
%   notebook (Part 12) on this SAME synthetic dataset, so the classical and autograd fits can
%   be compared on identical data.
%
%   results = PART6_KINETICS_FIT(k_true, C0, noise_std, seed) with defaults k_true=0.35,
%   C0=1.0, noise_std=0.03, seed=1.

    if nargin < 1 || isempty(k_true), k_true = 0.35; end
    if nargin < 2 || isempty(C0), C0 = 1.0; end
    if nargin < 3 || isempty(noise_std), noise_std = 0.03; end
    if nargin < 4 || isempty(seed), seed = 1; end
    rng(seed);

    alpha = 2.4;   % hypothetical detector responsivity (arbitrary units / concentration unit)
    t = linspace(0, 10, 40)';
    I_clean = alpha * C0 * exp(-k_true * t);
    I_noisy = I_clean + noise_std * randn(size(t));
    I_noisy = max(I_noisy, 1e-6);   % detector cannot report negative intensity

    fprintf('=== synthetic optical measurement of first-order decay ===\n');
    fprintf('true k = %.4f, true C0 = %.4f, alpha (detector responsivity) = %.4f\n', k_true, C0, alpha);

    % ---- (1) analytic solution: linearize ln(I) = ln(alpha*C0) - k*t ----
    p = polyfit(t, log(I_noisy), 1);
    k_analytic = -p(1);
    alphaC0_analytic = exp(p(2));

    fprintf('\n(1) analytic (linearized log-fit):\n');
    fprintf('    k_analytic = %.4f  (true %.4f, error %.2f%%)\n', k_analytic, k_true, 100 * abs(k_analytic - k_true) / k_true);

    % ---- (2) MATLAB nonlinear optimization directly on the noisy (non-log) data ----
    model = @(p, tt) p(1) * exp(-p(2) * tt);   % p = [alphaC0, k]
    p0 = [I_noisy(1), 0.1];
    opts = optimoptions('lsqcurvefit', 'Display', 'off');
    p_fit = lsqcurvefit(model, p0, t, I_noisy, [0, 0], [Inf, Inf], opts);
    alphaC0_fit = p_fit(1);
    k_fit = p_fit(2);

    fprintf('\n(2) MATLAB lsqcurvefit (nonlinear least squares on raw data):\n');
    fprintf('    k_fit = %.4f  (true %.4f, error %.2f%%)\n', k_fit, k_true, 100 * abs(k_fit - k_true) / k_true);

    if abs(k_analytic - k_true) / k_true > 0.15
        error('part6_kinetics_fit:analytic', 'analytic log-linear fit is too far from the true k');
    end
    if abs(k_fit - k_true) / k_true > 0.15
        error('part6_kinetics_fit:lsqcurvefit', 'lsqcurvefit is too far from the true k');
    end

    % ---- export the synthetic dataset so the PyTorch notebook (Part 12) can fit the SAME data ----
    out_dir = fileparts(mfilename('fullpath'));
    writematrix([t, I_noisy], fullfile(out_dir, 'part6_kinetics_data.csv'));

    fig = figure('Visible', 'off');
    plot(t, I_noisy, 'ko', 'DisplayName', 'noisy detector samples'); hold on;
    plot(t, I_clean, 'b-', 'LineWidth', 1.5, 'DisplayName', 'true I(t)');
    plot(t, model(p_fit, t), 'r--', 'LineWidth', 1.5, 'DisplayName', 'lsqcurvefit fit');
    xlabel('t'); ylabel('detector intensity I(t)');
    title(sprintf('first-order decay fit: k_{true}=%.3f, k_{fit}=%.3f', k_true, k_fit));
    legend show; grid on;
    saveas(fig, fullfile(out_dir, 'part6_kinetics_fit.png'));
    close(fig);

    results.t = t;
    results.I_noisy = I_noisy;
    results.I_clean = I_clean;
    results.k_true = k_true;
    results.k_analytic = k_analytic;
    results.k_fit = k_fit;
    results.alphaC0_analytic = alphaC0_analytic;
    results.alphaC0_fit = alphaC0_fit;

    fprintf('\npart6_kinetics_fit: all checks passed. Data exported to part6_kinetics_data.csv, figure saved.\n');
end
