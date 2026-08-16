function results = part3_even_odd_symmetry()
%PART3_EVEN_ODD_SYMMETRY  Even/odd decomposition and its effect on the Fourier transform.
%
%   TEXTBOOK MATH (public domain: every signal-processing/Fourier-analysis textbook). ANY
%   function f(x) splits uniquely into an even part f_e(x) = [f(x)+f(-x)]/2 (f_e(-x)=f_e(x))
%   and an odd part f_o(x) = [f(x)-f(-x)]/2 (f_o(-x)=-f_o(x)), f = f_e + f_o. The DFT/FFT of a
%   REAL signal inherits this structure directly: the real part of the spectrum comes from the
%   even part of the signal, the imaginary part from the odd part (this is exactly why a real
%   even signal has a real spectrum, and a real odd signal has a purely imaginary spectrum --
%   checked numerically below, not just asserted).
%
%   OPTICAL-MODE CONNECTION: the same decomposition classifies waveguide/cavity modes. A mode
%   symmetric about the guide's center axis (even) and one antisymmetric about it (odd) behave
%   differently under a symmetric perturbation (e.g. uniform heating): even modes couple to
%   symmetric perturbations, odd modes are protected by their own antisymmetry (their overlap
%   integral with any even perturbation function vanishes by parity alone) -- the same
%   even-index-vs-odd-index bookkeeping used throughout Fourier/waveguide analysis.
%
%   results = PART3_EVEN_ODD_SYMMETRY() returns a struct with the decomposition and FFT checks.

    x = linspace(-10, 10, 4001);
    dx = x(2) - x(1);

    % an arbitrary, NOT purely even or odd, signal: a shifted Gaussian bump plus a shifted sine ramp
    f = exp(-(x - 1.3).^2 / 2) + 0.4 * sin(0.7 * x + 0.9);

    % ---- even/odd decomposition (interpolate f(-x) onto the same grid) ----
    f_flipped = interp1(x, f, -x, 'linear', 'extrap');
    f_even = 0.5 * (f + f_flipped);
    f_odd  = 0.5 * (f - f_flipped);

    recon_error = max(abs((f_even + f_odd) - f));
    fprintf('=== even/odd decomposition ===\n');
    fprintf('max|f_even + f_odd - f| (should be ~0): %.3e\n', recon_error);

    % verify the defining symmetry properties directly (not just the reconstruction)
    f_even_flipped = interp1(x, f_even, -x, 'linear', 'extrap');
    f_odd_flipped  = interp1(x, f_odd, -x, 'linear', 'extrap');
    even_symmetry_error = max(abs(f_even_flipped - f_even));
    odd_symmetry_error  = max(abs(f_odd_flipped + f_odd));
    fprintf('max|f_even(-x) - f_even(x)| (should be ~0): %.3e\n', even_symmetry_error);
    fprintf('max|f_odd(-x) + f_odd(x)|  (should be ~0): %.3e\n', odd_symmetry_error);

    tol = 1e-2;   % interpolation-grid tolerance, not machine precision (finite dx, edge effects)
    if recon_error > tol || even_symmetry_error > tol || odd_symmetry_error > tol
        error('part3_even_odd_symmetry:decomposition', 'even/odd decomposition failed its own defining checks');
    end

    % ---- FFT of a purely even and a purely odd signal (built directly, not interpolated) ----
    % N chosen ODD so the sample grid is exactly symmetric about x=0 (an even-length grid
    % is off-center by half a sample, which breaks exact odd symmetry at the sample level
    % even for an analytically odd function -- verified: this was the first thing tried,
    % and it left a ~0.5% residual purely from that indexing asymmetry, not from the math).
    N = 513;
    half = (N - 1) / 2;
    xs = (-half:half) * dx;
    g_even = exp(-xs.^2 / 4);                 % Gaussian: exactly even
    g_odd  = xs .* exp(-xs.^2 / 4);           % x * Gaussian: exactly odd

    G_even = fftshift(fft(ifftshift(g_even)));
    G_odd  = fftshift(fft(ifftshift(g_odd)));

    max_imag_over_real_even = max(abs(imag(G_even))) / max(abs(real(G_even)));
    max_real_over_imag_odd  = max(abs(real(G_odd))) / max(abs(imag(G_odd)));

    fprintf('\n=== FFT symmetry: even real signal -> real spectrum, odd real signal -> imaginary spectrum ===\n');
    fprintf('max|Im(FFT(even))| / max|Re(FFT(even))| (should be small): %.3e\n', max_imag_over_real_even);
    fprintf('max|Re(FFT(odd))| / max|Im(FFT(odd))|  (should be small): %.3e\n', max_real_over_imag_odd);

    if max_imag_over_real_even > 1e-8 || max_real_over_imag_odd > 1e-8
        error('part3_even_odd_symmetry:fft_parity', 'FFT did not preserve the expected even/odd -> real/imaginary correspondence');
    end

    fig = figure('Visible', 'off');
    subplot(2, 2, 1); plot(x, f, x, f_even, x, f_odd);
    legend('f', 'f_{even}', 'f_{odd}'); title('signal decomposition'); grid on;
    subplot(2, 2, 2); plot(xs, g_even, xs, g_odd);
    legend('g_{even}', 'g_{odd}'); title('pure even/odd test signals'); grid on;
    subplot(2, 2, 3); plot(real(G_even), 'DisplayName', 'Re'); hold on; plot(imag(G_even), 'DisplayName', 'Im');
    legend show; title('FFT(even signal): Im \approx 0'); grid on;
    subplot(2, 2, 4); plot(real(G_odd), 'DisplayName', 'Re'); hold on; plot(imag(G_odd), 'DisplayName', 'Im');
    legend show; title('FFT(odd signal): Re \approx 0'); grid on;
    out_dir = fileparts(mfilename('fullpath'));
    saveas(fig, fullfile(out_dir, 'part3_even_odd_fft.png'));
    close(fig);

    results.f = f; results.f_even = f_even; results.f_odd = f_odd;
    results.recon_error = recon_error;
    results.even_symmetry_error = even_symmetry_error;
    results.odd_symmetry_error = odd_symmetry_error;
    results.max_imag_over_real_even = max_imag_over_real_even;
    results.max_real_over_imag_odd = max_real_over_imag_odd;

    fprintf('\npart3_even_odd_symmetry: all checks passed. Figure saved to part3_even_odd_fft.png\n');
end
