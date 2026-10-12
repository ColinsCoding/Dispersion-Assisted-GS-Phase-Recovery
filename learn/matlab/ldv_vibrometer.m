% ldv_vibrometer.m -- laser Doppler vibrometer in MATLAB: Bessel harmonics,
% Bessel-null calibration, and heterodyne CIC demodulation of v(t).
%
% Same formulas as dgs/laser_doppler_vibrometer.py and hardware/cuda_ldv/ldv_demod.cu:
%   detector  cos(phi0 + b sin(W t)),  b = 4 pi a / lambda   -> harmonics 2 |J_n(b)| (|cos| or |sin| phi0)
%   null      J0(b) = 0  ->  a = j_{0,m} lambda / (4 pi)
%   LDV       s = cos(2 pi fB t + 4 pi x(t)/lambda), x = vp (1 - cos wt)/w
%             mix to baseband, 3 cascaded length-5 boxcars (filter), unwrap(angle), central difference
%
% Run headless from the repo root:
%     matlab -batch "run('learn/matlab/ldv_vibrometer.m')"
% Writes learn/matlab/ldv_matlab.csv (metric, value) for the notebook to compare.

lam = 632.8e-9;
out = {};

% ---- 1. Bessel harmonics: formula vs FFT -----------------------------------
b = 1.8; Ns = 4096; th = 2*pi*(0:Ns-1)'/Ns; worst = 0;
for phi0 = [0, pi/2, pi/4]
    S = abs(fft(cos(phi0 + b*sin(th)))) / Ns;
    S = S(1:9); S(2:end) = 2*S(2:end);
    n = (0:8)';
    f = abs(besselj(n, b)) .* (mod(n,2)==0) * abs(cos(phi0)) + abs(besselj(n, b)) .* (mod(n,2)==1) * abs(sin(phi0));
    f(2:end) = 2*f(2:end);
    worst = max(worst, max(abs(S - f)));
end
fprintf('harmonics: max |FFT - Bessel formula| = %.2e\n', worst);
out(end+1,:) = {'harmonics_max_error', worst};

% ---- 2. Bessel-null calibration amplitudes ---------------------------------
guesses = [2.4 5.5 8.65];
for m = 1:3
    z = fzero(@(q) besselj(0, q), guesses(m));
    fprintf('J0 zero %d: b = %.6f  ->  a = %.2f nm\n', m, z, z*lam/(4*pi)*1e9);
    out(end+1,:) = {sprintf('null_amplitude_nm_%d', m), z*lam/(4*pi)*1e9};
end

% ---- 3. heterodyne LDV with a CIC demodulator ------------------------------
fs = 200e6; fB = 40e6; fv = 5e3; vp = 0.5; N = 400000;
t = (0:N-1)'/fs; w = 2*pi*fv;
x = vp*(1 - cos(w*t))/w;
s0 = cos(2*pi*fB*t + 4*pi*x/lam);
vtrue = vp*sin(w*t);
rng(0);
for snr_db = [Inf 30]
    s = s0;
    if isfinite(snr_db), s = s + sqrt(0.5)*10^(-snr_db/20)*randn(N,1); end
    z = s .* exp(-1i*2*pi*fB*t);
    for k = 1:3, z = filter(ones(1,5)/5, 1, z); end      % causal, same as np.convolve[:N]
    ph = unwrap(angle(z));
    v = nan(N,1);
    v(2:end-1) = lam/(4*pi) * (ph(3:end) - ph(1:end-2)) * fs/2;
    v(1:13) = NaN;                                       % filter not yet filled
    e = v(7:end) - vtrue(1:end-6);                       % group delay 6 samples
    r = sqrt(mean(e(~isnan(e)).^2));
    fprintf('CIC demod, SNR %s dB: RMS velocity error = %.4e m/s\n', num2str(snr_db), r);
    out(end+1,:) = {sprintf('cic_rms_snr_%s', num2str(snr_db)), r};
end

here = fileparts(mfilename('fullpath'));
writecell([{'metric','value'}; out], fullfile(here, 'ldv_matlab.csv'));
