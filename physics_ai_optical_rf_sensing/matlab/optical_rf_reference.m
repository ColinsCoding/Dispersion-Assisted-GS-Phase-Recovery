% optical_rf_reference.m
% Reference calculation of the basic physics features used in the Python
% project, exported as CSV for cross-checking. SYNTHETIC / deterministic only.
%
% Run from the project root (or anywhere -- paths are relative to this file):
%   matlab -batch "run('matlab/optical_rf_reference.m')"
% Output: results/matlab_reference_features.csv

here = fileparts(mfilename('fullpath'));
root = fileparts(here);
outdir = fullfile(root, 'results');
if ~exist(outdir, 'dir'); mkdir(outdir); end

h = 6.62607015e-34; c = 299792458; q = 1.602176634e-19;

% 1. photon energy E = h c / lambda
lam_nm = [405 532 633 850 1310 1550];
E_eV = h * c ./ (lam_nm * 1e-9) / q;

% 2. Beer-Lambert I = I0 exp(-alpha L), alpha = 120 1/m
I0 = 1.0; alpha = 120; L_m = [0 0.005 0.01 0.02 0.05 0.1];
I_trans = I0 * exp(-alpha * L_m);

% 3. series RLC against 50 ohm: resonance, Q, |Gamma| minimum, return loss
R = 45; f0 = 2.0e9; Q = 30;
L = Q * R / (2*pi*f0); C = 1 / ((2*pi*f0)^2 * L);
f = linspace(1e9, 3e9, 4001);
Z = R + 1i*(2*pi*f*L - 1./(2*pi*f*C));
G = (Z - 50) ./ (Z + 50);
[gmin, k] = min(abs(G));
rl_db = -20*log10(gmin);
f_res = 1/(2*pi*sqrt(L*C));
Q_calc = sqrt(L/C) / R;

% 4. power P = V^2 / R with first-order uncertainty
V = 5.0; sV = 0.05; Rload = 100; sR = 1.0;
P = V^2 / Rload; sP = P * sqrt((2*sV/V)^2 + (sR/Rload)^2);

names  = [compose("photon_energy_eV_%dnm", lam_nm), compose("beer_lambert_I_L%gm", L_m), ...
          "rlc_resonance_hz", "rlc_Q", "rlc_min_abs_gamma", "rlc_f_at_min_hz", "rlc_return_loss_db", ...
          "power_W", "power_sigma_W"];
values = [E_eV, I_trans, f_res, Q_calc, gmin, f(k), rl_db, P, sP];
T = table(names', values', 'VariableNames', {'quantity', 'value'});
outfile = fullfile(outdir, 'matlab_reference_features.csv');
writetable(T, outfile);
fprintf('wrote %s (%d quantities)\n', outfile, height(T));
fprintf('RLC: f0 = %.6g Hz, Q = %.4g, min|Gamma| = %.4g, RL = %.3f dB; P = %.4g +- %.3g W\n', ...
        f_res, Q_calc, gmin, rl_db, P, sP);
