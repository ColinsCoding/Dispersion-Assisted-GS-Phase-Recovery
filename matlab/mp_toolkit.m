function R = mp_toolkit(out_json)
%MP_TOOLKIT  Modern physics + phase retrieval in MATLAB, mirroring the Python toolkit.
%
%   R = mp_toolkit()            returns a struct of results
%   mp_toolkit('out.json')      also writes them as JSON (for the Python cross-check)
%
% Mirrors dgs/one_d_quantum.py (units: nm, eV, electron masses) and dgs/gs_core.py
% (dispersion H = exp(i*pi*D*nu^2), two-measurement Gerchberg-Saxton). Run headless:
%   matlab -batch "addpath('matlab'); mp_toolkit('out.json')"

C_NM = 0.0380998211383624;        % hbar^2 / (2 m_e) in eV nm^2 (same value as dgs/one_d_quantum.py)

% --- particle in a box: E_n = n^2 pi^2 C / L^2 ---------------------------------------------
L = 0.5;
R.box_E = arrayfun(@(n) box_energy(n, L, C_NM), 1:4);

% --- finite-difference Schrodinger solver: finite well -------------------------------------
x = linspace(-1.0, 1.5, 801);
U = 5.0 * ~(x > 0 & x < 0.5);
[E, psi] = solve_schrodinger(x, U, C_NM, 3);
R.fd_E = E(:)';
dx = x(2) - x(1);
R.fd_norm = sum(abs(psi(:, 1)).^2) * dx;

% --- tunneling estimate exp(-2w/eta) --------------------------------------------------------
eta = sqrt(C_NM / (5.0 - 3.0));
R.eta = eta;
R.tunnel = exp(-2 * 0.4 / eta);

% --- dispersion and GS ----------------------------------------------------------------------
N = 256; D1 = -5000; D2 = -5750;
k = 0:N-1;
phi_true = pi/2 * mod(floor(k / 8) * 7 + 3, 4);          % deterministic QPSK-like symbols
E_true = exp(1i * phi_true);
I1 = abs(disperse(E_true, D1)).^2;
I2 = abs(disperse(E_true, D2)).^2;
R.energy_ratio = sum(abs(disperse(E_true, D1)).^2) / sum(abs(E_true).^2);   % Parseval: 1
R.roundtrip_err = max(abs(undisperse(disperse(E_true, D1), D1) - E_true));
Eg = undisperse(complex(sqrt(I1)), D1);
errs = zeros(1, 30);
for it = 1:30
    Eg = gs_iteration(Eg, I1, I2, D1, D2);
    errs(it) = sqrt(mean((abs(disperse(Eg, D2)).^2 - I2).^2));
end
R.gs_errors = errs;

if nargin > 0
    fid = fopen(out_json, 'w');
    fprintf(fid, '%s', jsonencode(R));
    fclose(fid);
end
end

% ============================================================================================
function E = box_energy(n, L, C)
E = n^2 * pi^2 * C / L^2;
end

function [E, psi] = solve_schrodinger(x, U, C, n_states)
% -C psi'' + U psi = E psi, with psi = 0 just outside the grid (tridiagonal matrix).
% Full eig (not eigs): eigs' 'smallestreal' can fail to converge and return NaN.
N = numel(x); dx = x(2) - x(1); t = C / dx^2;
H = diag(U(:) + 2 * t) - t * diag(ones(N - 1, 1), 1) - t * diag(ones(N - 1, 1), -1);
[V, Dg] = eig(H);
[E, idx] = sort(diag(Dg));
E = E(1:n_states);
psi = V(:, idx(1:n_states)) / sqrt(dx);
end

function nu = fftfreq(N)
nu = [0:ceil(N/2)-1, -floor(N/2):-1] / N;
end

function Ed = disperse(E, D)
Ed = ifft(fft(E) .* exp(1i * pi * D * fftfreq(numel(E)).^2));
end

function E = undisperse(Ed, D)
E = ifft(fft(Ed) .* exp(-1i * pi * D * fftfreq(numel(Ed)).^2));
end

function E = gs_iteration(E, I1, I2, D1, D2)
% project onto each measured intensity, then onto |E| = 1 (as gs_core, unit_amplitude=True)
E = undisperse(sqrt(I1) .* exp(1i * angle(disperse(E, D1))), D1);
E = exp(1i * angle(E));
E = undisperse(sqrt(I2) .* exp(1i * angle(disperse(E, D2))), D2);
E = exp(1i * angle(E));
end
