% thomson_vs_rutherford.m -- why the plum-pudding atom failed (Knight 37.6).
% MATLAB twin of dgs/thomson_vs_rutherford.py; same assumptions.
% Run headless:  matlab -batch "run('learn/matlab/thomson_vs_rutherford.m')"

ke2   = 1.439964;      % e^2/(4 pi eps0), MeV*fm
fm    = 1e-15;
Z     = 79;  R = 1.44e-10;  n = 5.90e28;   % gold: atomic radius (m), atoms per m^3
q     = 2;   K = 7.7;       mratio = 7294.3; % alpha: charge (e), KE (MeV), m_alpha/m_e
t     = 4.0e-7;                              % foil thickness (m), assumed

% Thomson: largest deflection by the smeared positive sphere, kqQ/(2KR)
theta_sphere   = q*Z*ke2 / (2*K*R/fm);
% ... and by one electron at rest
theta_electron = asin(1/mratio);
theta_1 = max(theta_sphere, theta_electron);        % generous: max kick every time

N         = t * n^(1/3);                            % atoms crossed
theta_rms = sqrt(N) * theta_1;                      % 2-D random walk
log10P    = -(pi/2/theta_rms)^2 / log(10);          % P(theta > 90 deg)

% Rutherford: one close encounter with a point nucleus
d    = q*Z*ke2/K*fm;
b90  = 0.5*d/tan(pi/4);
P_R  = pi*b90^2*n*t;

fprintf('Thomson sphere max   %.3e rad\n', theta_sphere);
fprintf('electron max         %.3e rad\n', theta_electron);
fprintf('collisions N         %.0f\n', N);
fprintf('theta_rms            %.4f deg\n', rad2deg(theta_rms));
fprintf('Thomson   P(>90 deg) = 10^(%.0f)\n', log10P);
fprintf('Rutherford P(>90 deg) = %.2e\n', P_R);

assert(abs(theta_sphere - 1.026e-4) < 1e-6);
assert(log10P < -1000 && P_R > 1e-6);
disp('MATLAB checks passed');
