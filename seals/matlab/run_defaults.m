% Non-interactive copy of the lab's main.m default branch: writes arrays to CSV.
dia = 9940e-9; npar = 1.39; nmed = 1; r = .1;
d = 9.0909e-7; D = .065; a = .9023; dcorr = -4.2448e-4; P = .0058; NA = .7; mangle = 20;
lam1 = 1580e-9; lam2 = 1600e-9; lambda = (lam1 + lam2)/2;
lamvec = linspace(lam1, lam2, 500);
[y, theta] = SEALS(d, D, a, dcorr, P, NA, lamvec);
theta = theta + mangle;
[cs, I_p, I_s, an, bn, T_p, T_s] = mie(npar, nmed, dia, lambda, deg2rad(theta), r);
I_rd = rayleighdebye(dia, lambda, nmed, npar, deg2rad(theta), r);
writematrix([lamvec' y' theta' I_p' I_s' T_p' T_s' I_rd'], 'matlab_defaults.csv');
fprintf('MATLAB_CS %.12e\n', cs);
fprintf('MATLAB_NMAX %d\n', numel(an));
