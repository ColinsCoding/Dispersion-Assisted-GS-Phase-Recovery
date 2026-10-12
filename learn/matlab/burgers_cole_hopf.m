% burgers_cole_hopf.m -- viscous Burgers u_t + u u_x = nu u_xx on [0,1), periodic,
% u0 = sin(2 pi x). Finite-difference RK4 solution vs the exact Cole-Hopf solution.
%
% Same scheme and exact solution as dgs/lagrangian_fluids.py, so the max errors
% printed here should match the Python ones (about 1.3e-4 for nu = 0.01, N = 512).
%
% Cole-Hopf: u = -2 nu phi_x / phi turns Burgers into the homogeneous heat
% equation phi_t = nu phi_xx, which each Fourier mode solves exactly:
%     phi_k(t) = phi_k(0) exp(-nu k^2 t).
%
% Run headless from the repo root:
%     matlab -batch "run('learn/matlab/burgers_cole_hopf.m')"
% Writes learn/matlab/burgers_matlab.csv (x, u_fd, u_exact) for nu = 0.01.

nus = [0.05 0.02 0.01];
N = 512; t_end = 0.5;
dx = 1 / N; x = (0:N-1)' * dx;

for nu = nus
    % ---- finite differences + RK4 (vectorised; circshift = periodic roll)
    dt = 0.4 * min(dx, dx^2 / (2 * nu));
    steps = ceil(t_end / dt); dt = t_end / steps;
    rhs = @(u) -(circshift(u, -1).^2 - circshift(u, 1).^2) / (4 * dx) ...
               + nu * (circshift(u, -1) - 2 * u + circshift(u, 1)) / dx^2;
    u = sin(2 * pi * x);
    for s = 1:steps
        k1 = rhs(u); k2 = rhs(u + dt/2 * k1); k3 = rhs(u + dt/2 * k2); k4 = rhs(u + dt * k3);
        u = u + dt / 6 * (k1 + 2*k2 + 2*k3 + k4);
    end

    % ---- exact Cole-Hopf on a fine grid, sampled at the FD points
    Nf = 4096; xf = (0:Nf-1)' / Nf;
    phi0 = exp(-(1 - cos(2 * pi * xf)) / (4 * pi * nu));
    k = 2 * pi * [0:Nf/2-1, -Nf/2:-1]';
    P = fft(phi0) .* exp(-nu * k.^2 * t_end);
    phi = real(ifft(P)); phix = real(ifft(1i * k .* P));
    uf = -2 * nu * phix ./ phi;
    ue = uf(1:Nf/N:end);                 % FD grid is a subset of the fine grid

    fprintf('nu = %.3f  steps = %6d  max |u_fd - u_exact| = %.3e  max |u_x| = %.2f\n', ...
            nu, steps, max(abs(u - ue)), max(abs(diff(ue))) / dx);
    if nu == 0.01
        here = fileparts(mfilename('fullpath'));
        writematrix([x u ue], fullfile(here, 'burgers_matlab.csv'));
    end
end
