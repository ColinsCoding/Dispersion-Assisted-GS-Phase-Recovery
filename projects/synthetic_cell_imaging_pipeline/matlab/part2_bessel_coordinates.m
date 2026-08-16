function results = part2_bessel_coordinates()
%PART2_BESSEL_COORDINATES  Why cylindrical symmetry produces Bessel's equation.
%
%   TEXTBOOK PHYSICS/MATH (public domain: separation of variables, a technique in every PDE
%   textbook). Coordinate choice does not change the physics, only how tractable the algebra
%   is. In CARTESIAN coordinates, the Helmholtz equation nabla^2(psi) + k^2*psi = 0 separates
%   into three ordinary (sinusoidal) equations, one per axis. In CYLINDRICAL coordinates
%   (rho, phi, z), the SAME equation for a field with no z- or phi-dependence reduces to
%
%       rho^2 * R''(rho) + rho * R'(rho) + (k^2*rho^2) * R(rho) = 0,
%
%   Bessel's equation of order 0, because the radial Laplacian itself picks up a 1/rho * d/drho
%   term (the "curvature" of circles) that Cartesian coordinates never see. That extra term is
%   NOT an approximation or an added assumption -- it falls straight out of expanding
%   nabla^2 in cylindrical coordinates. Bessel functions J_0, J_1, ... are simply "what a
%   sinusoid looks like once you insist it also respect circular symmetry."
%
%   This file (1) derives the radial ODE symbolically from the cylindrical Laplacian, (2)
%   verifies J_0(k*rho) solves it (residual ~0), and (3) plots J_0, J_1 and their zero
%   crossings.
%
%   results = PART2_BESSEL_COORDINATES() returns a struct with the symbolic ODE, the
%   verification residual, and the numeric Bessel data plotted.

    syms rho k R(rho)
    assume(rho, 'positive')
    assume(k, 'positive')

    % ---- derive the radial ODE from the cylindrical Laplacian of a rho-only field ----
    % nabla^2(psi) in cylindrical coords, psi=psi(rho) only (no phi, z dependence):
    %   nabla^2(psi) = (1/rho) d/drho( rho * dpsi/drho )
    psi = R(rho);
    lap_cyl = (1/rho) * diff(rho * diff(psi, rho), rho);
    helmholtz_eq = lap_cyl + k^2 * psi == 0;

    fprintf('=== Helmholtz equation for a rho-only cylindrical field ===\n');
    fprintf('(1/rho) d/drho(rho dR/drho) + k^2 R = 0\n');
    disp(expand(lap_cyl + k^2*psi));

    % expand and compare against Bessel's equation of order 0:
    %   rho^2 R'' + rho R' + k^2 rho^2 R = 0
    lhs_expanded = expand(rho^2 * lap_cyl);
    bessel_form = rho^2 * diff(psi, rho, 2) + rho * diff(psi, rho) + k^2 * rho^2 * psi;
    match = simplify(lhs_expanded - (bessel_form - k^2*rho^2*psi)) == 0;   % same ODE up to the k^2 term already isolated
    fprintf('multiplying through by rho^2 matches Bessel''s-equation form: %d\n', logical(match));

    if ~match
        error('part2_bessel_coordinates:derivation', 'radial ODE does not match Bessel''s equation form');
    end

    % ---- verify J_0(k*rho) actually solves the ODE (symbolic residual) ----
    syms rho_s positive
    J0_expr = besselj(0, k * rho_s);
    residual = simplify(rho_s^2 * diff(J0_expr, rho_s, 2) + rho_s * diff(J0_expr, rho_s) + k^2 * rho_s^2 * J0_expr);
    fprintf('\nresidual of J_0(k*rho) in Bessel''s equation (expect 0): %s\n', char(residual));

    if residual ~= 0
        error('part2_bessel_coordinates:verify_J0', 'J_0(k*rho) does not satisfy the derived Bessel equation');
    end

    % ---- numeric J_0, J_1 and their zeros ----
    rho_num = linspace(0, 20, 2000);
    J0_num = besselj(0, rho_num);
    J1_num = besselj(1, rho_num);

    % first 4 zeros of J0 and J1 (built-in MATLAB besselzero is not available in base;
    % find them by sign change + fzero, a standard robust root-finding approach)
    zeros_J0 = local_find_bessel_zeros(0, 4);
    zeros_J1 = local_find_bessel_zeros(1, 4);
    fprintf('\nfirst 4 zeros of J_0: %s\n', mat2str(zeros_J0, 5));
    fprintf('first 4 zeros of J_1: %s\n', mat2str(zeros_J1, 5));

    % known reference values (Abramowitz & Stegun, public-domain tables) for a sanity check
    known_J0_zeros = [2.4048, 5.5201, 8.6537, 11.7915];
    if max(abs(zeros_J0 - known_J0_zeros)) > 1e-3
        error('part2_bessel_coordinates:zeros_J0', 'computed J_0 zeros do not match known reference values');
    end

    fig = figure('Visible', 'off');
    plot(rho_num, J0_num, 'LineWidth', 1.5); hold on;
    plot(rho_num, J1_num, 'LineWidth', 1.5);
    plot(zeros_J0, zeros(size(zeros_J0)), 'ko', 'MarkerFaceColor', 'k');
    plot(zeros_J1, zeros(size(zeros_J1)), 'ks', 'MarkerFaceColor', 'w');
    yline(0, 'k:');
    xlabel('k\rho'); ylabel('amplitude');
    title('Bessel functions J_0(k\rho), J_1(k\rho) -- the radial mode shapes of cylindrical symmetry');
    legend('J_0', 'J_1', 'zeros of J_0', 'zeros of J_1', 'Location', 'northeast');
    grid on;
    out_dir = fileparts(mfilename('fullpath'));
    saveas(fig, fullfile(out_dir, 'part2_bessel_functions.png'));
    close(fig);

    results.helmholtz_eq = helmholtz_eq;
    results.J0_residual = residual;
    results.zeros_J0 = zeros_J0;
    results.zeros_J1 = zeros_J1;
    results.rho = rho_num;
    results.J0 = J0_num;
    results.J1 = J1_num;

    fprintf('\npart2_bessel_coordinates: all checks passed. Figure saved to part2_bessel_functions.png\n');
end

function zs = local_find_bessel_zeros(order, n_zeros)
%LOCAL_FIND_BESSEL_ZEROS  First n_zeros positive zeros of besselj(order,.) via sign-change + fzero.
    zs = zeros(1, n_zeros);
    x = linspace(0.01, 30, 20000);
    y = besselj(order, x);
    sign_changes = find(diff(sign(y)) ~= 0);
    found = 0;
    idx = 1;
    while found < n_zeros && idx <= numel(sign_changes)
        bracket = [x(sign_changes(idx)), x(sign_changes(idx) + 1)];
        z = fzero(@(xx) besselj(order, xx), bracket);
        if z > 1e-6
            found = found + 1;
            zs(found) = z;
        end
        idx = idx + 1;
    end
end
