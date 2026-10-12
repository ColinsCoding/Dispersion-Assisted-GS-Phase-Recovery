function run_scarlett_plots(p, output_dir)
%RUN_SCARLETT_PLOTS  Non-interactive replacement for the original main.m.
%
%   RUN_SCARLETT_PLOTS()                 default parameters, saves to results/matlab
%   RUN_SCARLETT_PLOTS(p)                p from scarlett_default_parameters()
%   RUN_SCARLETT_PLOTS(p, output_dir)    choose the output folder
%
%   Saves PNG figures and two CSV files (used to compare MATLAB with the
%   Python port). No prompts, no absolute paths: the default output folder is
%   <repository>/results/matlab, found relative to this file.
%
%   Figures (1-6 are the figures of main.m, corrected; 7-8 are new)
%     1  beam displacement vs wavelength
%     2  scattering angle vs wavelength
%     3  scattered irradiance vs scattering angle: Mie polarization components, Rayleigh-Debye
%     4  polar plot of the Mie pattern (log scale)
%     5  irradiance vs wavelength including the source spectrum
%     6  phase of the Mie field vs wavelength
%     7  scattering efficiency vs diameter: Rayleigh, Rayleigh-Debye, Mie
%     8  sensitivity of the pattern to the diameter uncertainty
%
%   All values are model output. Incident irradiance is 1 W/m^2.

    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, '..', 'physics'));
    if nargin < 1 || isempty(p)
        p = scarlett_default_parameters();
    end
    if nargin < 2 || isempty(output_dir)
        output_dir = fullfile(here, '..', '..', 'results', 'matlab');
    end
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    lambda_vec = linspace(p.lambda_min, p.lambda_max, p.n_wavelengths);
    lambda_center = (p.lambda_min + p.lambda_max)/2;
    [y, theta_illum] = seals_wavelength_to_angle(p.groove_spacing, p.grating_distance, ...
        p.grating_angle, p.dcorr, p.pupil_diameter, p.numerical_aperture, lambda_vec);
    theta = theta_illum + p.measurement_angle;          % scattering angle [rad]
    theta_deg = rad2deg(theta);
    lineshape = laser_lineshape(lambda_vec, lambda_center, p.source_bandwidth);

    mie = mie_scattering(p.npar, p.nmed, p.dia, lambda_center, theta, p.r);
    I_rdg = rayleigh_debye_scattering(p.dia, lambda_center, p.nmed, p.npar, theta, p.r);

    fig = figure('Visible', 'off');
    plot(lambda_vec*1e9, y*1e3, 'LineWidth', 1.2); grid on;
    xlabel('Wavelength (nm)'); ylabel('Beam displacement (mm)');
    title('SEALS: beam displacement vs wavelength');
    save_figure(fig, output_dir, 'fig1_displacement_vs_wavelength');

    fig = figure('Visible', 'off');
    plot(lambda_vec*1e9, theta_deg, 'LineWidth', 1.2); grid on;
    xlabel('Wavelength (nm)'); ylabel('Scattering angle (deg)');
    title('SEALS: scattering angle vs wavelength');
    save_figure(fig, output_dir, 'fig2_angle_vs_wavelength');

    fig = figure('Visible', 'off');
    semilogy(theta_deg, mie.I_par, theta_deg, mie.I_perp, theta_deg, mie.I_unpolarized, ...
             theta_deg, I_rdg, '--', 'LineWidth', 1.1); grid on;
    xlabel('Scattering angle (deg)'); ylabel('Scattered irradiance (W/m^2), log scale');
    legend('Mie, parallel input', 'Mie, perpendicular input', 'Mie, unpolarized', ...
           'Rayleigh-Debye (outside its valid range here)', 'Location', 'best');
    title(sprintf('dia = %.2f um, x = %.1f, r = %.2f m, I_0 = 1 W/m^2', p.dia*1e6, mie.x, p.r));
    save_figure(fig, output_dir, 'fig3_irradiance_vs_angle');

    fig = figure('Visible', 'off');
    polarplot(theta, log10(mie.I_unpolarized) - min(log10(mie.I_unpolarized)), 'LineWidth', 1.1);
    title('Mie pattern: log_{10}(I / I_{min}) vs scattering angle');
    save_figure(fig, output_dir, 'fig4_polar_pattern');

    fig = figure('Visible', 'off');
    semilogy(lambda_vec*1e9, mie.I_unpolarized.*lineshape, 'LineWidth', 1.2); grid on;
    xlabel('Wavelength (nm)'); ylabel('Irradiance x source spectrum (W/m^2), log scale');
    title('Detected spectrum (Mie, unpolarized)');
    save_figure(fig, output_dir, 'fig5_irradiance_vs_wavelength');

    fig = figure('Visible', 'off');
    plot(lambda_vec*1e9, rad2deg(mie.phase_par), lambda_vec*1e9, rad2deg(mie.phase_perp), 'LineWidth', 1.1);
    grid on; xlabel('Wavelength (nm)'); ylabel('Phase (deg), wrapped to [-180, 180]');
    legend('parallel (p)', 'perpendicular (s)', 'Location', 'best');
    title('Phase of the scattered Mie field vs wavelength');
    save_figure(fig, output_dir, 'fig6_phase_vs_wavelength');

    % Regimes: efficiency Q = sigma / (pi radius^2) against diameter.
    diameters = logspace(log10(20e-9), log10(20e-6), 160);
    q_mie = zeros(size(diameters)); q_ray = q_mie; q_rdg = q_mie;
    th_int = linspace(0, pi, 2001);
    for i = 1:numel(diameters)
        d = diameters(i);
        area = pi*d^2/4;
        m_i = mie_scattering(p.npar, p.nmed, d, lambda_center, 0, 1);
        q_mie(i) = m_i.qsca;
        [~, sigma_ray] = rayleigh_scattering(d, lambda_center, p.nmed, p.npar, 0, 1);
        q_ray(i) = sigma_ray/area;
        dsigma = rayleigh_debye_scattering(d, lambda_center, p.nmed, p.npar, th_int, 1);
        q_rdg(i) = 2*pi*trapz(th_int, dsigma.*sin(th_int))/area;
    end
    fig = figure('Visible', 'off');
    loglog(diameters*1e6, q_mie, diameters*1e6, q_ray, '--', diameters*1e6, q_rdg, ':', 'LineWidth', 1.2);
    grid on; xlabel('Particle diameter (um)'); ylabel('Scattering efficiency Q_{sca} (dimensionless)');
    legend('Mie (exact for a sphere)', 'Rayleigh', 'Rayleigh-Debye', 'Location', 'northwest');
    title(sprintf('Regimes at \\lambda = %.0f nm, n_{par} = %.2f, n_{med} = %.2f', ...
                  lambda_center*1e9, p.npar, p.nmed));
    ylim([1e-9 1e2]);
    save_figure(fig, output_dir, 'fig7_regimes_vs_diameter');

    % Diameter sensitivity: dia +/- uncertainty.
    lo = mie_scattering(p.npar, p.nmed, p.dia*(1 - p.dia_uncertainty), lambda_center, theta, p.r);
    hi = mie_scattering(p.npar, p.nmed, p.dia*(1 + p.dia_uncertainty), lambda_center, theta, p.r);
    fig = figure('Visible', 'off');
    semilogy(theta_deg, lo.I_unpolarized, theta_deg, mie.I_unpolarized, theta_deg, hi.I_unpolarized, 'LineWidth', 1.1);
    grid on; xlabel('Scattering angle (deg)'); ylabel('Scattered irradiance (W/m^2), log scale');
    legend(sprintf('dia - %g %%', 100*p.dia_uncertainty), 'nominal dia', ...
           sprintf('dia + %g %%', 100*p.dia_uncertainty), 'Location', 'best');
    title('Sensitivity of the Mie pattern to diameter');
    save_figure(fig, output_dir, 'fig8_diameter_sensitivity');

    % Numbers for the MATLAB <-> Python comparison.
    T = table(theta(:), mie.I_par(:), mie.I_perp(:), mie.phase_par(:), mie.phase_perp(:), ...
              I_rdg(:), y(:), lineshape(:), ...
              'VariableNames', {'theta_rad', 'I_par', 'I_perp', 'phase_par', 'phase_perp', ...
                                'I_rayleigh_debye', 'beam_displacement_m', 'lineshape'});
    writetable(T, fullfile(output_dir, 'parity_default.csv'));
    S = table(mie.x, mie.sigma_sca, mie.qsca, 'VariableNames', {'x', 'sigma_sca_m2', 'qsca'});
    writetable(S, fullfile(output_dir, 'parity_scalars.csv'));

    fprintf('Size parameter x            : %.4f\n', mie.x);
    fprintf('Mie scattering cross section: %.6e m^2\n', mie.sigma_sca);
    fprintf('Saved 8 figures and 2 CSV files to %s\n', output_dir);
end

function save_figure(fig, output_dir, name)
    exportgraphics(fig, fullfile(output_dir, [name '.png']), 'Resolution', 130);
    close(fig);
end
