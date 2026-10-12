function test_rayleigh_mie_limits()
%TEST_RAYLEIGH_MIE_LIMITS  Unit tests: Rayleigh, Rayleigh-Debye and Mie must
%agree where their ranges of validity overlap, and Mie must obey exact identities.
%
%   Run from the repository root:
%       matlab -batch "addpath('matlab'); scarlett_setup; test_rayleigh_mie_limits"
%   Every check is an assert; the function prints one line per group and
%   errors on the first failure.

    lambda = 1590e-9;
    theta = linspace(0.05, pi - 0.05, 41);

    % 1. Size parameter definition.
    assert(abs(size_parameter(1e-6, 500e-9, 1.33) - pi*1e-6*1.33/500e-9) < 1e-12);
    fprintf('ok  size parameter\n');

    % 2. Small particle: Mie -> Rayleigh (angular pattern and cross section),
    %    in vacuum and in a medium.
    for nmed = [1.0, 1.33]
        npar = 1.39*nmed;
        dia = 20e-9;
        mie = mie_scattering(npar, nmed, dia, lambda, theta, 0.1);
        [I_ray, sigma_ray] = rayleigh_scattering(dia, lambda, nmed, npar, theta, 0.1);
        assert(mie.x < 0.06);
        assert(max(abs(mie.I_unpolarized./I_ray - 1)) < 5e-3);
        assert(abs(mie.sigma_sca/sigma_ray - 1) < 1e-3);
        % Rayleigh polarization: perpendicular is isotropic, parallel ~ cos^2.
        assert(max(abs(mie.I_par./mie.I_perp - cos(theta).^2)) < 5e-3);
    end
    fprintf('ok  Mie reduces to Rayleigh for x << 1\n');

    % 3. Rayleigh scaling laws: sigma ~ dia^6 and ~ lambda^-4.
    [~, s1] = rayleigh_scattering(10e-9, lambda, 1, 1.39, 0, 1);
    [~, s2] = rayleigh_scattering(20e-9, lambda, 1, 1.39, 0, 1);
    [~, s3] = rayleigh_scattering(10e-9, lambda/2, 1, 1.39, 0, 1);
    assert(abs(s2/s1 - 2^6) < 1e-9 && abs(s3/s1 - 2^4) < 1e-9);
    m1 = mie_scattering(1.39, 1, 10e-9, lambda, 0, 1);
    m2 = mie_scattering(1.39, 1, 20e-9, lambda, 0, 1);
    assert(abs(m2.sigma_sca/m1.sigma_sca/2^6 - 1) < 1e-3);
    fprintf('ok  dia^6 and lambda^-4 scaling\n');

    % 4. Rayleigh-Debye reduces to Rayleigh for a small particle, and agrees
    %    with Mie for an optically soft particle (m close to 1).
    I_rdg = rayleigh_debye_scattering(20e-9, lambda, 1.33, 1.39, theta, 0.1);
    I_ray = rayleigh_scattering(20e-9, lambda, 1.33, 1.39, theta, 0.1);
    assert(max(abs(I_rdg./I_ray - 1)) < 5e-3);       % form factor is 1 - u^2/5, u <= 0.105 here
    soft = mie_scattering(1.34, 1.33, 1e-6, 633e-9, theta(1:8), 0.1);     % forward lobe
    I_soft = rayleigh_debye_scattering(1e-6, 633e-9, 1.33, 1.34, theta(1:8), 0.1);
    assert(max(abs(I_soft./soft.I_unpolarized - 1)) < 0.05);
    assert(isfinite(rayleigh_debye_scattering(1e-6, 633e-9, 1.33, 1.34, 0, 0.1)));   % no NaN at theta = 0
    fprintf('ok  Rayleigh-Debye limits\n');

    % 5. Mie identities for a non-absorbing sphere of any size.
    for x_target = [0.5, 3, 20, 60]
        dia = x_target*lambda/pi;
        mie = mie_scattering(1.39, 1, dia, lambda, [0, 0.3], 1);
        % optical theorem: Q_ext = (4/x^2) Re S(0); no absorption => Q_ext = Q_sca
        assert(abs(4/mie.x^2*real(mie.S1(1)) - mie.qsca) < 1e-9*max(1, mie.qsca));
        assert(abs(mie.S1(1) - mie.S2(1)) < 1e-9*abs(mie.S1(1)));      % S1(0) = S2(0)
        % the differential cross section integrates to the total
        th = linspace(0, pi, 20001);
        full = mie_scattering(1.39, 1, dia, lambda, th, 1);
        total = 2*pi*trapz(th, full.I_unpolarized.*sin(th));           % r = 1, I0 = 1
        assert(abs(total/full.sigma_sca - 1) < 1e-4);
    end
    fprintf('ok  optical theorem and energy conservation\n');

    % 6. Large particle: Rayleigh must FAIL (it is not valid there).
    big = mie_scattering(1.39, 1, 9940e-9, lambda, 0, 1);
    [~, sigma_ray_big] = rayleigh_scattering(9940e-9, lambda, 1, 1.39, 0, 1);
    assert(sigma_ray_big/big.sigma_sca > 100);
    fprintf('ok  Rayleigh overestimates by >100x at x = %.1f\n', big.x);

    % 7. The original plot quantity I_p + I_s is twice the unpolarized irradiance.
    mie = mie_scattering(1.39, 1, 9940e-9, lambda, theta, 0.1);
    assert(max(abs((mie.I_par + mie.I_perp)./mie.I_unpolarized - 2)) < 1e-12);
    % ... and the legacy Rayleigh-Debye code is off by nmed^4 in a medium.
    I_legacy = rayleigh_debye_scattering(20e-9, lambda, 1.33, 1.39*1.33, theta, 0.1, 'legacy');
    I_fixed  = rayleigh_debye_scattering(20e-9, lambda, 1.33, 1.39*1.33, theta, 0.1);
    assert(max(abs(I_fixed./I_legacy/1.33^4 - 1)) < 2e-3);
    fprintf('ok  documented legacy discrepancies reproduce\n');

    % 8. SEALS map: monotonic, matches the paper''s 5.2 mm displacement span,
    %    and rejects an evanescent grating order.
    p = scarlett_default_parameters();
    lam = linspace(p.lambda_min, p.lambda_max, 500);
    [y, th_illum] = seals_wavelength_to_angle(p.groove_spacing, p.grating_distance, p.grating_angle, ...
        p.dcorr, p.pupil_diameter, p.numerical_aperture, lam);
    assert(all(diff(th_illum) < 0) && y(end) == 0);
    assert(abs(y(1) - 5.2275e-3) < 1e-6);
    failed = false;
    try
        seals_wavelength_to_angle(p.groove_spacing, p.grating_distance, 0.1, p.dcorr, ...
            p.pupil_diameter, p.numerical_aperture, lam);
    catch
        failed = true;
    end
    assert(failed);
    fprintf('ok  SEALS wavelength-to-angle map\n');

    % 9. Lineshape peaks at the centre wavelength and is symmetric in frequency.
    shape = laser_lineshape(lam, 1590e-9, 20e-9);
    [~, imax] = max(shape);
    assert(abs(lam(imax) - 1590e-9) < 0.05e-9 && abs(max(shape) - 1) < 1e-12);
    fprintf('ok  laser lineshape\n');

    % 10. Sensor chain: Nyquist check and a hand-calculated operating point.
    s = photodiode_tia_adc(1e-9);               % 1 nW
    assert(abs(s.current - 2e-9) < 1e-18 && abs(s.volts - 2e-3) < 1e-12);
    assert(s.gain == 128 && s.codes == round(2e-3*128/(1/4096)) && ~s.clipped);
    failed = false;
    try
        photodiode_tia_adc(1e-9, struct('antialias_cutoff', 60e3));
    catch
        failed = true;
    end
    assert(failed);
    fprintf('ok  photodiode / TIA / ADC model\n');

    fprintf('test_rayleigh_mie_limits: all checks passed\n');
end
