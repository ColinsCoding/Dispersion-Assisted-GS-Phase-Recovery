function check_against_original(original_dir)
%CHECK_AGAINST_ORIGINAL  Prove the cleaned functions preserve the original equations.
%
%   CHECK_AGAINST_ORIGINAL(original_dir)
%
%   original_dir must contain the original files (not distributed with this
%   repository): mie.m (or "mie-2.m"), rayleighdebye.m and SEALS.m.
%   They are copied to a temporary folder under valid function names and
%   called with the default parameters of main.m; the cleaned functions are
%   then required to return the same numbers.

    if nargin < 1 || ~exist(original_dir, 'dir')
        error('check_against_original:input', 'Give the folder that holds the original .m files.');
    end
    work = tempname;
    mkdir(work);
    copy_first(original_dir, {'mie.m', 'mie-2.m'}, fullfile(work, 'mie.m'));
    copy_first(original_dir, {'rayleighdebye.m'}, fullfile(work, 'rayleighdebye.m'));
    copy_first(original_dir, {'SEALS.m'}, fullfile(work, 'SEALS.m'));
    addpath(work);
    cleanup = onCleanup(@() remove_work_folder(work));   %#ok<NASGU>

    p = scarlett_default_parameters();
    lamvec = linspace(p.lambda_min, p.lambda_max, 500);     % original mie.m needs exactly 500 angles
    lambda = (p.lambda_min + p.lambda_max)/2;

    % SEALS.m (degrees) vs seals_wavelength_to_angle (radians)
    [y_old, theta_old_deg] = SEALS(p.groove_spacing, p.grating_distance, p.grating_angle, ...
        p.dcorr, p.pupil_diameter, p.numerical_aperture, lamvec);
    [y_new, theta_new] = seals_wavelength_to_angle(p.groove_spacing, p.grating_distance, ...
        p.grating_angle, p.dcorr, p.pupil_diameter, p.numerical_aperture, lamvec);
    report('SEALS displacement', y_new, y_old);
    report('SEALS angle', rad2deg(theta_new), theta_old_deg);

    theta = deg2rad(theta_old_deg) + p.measurement_angle;

    % mie.m vs mie_scattering
    [cs, I_p, I_s, an, bn, T_p, T_s] = mie(p.npar, p.nmed, p.dia, lambda, theta, p.r);
    new = mie_scattering(p.npar, p.nmed, p.dia, lambda, theta, p.r);
    report('Mie cross section', new.sigma_sca, cs);
    report('Mie I_p', new.I_par, I_p);
    report('Mie I_s', new.I_perp, I_s);
    report('Mie a_n', new.an, an);
    report('Mie b_n', new.bn, bn);
    report('Mie phase p', exp(1i*new.phase_par), exp(1i*T_p));
    report('Mie phase s', exp(1i*new.phase_perp), exp(1i*T_s));

    % rayleighdebye.m vs rayleigh_debye_scattering(..., 'legacy')
    I_old = rayleighdebye(p.dia, lambda, p.nmed, p.npar, theta, p.r);
    I_new = rayleigh_debye_scattering(p.dia, lambda, p.nmed, p.npar, theta, p.r, 'legacy');
    report('Rayleigh-Debye (legacy mode)', I_new, I_old);

    fprintf('check_against_original: cleaned functions reproduce the originals\n');
end

function remove_work_folder(work)
    rmpath(work);
    rmdir(work, 's');
end

function copy_first(folder, candidates, target)
    for i = 1:numel(candidates)
        source = fullfile(folder, candidates{i});
        if exist(source, 'file')
            copyfile(source, target);
            return
        end
    end
    error('check_against_original:missing', 'None of %s found in %s.', strjoin(candidates, ', '), folder);
end

function report(name, new_value, old_value)
    scale = max(abs(old_value(:)));
    err = max(abs(new_value(:) - old_value(:)))/scale;
    fprintf('%-30s max relative difference %.2e\n', name, err);
    assert(err < 1e-9, 'Mismatch in %s', name);
end
