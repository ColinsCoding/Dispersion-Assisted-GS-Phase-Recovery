function I = rayleigh_debye_scattering(dia, lambda, nmed, npar, theta, r, mode, incident_irradiance)
%RAYLEIGH_DEBYE_SCATTERING  Scattered irradiance, Rayleigh-Debye model [W/m^2].
%
%   I = RAYLEIGH_DEBYE_SCATTERING(dia, lambda, nmed, npar, theta, r)
%   I = RAYLEIGH_DEBYE_SCATTERING(..., mode)        mode = 'corrected' | 'legacy'
%   I = RAYLEIGH_DEBYE_SCATTERING(..., mode, incident_irradiance)
%
%   Unpolarized incident light. Valid when |m - 1| << 1 and 2*x*|m - 1| << 1,
%   with m = npar/nmed and x the size parameter.
%
%   'corrected' (default)
%       I = I0 * (1 + cos(theta)^2)/(2 r^2) * k^4 * radius^6 * |K|^2 * P(theta)
%       K = (m^2 - 1)/(m^2 + 2)
%       P(theta) = [3 (sin u - u cos u)/u^3]^2,  u = 2 k radius sin(theta/2)
%       k = 2 pi nmed / lambda   (wavenumber IN THE MEDIUM)
%
%   'legacy'
%       Reproduces the original rayleighdebye.m exactly. It differs in two ways:
%         1. the form amplitude 3(sin u - u cos u)/u^3 enters to the FIRST
%            power (wrapped in abs) instead of squared;
%         2. the k^4 prefactor uses the vacuum wavenumber 2*pi/lambda, which
%            is wrong by nmed^4 when nmed ~= 1.
%       It also returns NaN at theta = 0 (0/0).
%
%   Inputs in SI units; theta in radians.

    if nargin < 7 || isempty(mode)
        mode = 'corrected';
    end
    if nargin < 8
        incident_irradiance = 1;
    end
    if ~(dia > 0 && lambda > 0 && nmed > 0 && r > 0)
        error('rayleigh_debye_scattering:input', 'dia, lambda, nmed and r must be positive.');
    end

    radius = dia/2;
    k = 2*pi*nmed/lambda;
    n_rel = npar/nmed;
    lorentz = abs((n_rel^2 - 1)/(n_rel^2 + 2))^2;
    u = 2*k*radius*sin(theta/2);
    dipole = 1 + cos(theta).^2;

    switch lower(mode)
        case 'legacy'
            form_amplitude = 3*(sin(u) - u.*cos(u))./u.^3;
            I = abs(form_amplitude.*dipole*lorentz*(1/(2*r^2))*((2*pi/lambda)^4)*(radius^6));
        case 'corrected'
            form_amplitude = ones(size(u));
            big = abs(u) > 1e-4;
            form_amplitude(big) = 3*(sin(u(big)) - u(big).*cos(u(big)))./u(big).^3;
            form_amplitude(~big) = 1 - u(~big).^2/10;      % series limit, avoids 0/0
            I = incident_irradiance*dipole/(2*r^2)*k^4*radius^6*lorentz.*form_amplitude.^2;
        otherwise
            error('rayleigh_debye_scattering:mode', 'mode must be ''corrected'' or ''legacy''.');
    end
end
