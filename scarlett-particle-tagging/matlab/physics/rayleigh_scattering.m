function [I, sigma_sca] = rayleigh_scattering(dia, lambda, nmed, npar, theta, r, incident_irradiance)
%RAYLEIGH_SCATTERING  Point-dipole (Rayleigh) limit for a small sphere.
%
%   [I, sigma_sca] = RAYLEIGH_SCATTERING(dia, lambda, nmed, npar, theta, r)
%
%   I          irradiance for unpolarized incident light [W/m^2]
%                I = I0 * (1 + cos(theta)^2)/(2 r^2) * k^4 * radius^6 * |K|^2
%   sigma_sca  total scattering cross section [m^2]
%                sigma = (8 pi/3) * k^4 * radius^6 * |K|^2
%   with k = 2 pi nmed/lambda and K = (m^2 - 1)/(m^2 + 2), m = npar/nmed.
%
%   Valid when x = pi*dia*nmed/lambda << 1 and |m|*x << 1.
%   SI units; theta in radians.

    if nargin < 7
        incident_irradiance = 1;
    end
    if ~(dia > 0 && lambda > 0 && nmed > 0 && r > 0)
        error('rayleigh_scattering:input', 'dia, lambda, nmed and r must be positive.');
    end
    k = 2*pi*nmed/lambda;
    m = npar/nmed;
    lorentz = abs((m^2 - 1)/(m^2 + 2))^2;
    I = incident_irradiance*(1 + cos(theta).^2)/(2*r^2)*k^4*(dia/2)^6*lorentz;
    sigma_sca = 8*pi/3*k^4*(dia/2)^6*lorentz;
end
