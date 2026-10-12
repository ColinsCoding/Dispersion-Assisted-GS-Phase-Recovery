function [beam_displacement, theta] = seals_wavelength_to_angle(groove_spacing, grating_distance, ...
        grating_angle, dcorr, pupil_diameter, numerical_aperture, lambda_vec, magnification)
%SEALS_WAVELENGTH_TO_ANGLE  SEALS wavelength-to-angle mapping.
%
%   [y, theta] = SEALS_WAVELENGTH_TO_ANGLE(d, D, alpha, dcorr, P, NA, lambda_vec)
%   [y, theta] = SEALS_WAVELENGTH_TO_ANGLE(..., magnification)
%
%   Cleaned version of the original SEALS.m (Adam et al., "Spectrally encoded
%   angular light scattering", Opt. Express 21, 28960 (2013), Eqs. 1 and 2).
%
%       beta  = asin(lambda/d - sin(alpha))                   first diffraction order
%       y     = M * D * tan(alpha - beta) / (1 + tan(alpha - beta)*tan(alpha))
%       y     = y - y(end)                                    displacement from the long-wavelength edge
%       y_c   = (y(1) - y(end))/2
%       theta = atan( 2/P * (y - y_c + dcorr) * tan(asin(NA)) )
%
%   Inputs (SI units)
%     groove_spacing      d, grating groove spacing [m]
%     grating_distance    D, distance between the gratings [m]
%     grating_angle       alpha, incidence angle on the grating [rad]
%     dcorr               beam-centre correction term [m]
%     pupil_diameter      P, objective entrance pupil diameter [m]
%     numerical_aperture  NA of the objective (in air) [-]
%     lambda_vec          increasing vector of wavelengths [m]
%     magnification       M, telescope magnification, default 1/6
%
%   Outputs
%     beam_displacement   y [m]
%     theta               illumination angle [rad]   (SEALS.m returned DEGREES)
%
%   Changes from SEALS.m: theta is in radians; the telescope magnification
%   1/6 is a named parameter instead of a bare number; the grating equation
%   is checked so an evanescent order raises an error instead of silently
%   producing complex numbers. The sign convention of y follows the code
%   (+M); the paper's Eq. (1) prints -M, which only flips the y axis.

    if nargin < 8
        magnification = 1/6;
    end
    if numel(lambda_vec) < 2 || any(diff(lambda_vec) <= 0)
        error('seals_wavelength_to_angle:lambda', 'lambda_vec must be increasing with at least 2 points.');
    end
    if ~(numerical_aperture > 0 && numerical_aperture < 1)
        error('seals_wavelength_to_angle:na', 'numerical_aperture must be in (0, 1) for a lens in air.');
    end
    s = lambda_vec./groove_spacing - sin(grating_angle);
    if any(abs(s) > 1)
        error('seals_wavelength_to_angle:grating', ...
              'Grating equation has no solution: |lambda/d - sin(alpha)| > 1 for some wavelength.');
    end

    t = tan(grating_angle - asin(s));
    beam_displacement = magnification.*grating_distance.*t./(1 + t.*tan(grating_angle));
    beam_displacement = beam_displacement - beam_displacement(end);
    y_center = (beam_displacement(1) - beam_displacement(end))/2;
    theta = atan(2./pupil_diameter.*(beam_displacement - y_center + dcorr)*tan(asin(numerical_aperture)));
end
