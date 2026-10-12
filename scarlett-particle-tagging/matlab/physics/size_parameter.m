function x = size_parameter(dia, lambda, nmed)
%SIZE_PARAMETER  Mie size parameter x = pi*dia/(lambda/nmed).
%
%   x = SIZE_PARAMETER(dia, lambda, nmed)
%
%   dia     particle diameter [m] (scalar or array)
%   lambda  vacuum wavelength [m]
%   nmed    medium refractive index [-], default 1
%
%   x is the particle circumference divided by the wavelength in the medium.
%   x << 1: Rayleigh regime.  x ~ 1 or larger: Mie theory is needed.

    if nargin < 3
        nmed = 1;
    end
    if any(dia(:) <= 0) || any(lambda(:) <= 0) || any(nmed(:) <= 0)
        error('size_parameter:input', 'dia, lambda and nmed must be positive.');
    end
    x = pi.*dia./(lambda./nmed);
end
