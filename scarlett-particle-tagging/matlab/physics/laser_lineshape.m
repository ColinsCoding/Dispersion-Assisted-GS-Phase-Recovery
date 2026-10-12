function shape = laser_lineshape(lambda_vec, lambda_center, bandwidth, mode)
%LASER_LINESHAPE  Lorentzian source spectrum, peak-normalized.
%
%   shape = LASER_LINESHAPE(lambda_vec, lambda_center, bandwidth)
%   shape = LASER_LINESHAPE(..., mode)       mode = 'corrected' | 'legacy'
%
%       nu  = c/lambda,  nu0 = c/lambda_center,  dnu = c*bandwidth/lambda_center^2
%       L   = dnu / (2 pi ((nu - nu0)^2 + (dnu/2)^2)),   shape = L/max(L)
%
%   'corrected' (default) evaluates L at the frequency of each wavelength in
%   lambda_vec, so shape(i) belongs to lambda_vec(i).
%
%   'legacy' reproduces the original main.m, which evaluated L on
%   linspace(c/lambda_max, c/lambda_min, N) (increasing FREQUENCY) and then
%   multiplied it element by element with data ordered by increasing
%   WAVELENGTH. The spectrum was therefore applied back to front, on a grid
%   that is not the wavelength grid. It also used c = 3e8.
%
%   Wavelengths and bandwidth in metres.

    if nargin < 4 || isempty(mode)
        mode = 'corrected';
    end
    if ~(bandwidth > 0 && lambda_center > 0)
        error('laser_lineshape:input', 'bandwidth and lambda_center must be positive.');
    end
    switch lower(mode)
        case 'legacy'
            c = 3e8;
            nu = linspace(c/lambda_vec(end), c/lambda_vec(1), numel(lambda_vec));
        case 'corrected'
            c = 299792458;
            nu = c./lambda_vec;
        otherwise
            error('laser_lineshape:mode', 'mode must be ''corrected'' or ''legacy''.');
    end
    nu0 = c/lambda_center;
    dnu = c/lambda_center^2*bandwidth;
    shape = dnu./(2*pi*((nu - nu0).^2 + (dnu/2)^2));
    shape = shape./max(shape);
end
