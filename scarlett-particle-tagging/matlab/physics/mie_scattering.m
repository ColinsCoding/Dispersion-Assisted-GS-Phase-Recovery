function out = mie_scattering(npar, nmed, dia, lambda, theta, r, incident_irradiance)
%MIE_SCATTERING  Far-field Mie scattering of a plane wave by a homogeneous sphere.
%
%   out = MIE_SCATTERING(npar, nmed, dia, lambda, theta, r)
%   out = MIE_SCATTERING(..., incident_irradiance)
%
%   Cleaned version of the original mie.m. The equations are unchanged
%   (C. Maetzler's formulation of Bohren & Huffman). What changed:
%     * any number of angles (the original required exactly 500);
%     * results returned in a struct with named fields and units;
%     * the unpolarized irradiance is returned explicitly;
%     * inputs are validated.
%
%   Inputs (SI units)
%     npar    particle refractive index [-] (may be complex for absorption)
%     nmed    medium refractive index [-]
%     dia     particle diameter [m]
%     lambda  vacuum wavelength [m]
%     theta   scattering angles [rad], any shape, 0 = forward
%     r       distance from particle to detector [m]
%     incident_irradiance  [W/m^2], default 1
%
%   Output struct fields
%     x              size parameter x = pi*dia/(lambda/nmed) [-]
%     sigma_sca      scattering cross section [m^2]
%     qsca           scattering efficiency sigma_sca / (pi*dia^2/4) [-]
%     an, bn         Mie coefficients, n = 1..nmax
%     S1, S2         scattering amplitudes (perpendicular, parallel) [-]
%     E_par, E_perp  scattered field per unit incident field [-]
%     I_par          irradiance for parallel-polarized incident light [W/m^2]
%     I_perp         irradiance for perpendicular-polarized incident light [W/m^2]
%     I_unpolarized  (I_par + I_perp)/2, unpolarized incident light [W/m^2]
%     phase_par, phase_perp  wrapped phase of E_par, E_perp [rad]
%
%   NOTE on the original main.m: it plotted I_p + I_s. That is the sum of
%   two separate experiments (parallel, then perpendicular illumination, each
%   at full irradiance) and equals 2 * I_unpolarized.

    if nargin < 7
        incident_irradiance = 1;
    end
    if ~(isscalar(dia) && dia > 0 && isscalar(lambda) && lambda > 0 && ...
         isscalar(nmed) && nmed > 0 && isscalar(r) && r > 0 && isscalar(npar))
        error('mie_scattering:input', ...
              'npar, nmed, dia, lambda and r must be scalars; nmed, dia, lambda, r positive.');
    end

    x    = pi*dia/(lambda/nmed);            % size parameter
    k    = 2*pi/(lambda/nmed);              % wavenumber in the medium [rad/m]
    phi  = pi;                              % scattering plane (as in mie.m)
    geometric_area = pi*dia^2/4;            % [m^2]
    nmax = round(2 + x + 4*x^(1/3));        % number of multipoles (Wiscombe)
    m    = npar/nmed;                       % relative refractive index

    n  = 1:nmax;
    nu = n + 0.5;                           % Bessel function order
    z  = m*x;
    m2 = m*m;
    jx  = besselj(nu, x).*sqrt(0.5*pi/x);   % spherical Bessel j_n(x)
    jz  = besselj(nu, z).*sqrt(0.5*pi/z);   % spherical Bessel j_n(z)
    yx  = bessely(nu, x).*sqrt(0.5*pi/x);   % spherical Bessel y_n(x)
    hx  = jx + 1i*yx;                       % spherical Hankel h_n^(1)(x)
    j1x = [sin(x)/x, jx(1:nmax-1)];         % order n-1
    j1z = [sin(z)/z, jz(1:nmax-1)];
    y1x = [-cos(x)/x, yx(1:nmax-1)];
    h1x = j1x + 1i*y1x;
    ax  = x.*j1x - n.*jx;                   % d/dx [x j_n(x)]
    az  = z.*j1z - n.*jz;                   % d/dz [z j_n(z)]
    ahx = x.*h1x - n.*hx;                   % d/dx [x h_n(x)]
    an  = (m2.*jz.*ax - jx.*az)./(m2.*jz.*ahx - hx.*az);
    bn  = (jz.*ax - jx.*az)./(jz.*ahx - hx.*az);

    qsca = 2/(x*x) * sum((2*n + 1).*(abs(an).^2 + abs(bn).^2));

    % Angular functions pi_n and tau_n for all angles at once.
    shape_in = size(theta);
    theta = theta(:).';
    u = cos(theta);
    pin  = zeros(nmax, numel(theta));
    taun = zeros(nmax, numel(theta));
    pin(1, :)  = 1;
    taun(1, :) = u;
    if nmax >= 2
        pin(2, :)  = 3*u;
        taun(2, :) = 3*cos(2*theta);
    end
    for j = 3:nmax
        pin(j, :)  = (2*j - 1)/(j - 1).*pin(j-1, :).*u - j/(j - 1).*pin(j-2, :);
        taun(j, :) = j*u.*pin(j, :) - (j + 1).*pin(j-1, :);
    end
    weight = ((2*n + 1)./(n.*(n + 1))).';   % column, one row per multipole
    S1 = sum(weight.*(an.'.*pin  + bn.'.*taun), 1);
    S2 = sum(weight.*(an.'.*taun + bn.'.*pin), 1);

    E_par  = exp(1i*k*r)/(-1i*k*r)*cos(phi)*S2;   % parallel (p)
    E_perp = exp(1i*k*r)/( 1i*k*r)*cos(phi)*S1;   % perpendicular (s)

    out.x = x;
    out.qsca = qsca;
    out.sigma_sca = qsca*geometric_area;
    out.an = an;
    out.bn = bn;
    out.S1 = reshape(S1, shape_in);
    out.S2 = reshape(S2, shape_in);
    out.E_par  = reshape(E_par, shape_in);
    out.E_perp = reshape(E_perp, shape_in);
    out.I_par  = incident_irradiance*abs(out.E_par).^2;
    out.I_perp = incident_irradiance*abs(out.E_perp).^2;
    out.I_unpolarized = 0.5*(out.I_par + out.I_perp);
    out.phase_par  = angle(out.E_par);
    out.phase_perp = angle(out.E_perp);
end
