function p = scarlett_default_parameters()
%SCARLETT_DEFAULT_PARAMETERS  Default values of the original main.m, in SI units.
%
%   p = SCARLETT_DEFAULT_PARAMETERS() returns a struct. Replaces the
%   interactive "default or custom" prompts of main.m: copy the struct,
%   change the fields you want, and pass it to run_scarlett_plots.
%
%   Field               main.m name   Meaning                                  Unit
%   dia                 dia           particle diameter                        m
%   npar                npar          particle refractive index                -
%   nmed                nmed          medium refractive index                  -
%   r                   r             particle-to-detector distance            m
%   groove_spacing      d             grating groove spacing                   m
%   grating_distance    D             distance between gratings                m
%   grating_angle       a             grating incidence angle                  rad
%   dcorr               dcorr         beam-centre correction                   m
%   pupil_diameter      P             objective entrance pupil diameter        m
%   numerical_aperture  NA            objective numerical aperture             -
%   measurement_angle   mangle        detector angle (main.m used degrees)     rad
%   lambda_min          lam1          lower wavelength limit                   m
%   lambda_max          lam2          upper wavelength limit                   m
%   n_wavelengths       500           number of wavelength samples             -
%   source_bandwidth    band          source bandwidth                         m
%   dia_uncertainty     (new)         relative diameter uncertainty for sweeps -

    p.dia = 9940e-9;
    p.npar = 1.39;
    p.nmed = 1;
    p.r = 0.1;
    p.groove_spacing = 9.0909e-7;
    p.grating_distance = 0.065;
    p.grating_angle = 0.9023;
    p.dcorr = -4.2448e-4;
    p.pupil_diameter = 0.0058;
    p.numerical_aperture = 0.7;
    p.measurement_angle = deg2rad(20);
    p.lambda_min = 1580e-9;
    p.lambda_max = 1600e-9;
    p.n_wavelengths = 500;
    p.source_bandwidth = 20e-9;
    p.dia_uncertainty = 0.02;
end
