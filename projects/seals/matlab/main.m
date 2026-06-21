close all;
clear all;
disp('Qualities of the Particle and Medium');
disp('Would you like to use default or custom values?'); 
%option to use default SEALS parameters or custom variables.
default = 'z';
while default ~= 'd' & default ~= 'c'
    default = input('Default or custom values? Type d or c: ', 's');
end
if default == 'd'
    dia = 9940 * 1e-9; %all default parameters below.
    npar = 1.39;
    nmed = 1;
    raymie = 'b';
    r = .1;
    while raymie ~= 'r' & raymie ~= 'm'
        raymie = input('Rayleigh Debye or Mie Model? Type r or m: ', 's');
    end
    d =  9.0909e-7;
    D = .065;
    a =  .9023;
    dcorr = -4.2448e-4;
    P = .0058;
    NA = .7;
    mangle = 20;
    lam1 = 1580*10^-9;
    lam2 = 1600*10^-9;
    lambda = (lam1+lam2)/2;
    lamvec = linspace(lam1, lam2, 500);
else %all prompts for custom values.
    disp('Qualities of the Particle and Medium');
    dia =input('The diameter of particle in nm(e.g., 20):')*1e-9;
    npar =input('Refractive index of sphere (e.g., 1.57):');
    nmed =input('Refractive index of background (e.g., 1.33 for water):');
    r = input('Distance of Measurement Device from particle (m): ');
    raymie = 'b';
    while raymie ~= 'r' & raymie ~= 'm'
        raymie = input('Rayleigh Debye or Mie Model? Type r or m: ', 's');
    end %choice between rayleigh or mie.
    disp('Properties of the SEALS Device');
    d =input('Groove spacing in nm: ')*10^-9;
    D = input('Distance between gratings in m: ');
    a = input('Grating tilt angle compared to horizontal in rad: ');
    dcorr = input('Lens correctional term in m: ');
    P = input('Diameter of lens in m: ');
    NA = input('Numerical Aperture of lens: ');
    mangle = input('Measurement Angle (deg): ');
    lam1 = input('SEALS Laser lower wavelength limit (nm): ')*10^-9;
    lam2 = input('SEALS Laser upper wavelength limit (nm): ')*10^-9;
    lambda = (lam1+lam2)/2; %wavelength used for calculations considered as center of wavelength range.
    lamvec = linspace(lam1, lam2, 500);
end
[y, theta] = SEALS(d, D, a, dcorr, P, NA, lamvec); %running SEALS function
figure(1);
plot(lamvec,y);
xlabel('Wavelength (m)');
ylabel('Vertical Displacement (m)');
title('Vertical Displacement vs. Wavelength');
figure(2);
theta = theta + mangle; %the scattering angle is shifted based on measurement angle. 
plot(lamvec,theta);
title('Scattering Angle vs. Wavelength');
xlabel('Wavelength (m)');
ylabel('Scattering Angle (deg)'); %two plots correspond to displacement and scattering angle vs. wavelength.
band = 20*10^-9;
c = 3*10^8;
vband = c/(lambda^2)*band;
vvec = linspace(c/lam2,c/lam1,500);
lineshape = vband./(2*pi*((vvec-c/lambda).^2+(vband/2)^2));
lineshape = lineshape./max(lineshape); %taking into account the laser intensity distribution.
if raymie == 'm' 
    [cs, I_p, I_s, an, bn,T_p,T_s] = mie(npar, nmed, dia, lambda, deg2rad(theta), r);
    %running mie code function with given inputs.
    figure(3); %Plotting data, intensity vs. scattering angle and wavelength, phase information.
    I_tot = (I_p +I_s);
    plot((theta),10*log10(I_tot));
    title('Intensity vs. Scattering Angle for Mie Scattering');
    xlabel('Scattering Angle (deg)');
    ylabel('Intensity (dB)');
    figure(4);
    polarplot(deg2rad(theta),log10(I_tot) - min(log10(I_tot)));
    title('Intensity (log scale) vs. Scattering Angle for Mie Scattering');
    figure(5);
    Im = I_tot.*lineshape;
    plot (lamvec,10*log10(Im));
    title('Intensity vs. Wavelength for Mie Scattering');
    ylabel('Intensity (dB)');
    xlabel('Wavelength (m)');
    figure(6);
    plot (lamvec,rad2deg(T_p));
    hold on;
    plot(lamvec,rad2deg(T_s));
    hold off;
    title('Phase vs. Scattering Angle for Mie Scattering');
    xlabel('Wavelength (m)');
    ylabel('Phase (deg)');
    legend('P-polarized', 'S-polarized');
    X = ['The mie scattering cross section is ', num2str(cs),  '.'];
    disp(X);
else
    [I] = rayleighdebye(dia, lambda, nmed, npar, deg2rad(theta), r);
    %running the rayleighdebye function.
    hold on;
    figure(3); %plotting intensity vs. scattering angle and wavelength.
    plot(theta,10*log10(I));
    plot(theta,(I));
    title('Intensity vs. Scattering Angle for Rayleigh Scattering');
    xlabel('Scattering Angle (deg)');
    ylabel('Intensity (dB)');
    hold off;
    figure(4);
    polarplot(deg2rad(theta),log10(I) - min(log10(I)));
    title('Intensity (log scale) vs. Scattering Angle for Rayleigh Scattering');
    figure(5);
    plot (lamvec,10*log10(I));
    title('Intensity vs. Wavelength for Rayleigh Scattering');
    ylabel('Intensity (dB)');
    xlabel('Wavelength(m)');
end
disp('Note: All calculations are done in W/m^2. The laser is assumed to have peak intensity 1 W/m^2');





