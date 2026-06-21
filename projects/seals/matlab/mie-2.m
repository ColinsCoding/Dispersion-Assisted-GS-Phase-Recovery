function [sigma_s, I_p, I_s, an, bn, T_p, T_s]= mie(npar, nmed, dia, lambda, angl, r)
    x = pi*dia/(lambda/nmed);
    phi = pi;                   % Scattering plane
    k = 2*pi/(lambda/nmed);
    A = pi*dia^2/4;
    nmax= round(2+x+4*x^(1/3)); 
    m       = npar/nmed; 
    n   = (1:nmax); nu = (n+0.5);          % Bessel function order
    z   = m.*x; m2 = m.*m;                 % Variable & Parameters
    jx  = besselj(nu, x).*sqrt(0.5*pi./x); % Spherical Bessel function of first kind regarding x
    jz  = besselj(nu, z).*sqrt(0.5*pi./z); % Spherical Bessel function of first kind regarding z
    yx  = bessely(nu, x).*sqrt(0.5*pi./x); % Spherical Bessel function of second kind regarding x
    hx  = jx+1i*yx;                        % Spherical Hankel function of first type
    j1x = [sin(x)/x, jx(1:nmax-1)];        % lower order
    j1z = [sin(z)/z, jz(1:nmax-1)];        % lower order
    y1x = [-cos(x)/x, yx(1:nmax-1)];       % lower order
    h1x = j1x+1i*y1x;                      % lower order
    ax  = x.*j1x-n.*jx;                    % Derivative of x*j(x)
    az  = z.*j1z-n.*jz;                    % Derivative of z*j(z)
    ahx = x.*h1x-n.*hx;                    % Derivative of x*h(z)
    an  = (m2.*jz.*ax-jx.*az)./(m2.*jz.*ahx-hx.*az); % Mie coefficient an
    bn  = (jz.*ax-jx.*az)./(jz.*ahx-hx.*az);         % Mie coefficient bn
    x2  = x*x;
    anp = real(an); anpp = imag(an);       % Real and imaginary part of an
    bnp = real(bn); bnpp = imag(bn);       % Real and imaginary part of bn
    en  = (2*n+1).*(anp.*anp+anpp.*anpp+bnp.*bnp+bnpp.*bnpp); % Summation
    q   = sum(en);
    qsca = 2*q/x2;
    E_theta = zeros([1 500]);        % Space for p polarization
    E_phi   = zeros([1 500]);        % Space for s polarization
    for a=1:500
    theta= angl(a);
     % Scattering angle
    %theta = a*0.002*pi/6
    u    = cos(theta);
    p(1) = 1; %calculation of spherical harmonic functions.
    t(1) = u;
    p(2) = 3*u;
    t(2) = 3*cos(2*acos(u));
    for j  = 3:nmax
        p1 = (2*j-1)./(j-1).*p(j-1).*u;
        p2 = j./(j-1).*p(j-2);
        p(j) = p1-p2;
        t1   = j*u.*p(j);
        t2   = (j+1).*p(j-1);
        t(j) = t1-t2;
    end
    n2   = (2*n+1)./(n.*(n+1));
    pin  = n2.*p;
    tin  = n2.*t;
    S1   = (an.*pin+bn.*tin); %calculation process for the electric field.
    S2   = (an.*tin+bn.*pin);
    S1_c = sum(S1);                 % Scattering amplitude
    S2_c = sum(S2);  
    E_theta(a) = exp(1i*k*r)/(-1i*k*r)*cos(phi)*S2_c; % Parallel
    E_phi(a)   = exp(1i*k*r)/(1i*k*r)*cos(phi)*S1_c;  % Perpendicular
    end
    T_p = angle(E_theta); %finding phase.
    T_s = angle(E_phi);
    sigma_s = qsca*A; 
    I_p = abs(E_theta).^2; % finding intensity.
    I_s = abs(E_phi).^2;
    I_tot = I_p+I_s;
end

    

