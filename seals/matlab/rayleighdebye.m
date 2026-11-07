function [I]= rayleighdebye(diameter, lambda, n_background, n_sphere, theta, R)
    radius = diameter/2;
    k=2*pi*n_background/lambda;
    x=k*radius;
    n_rel=n_sphere/n_background; % real refractive index
    u=2*k*radius*sin(theta/2);
    f_theta=1+(cos(theta)).^2; %the base Rayleigh form factor.
    P_theta=3*(sin(u)-u.*cos(u))./u.^3; % the form factor P(theta)
    Rr=P_theta.*f_theta;
    I=abs(Rr*(abs((n_rel^2 - 1)/(n_rel^2 + 2))^2)*(1/(2*R^2))*(((2*pi)/(lambda))^4)*(radius^6));
    %Combining equations to get the intensity.
end 