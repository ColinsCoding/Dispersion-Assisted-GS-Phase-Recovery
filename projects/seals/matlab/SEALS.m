function [y, theta]= SEALS(d, D, a, dcorr, P, NA, lamvec)
    y = 1/6.*D.*tan(a-asin(lamvec./d-sin(a)))./(1+tan(a-asin(lamvec./d-sin(a))).*tan(a));
    y = y - y(end);
    %equation for beam displacement.
    ycenter = (y(1)-y(end))/2;
    theta = atand(2./P.*(y-ycenter+dcorr)*tan(asin(NA)));
    %equation for the theta to beam displacement mapping.
end %look for source material for phase