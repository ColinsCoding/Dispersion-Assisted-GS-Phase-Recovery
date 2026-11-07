% jaws_of_life_check.m  Independent MATLAB recomputation of dgs/jaws_of_life.py key numbers.
%
% In plain words: a hydraulic rescue spreader is a slider-crank (piston -> crosshead -> link ->
% ear of a bent arm). This script rebuilds the geometry from scratch, gets the tip force two ways
% (virtual work with a numerical derivative, and a free-body force balance), checks energy over
% the full stroke, and recomputes the cutter shear forces. All dimensions and ratings are
% illustrative and written from memory (700 bar, 60 mm bore, 400 mm arms, ...).
%
% Run headless:  matlab -batch "run('learn/matlab/jaws_of_life_check.m')"

L = 0.40; r = 0.12; l = 0.19; delta = deg2rad(25); bore = 0.060; p = 700e5; thmax = deg2rad(60);
A = pi*bore^2/4; Fp = p*A;
c  = @(th) r*cos(th+delta) + sqrt(l^2 - (r*sin(th+delta)).^2);
xp = @(th) c(0) - c(th);
w  = @(th) 2*L*sin(th);

% 1. loop closure and stroke
th = linspace(0, thmax, 7);
E = [-r*cos(th+delta); -r*sin(th+delta)];
C = [-c(th); zeros(size(th))];
closure_err = max(abs(vecnorm(E - C) - l));
fprintf('piston force   %.3f kN\n', Fp/1e3);
fprintf('stroke         %.3f mm\n', xp(thmax)*1e3);
fprintf('max gap        %.3f mm\n', w(thmax)*1e3);
fprintf('loop closure   max | |E-C| - l | = %.2e m\n', closure_err);

% 2. inverse kinematics (law of cosines) round trip
cc = c(th);
th_back = acos((cc.^2 + r^2 - l^2)./(2*cc*r)) - delta;
fprintf('inverse kin.   max |theta error| = %.2e rad\n', max(abs(th_back - th)));

% 3. tip force: virtual work (central difference) vs force balance
h = 1e-7;
fprintf('\n theta  F_s(VW num) kN  F_s(balance) kN\n');
Fvw = zeros(size(th)); Ffb = Fvw;
for k = 1:numel(th)
    t = th(k);
    tp = min(t + h, thmax); tm = max(t - h, 0);
    Fvw(k) = Fp*(xp(tp) - xp(tm))/(w(tp) - w(tm));
    Ek = E(:, k); Ck = C(:, k);
    u = (Ek - Ck)/norm(Ek - Ck);
    N = Fp/(2*u(1));
    F = N*u;
    tau = Ek(1)*F(2) - Ek(2)*F(1);
    Ffb(k) = tau/(L*cos(t));
    fprintf('%6.1f %15.3f %16.3f\n', rad2deg(t), Fvw(k)/1e3, Ffb(k)/1e3);
end
fprintf('max relative VW vs balance difference = %.2e\n', max(abs(Fvw - Ffb)./Ffb));

% 4. energy: integral of F_s dw over the stroke equals F_p * stroke
tt = linspace(0, thmax, 20001);
Fs = arrayfun(@(t) Fp*r*sin(t+delta)*c(t)/(2*L*cos(t)*sqrt(l^2-(r*sin(t+delta))^2)), tt);
W_tip = trapz(tt, Fs.*2*L.*cos(tt));
fprintf('\nenergy: F_p*stroke = %.3f J, integral F_s dw = %.3f J\n', Fp*xp(thmax), W_tip);

% 5. cutter (same model, cutter dimensions), tube shear
Lc = 0.18; rc = 0.09; lc = 0.15; dc = deg2rad(50); Ac = pi*0.070^2/4; Fpc = p*Ac;
cc_ = @(th) rc*cos(th+dc) + sqrt(lc^2 - (rc*sin(th+dc)).^2);
dxp = @(th) rc*sin(th+dc).*(1 + rc*cos(th+dc)./sqrt(lc^2 - (rc*sin(th+dc)).^2));
T15 = Fpc*dxp(deg2rad(15))/2;
At = pi*(0.040^2 - 0.036^2)/4;
fprintf('\ncutter piston force %.1f kN, blade torque at 15 deg %.3f kN m\n', Fpc/1e3, T15/1e3);
fprintf('tube 40x2 area %.2f mm^2; cut force mild %.1f kN, boron %.1f kN\n', At*1e6, 0.6*400e6*At/1e3, 0.6*1500e6*At/1e3);
fprintf('blade force at 15 deg: 50 mm %.1f kN, 70 mm %.1f kN\n', T15/0.05/1e3, T15/0.07/1e3);
fprintf('contact distance D = 40 mm at 20 deg: %.2f mm\n', 20/sind(20));

ok = closure_err < 1e-12 && max(abs(Fvw - Ffb)./Ffb) < 1e-5 && abs(W_tip - Fp*xp(thmax)) < 1e-3*Fp*xp(thmax);
if ok
    disp('ALL CHECKS PASS');
else
    error('jaws_of_life_check: a check failed');
end
