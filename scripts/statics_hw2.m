% statics_hw2.m -- Hibbeler Statics Homework 2 (1-12, 1-18, 1-19, 2-8,
% 2-10, 2-46, 2-101) in MATLAB, as an independent check of
% notebooks/statics_hw2_sympy.ipynb.
%
% Run headless: matlab -batch "run('scripts/statics_hw2.m')"

% ---- Chapter 1: units ----
ft = 0.3048;  lb = 4.4482216152605;  slug = lb/ft;  g = 9.81;
fprintf('1-12_rho_Mg_m3 %.6f\n', 1.94*slug/ft^3/1000);
fprintf('1-18_um_kg_N %.6f\n', 0.204*0.00457/34.6*1e6);
fprintf('1-19_rho_Mg_m3 %.6f\n', 520*lb/ft^3/g/1000);

% ---- 2-8: oblique components by solving [u v]*c = F2 ----
u = [cosd(0); sind(0)];  v = [cosd(75); sind(75)];
F2 = 6*[cosd(-30); sind(-30)];
c = [u v] \ F2;
fprintf('2-8_F2u_kN %.6f\n', c(1));
fprintf('2-8_F2v_kN %.6f\n', abs(c(2)));

% ---- 2-10: FR = F1 + F3 ----
FR = 400*[cosd(-30); sind(-30)] + 250*[cosd(120); sind(120)];
fprintf('2-10_FR_N %.6f\n', norm(FR));
fprintf('2-10_theta_deg %.6f\n', atan2d(FR(2), FR(1)));

% ---- 2-46: FB so the resultant is 1500 N along +y ----
FB = [0; 1500] - 700*[sind(30); cosd(30)];
fprintf('2-46_FB_N %.6f\n', norm(FB));
fprintf('2-46_theta_deg %.6f\n', atan2d(FB(2), -FB(1)));

% ---- 2-101: B(0,0,4); anchors (-3,0,0), (2,-3,0), A(x,y,0) ----
B = [0; 0; 4];
along = @(mag, anchor) mag*(anchor - B)/norm(anchor - B);
F1 = along(350, [-3; 0; 0]);
F2 = along(250, [2; -3; 0]);
lm = -(F1(1:2) + F2(1:2))/300;          % direction cosines of F3
n = sqrt(1 - sum(lm.^2));
xy = lm*4/n;
fprintf('2-101_x_m %.6f\n', xy(1));
fprintf('2-101_y_m %.6f\n', xy(2));
fprintf('2-101_FR_N %.6f\n', -(F1(3) + F2(3) - 300*n));
