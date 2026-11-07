% 3D Bode magnitude surface for a first-order low-pass filter
clear; clc; close all;

w  = logspace(-1, 5, 500);    % angular frequency [rad/s]
wc = logspace(0, 4, 100);     % cutoff frequency [rad/s]

[W, WC] = meshgrid(w, wc);

H = 1 ./ (1 + 1i .* W ./ WC);
magnitude_dB = 20 * log10(abs(H));
phase_deg = angle(H) * 180/pi;

figure;
surf(log10(W), log10(WC), magnitude_dB, 'EdgeColor', 'none');
xlabel('log_{10}(\omega) [rad/s]');
ylabel('log_{10}(\omega_c) [rad/s]');
zlabel('|H(j\omega)| [dB]');
title('3D Bode Magnitude Surface');
colorbar;
view(45, 30);
grid on;

figure;
surf(log10(W), log10(WC), phase_deg, 'EdgeColor', 'none');
xlabel('log_{10}(\omega) [rad/s]');
ylabel('log_{10}(\omega_c) [rad/s]');
zlabel('Phase [degrees]');
title('3D Bode Phase Surface');
colorbar;
view(45, 30);
grid on;