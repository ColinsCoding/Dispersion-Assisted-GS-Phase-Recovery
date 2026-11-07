
% desk_tdgsa.m - fixed
load('desk_capture.mat');            % t, dt, f, I1, I2, D1_psnm, D2_psnm, truth_re, truth_im

% constants
lambda = 1550e-9; c = 299792458;

% compute dispersion (double)
b1 = double(-D1_psnm*1e-3*lambda^2/(2*pi*c));
b2 = double(-D2_psnm*1e-3*lambda^2/(2*pi*c));

% frequency vector and shapes
f  = double(f(:));      % column
w  = 2*pi*f.';          % row

% element-wise phase with compatible sizes (use bsxfun if old MATLAB)
phase = (b2 - b1).';            % row
H12 = exp(1i * phase .* (w.^2) / 2);
% If implicit expansion unavailable:
% H12 = exp(1i * bsxfun(@times, (b2 - b1).', (w.^2) / 2));

f1 = sqrt(double(I1(:).')); f2 = sqrt(double(I2(:).'));
u = f1; err = zeros(1, n_iter);

for k = 1:n_iter
    u2 = ifft(fft(u) .* H12);
    err(k) = sqrt(mean((abs(u2)-f2).^2)) / max(f2);
    u2 = f2 .* exp(1i*angle(u2));
    u  = f1 .* exp(1i*angle(ifft(fft(u2) ./ H12)));
end

U = fft(u) .* exp(-1i * (b1).' .* (w.^2) / 2);  % ensure shapes match
truth = double(truth_re) + 1i*double(truth_im);
a = (U * truth') / (U * U'); U = conj(a) * U;
fprintf('field error vs truth: %.3f\n', norm(U-truth)/norm(truth));