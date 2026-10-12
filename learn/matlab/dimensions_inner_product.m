% dimensions_inner_product.m -- loop over integer and fractional dimensions,
% Monte Carlo ball volumes, and the complex inner product (with the ' vs .' trap).
% MATLAB twin of dgs/dimensions_inner_product.py.
% Run headless:  matlab -batch "run('learn/matlab/dimensions_inner_product.m')"

rng(0);
dims = [0.5 1 1.5 2 2.5 3 4 5 10];
fprintf('   d    V_d (Gamma formula)   Monte Carlo\n');
for d = dims
    V = pi^(d/2) / gamma(d/2 + 1);              % works for fractional d
    if d == round(d)
        n = 400000;
        pts = 2*rand(n, d) - 1;                 % cube [-1,1]^d
        Vmc = mean(sum(pts.^2, 2) <= 1) * 2^d;
        fprintf('%5.1f   %10.6f          %10.6f\n', d, V, Vmc);
    else
        fprintf('%5.1f   %10.6f          (no cube in fractional d)\n', d, V);
    end
end

% peak of V_d over real d
dpk = fminbnd(@(d) -pi^(d/2)/gamma(d/2+1), 1, 10);
fprintf('unit-ball volume peaks at d = %.4f\n', dpk);

% complex inner product <u,v> = sum conj(u) .* v
u = [1+2i; 3-1i];
v = [2i; 1];
ip_loop = 0;
for k = 1:numel(u)
    ip_loop = ip_loop + conj(u(k)) * v(k);
end
ip_ctranspose = u' * v;      % ' = CONJUGATE transpose  -> correct inner product
ip_transpose  = u.' * v;     % .' = plain transpose     -> NOT an inner product
fprintf('loop: %s   u''*v: %s   u.''*v: %s\n', num2str(ip_loop), num2str(ip_ctranspose), num2str(ip_transpose));

assert(abs(dpk - 5.2569) < 1e-3);
assert(ip_loop == ip_ctranspose && ip_loop == 7+3i && ip_transpose == -1+1i);
disp('MATLAB checks passed');
