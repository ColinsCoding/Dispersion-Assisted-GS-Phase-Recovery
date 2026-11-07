% circuit_rules_check.m -- independent MATLAB cross-check of dgs/circuit_rules.py,
% dgs/gates_to_internet.py and dgs/qubit_scope.py (numbers only, no toolboxes).
%
% In plain words: the Python notebook derives circuit rules symbolically. Here the same facts are
% re-computed numerically with plain matrices, so a disagreement would point at a bug in one of them.
%   1. Modified Nodal Analysis at s = j*omega for a transformer with load: reflected impedance and
%      the k = 1 turns ratio.
%   2. Series RLC: eigenvalues of the state matrix vs omega0 = 1/sqrt(LC), Q = sqrt(L/C)/R, and the
%      RC 50 % delay = RC ln 2 from expm.
%   3. CRC-32 bit by bit (shift register + XOR) -> the standard check value 0xCBF43926.
%   4. Manchester coding round trip.
%   5. Qubit Rabi formula vs expm of the 2 x 2 Hamiltonian.
% Run headless:  matlab -batch "run('learn/matlab/circuit_rules_check.m')"

fprintf('circuit_rules_check (MATLAB %s)\n', version);
pass = true;

% ---- 1. transformer via MNA: unknowns [v_p, v_q, i_V, i_L1, i_L2]
L1 = 2e-3; L2 = 0.5e-3; k = 0.9; RL = 10; w = 2*pi*10e3; Vs = 1;
M = k*sqrt(L1*L2); s = 1j*w;
A = [0 0 1 1 0;            % KCL at p: i_V + i_L1 = 0
     0 1/RL 0 0 1;         % KCL at q: v_q/RL + i_L2 = 0
     1 0 0 0 0;            % v_p = Vs
     1 0 0 -s*L1 -s*M;     % v_p = s L1 i1 + s M i2
     0 1 0 -s*M -s*L2];    % v_q = s M i1 + s L2 i2
x = A \ [0; 0; Vs; 0; 0];
Zin = Vs / (-x(3));
Zin_formula = s*L1 - s^2*M^2/(s*L2 + RL);
fprintf('reflected impedance  MNA %.6f%+.6fj  formula %.6f%+.6fj\n', real(Zin), imag(Zin), real(Zin_formula), imag(Zin_formula));
pass = pass && abs(Zin - Zin_formula) < 1e-9*abs(Zin);
M1 = sqrt(L1*L2);
A1 = A; A1(4,5) = -s*M1; A1(5,4) = -s*M1;
x1 = A1 \ [0; 0; Vs; 0; 0];
fprintf('k = 1 voltage ratio  %.12f  sqrt(L2/L1) = %.12f\n', abs(x1(2)/Vs), sqrt(L2/L1));
pass = pass && abs(x1(2)/Vs - sqrt(L2/L1)) < 1e-9;

% ---- 2. series RLC and RC delay
R = 20; L = 1e-3; C = 100e-9;
S = [0 1/C; -1/L -R/L];              % state [v_C; i_L]
lam = eig(S);
w0 = sqrt(prod(lam)); Q = w0 / (-sum(lam));
fprintf('series RLC  omega0 %.6g (1/sqrt(LC) %.6g)  Q %.6f (sqrt(L/C)/R %.6f)\n', w0, 1/sqrt(L*C), Q, sqrt(L/C)/R);
pass = pass && abs(w0 - 1/sqrt(L*C)) < 1e-6*w0 && abs(Q - sqrt(L/C)/R) < 1e-9;
Rc = 1e3; Cc = 10e-12; t50 = Rc*Cc*log(2);
v = 1 - expm(-t50/(Rc*Cc));
fprintf('RC 50%% delay %.6g s gives v = %.12f\n', t50, v);
pass = pass && abs(v - 0.5) < 1e-12;

% ---- 3. CRC-32
crc = crc32_bitwise(uint8('123456789'));
fprintf('CRC-32("123456789") = 0x%08X (expected 0xCBF43926)\n', crc);
pass = pass && crc == hex2dec('CBF43926');

% ---- 4. Manchester (IEEE 802.3: 0 -> high,low ; 1 -> low,high)
bits = [1 0 1 1 0 0 1 0];
chips = reshape([2*bits-1; 1-2*bits], 1, []) * -1;      % 1 -> [-1 +1], 0 -> [+1 -1]
decoded = chips(2:2:end) > chips(1:2:end);
fprintf('Manchester round trip ok: %d, DC sum %d\n', isequal(decoded, logical(bits)), sum(chips));
pass = pass && isequal(decoded, logical(bits)) && sum(chips) == 0;

% ---- 5. Rabi
Om = 2*pi*1e6; De = 2*pi*0.3e6; t = linspace(0, 3e-6, 31);
X = [0 1; 1 0]; Z = [1 0; 0 -1];
H = 0.5*Om*X - 0.5*De*Z;
P = zeros(size(t));
for n = 1:numel(t)
    psi = expm(-1j*H*t(n)) * [1; 0];
    P(n) = abs(psi(2))^2;
end
W = sqrt(Om^2 + De^2);
Pf = (Om/W)^2 * sin(W*t/2).^2;
fprintf('Rabi max |expm - formula| = %.2e\n', max(abs(P - Pf)));
pass = pass && max(abs(P - Pf)) < 1e-10;

if pass
    fprintf('ALL MATLAB CHECKS PASS\n');
else
    fprintf('SOME MATLAB CHECKS FAILED\n');
end

function crc = crc32_bitwise(data)
    crc = uint32(hex2dec('FFFFFFFF'));
    poly = uint32(hex2dec('EDB88320'));
    for b = data
        crc = bitxor(crc, uint32(b));
        for j = 1:8
            if bitand(crc, 1)
                crc = bitxor(bitshift(crc, -1), poly);
            else
                crc = bitshift(crc, -1);
            end
        end
    end
    crc = bitxor(crc, uint32(hex2dec('FFFFFFFF')));
end
